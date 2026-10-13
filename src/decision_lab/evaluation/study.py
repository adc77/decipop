import gc
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_model

from decision_lab.data.schema import Example
from decision_lab.data.validate import read_examples
from decision_lab.evaluation.config import EvaluationConfig
from decision_lab.evaluation.export import export_release, finalize_release
from decision_lab.evaluation.metrics import (
    abstention_metrics,
    abstention_thresholds,
    fit_temperature,
    scores,
)
from decision_lab.evaluation.performance import benchmark
from decision_lab.model_config import PilotConfig, ScratchConfig
from decision_lab.models.token_model import TokenDecisionModel
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.config import TrainingConfig
from decision_lab.training.core import (
    build_model,
    encode_rows,
    get_tokenizer,
    predict,
    seed_everything,
    synchronize,
)
from decision_lab.training.protocol import json_hash, verify_protocol
from decision_lab.training.runner import verify_inputs


def evaluation_code_hash():
    directory = Path(__file__).parent
    return json_hash(
        {
            name: file_hash(directory / name)
            for name in ("config.py", "metrics.py", "performance.py", "study.py", "export.py")
        }
    )


def settings(root):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Study evaluation requires a BF16-capable CUDA GPU")
    protocol = json.loads((root / "protocol.json").read_text())
    verify_protocol(protocol)
    verify_inputs(root, protocol)
    pilot = PilotConfig.model_validate(protocol["pilot"])
    scratch = ScratchConfig.model_validate(protocol["scratch"])
    training = TrainingConfig.model_validate(protocol["training"])
    evaluation = EvaluationConfig.model_validate(protocol["evaluation"])
    torch.set_num_threads(pilot.cpu_threads)
    seed_everything(training.seed, training.deterministic)
    return protocol, pilot, scratch, training, evaluation


def load_selected(root, name, pilot, scratch, training, selection):
    architecture = "token" if name == "pretrained" else name
    synchronize("cuda")
    started = time.perf_counter()
    model, optimizer = build_model(architecture, pilot, scratch, training, "cuda")
    del optimizer
    weights = None
    if name != "pretrained":
        weights = Path(selection[name]["selected"]) / "model.safetensors"
        if not weights.resolve().is_relative_to(root.resolve()):
            raise ValueError("Selected weights are outside the study")
        load_model(model, weights, strict=True)
    synchronize("cuda")
    return model, weights, time.perf_counter() - started


def decision_predictions(model, *args):
    if not isinstance(model, TokenDecisionModel):
        return predict(model, *args)
    unrestricted = []

    def record(_module, _inputs, output):
        membership = output.argmax(-1)[:, None] == model.answer_ids[None, :]
        decisions = membership.long().argmax(-1)
        unrestricted.append(torch.where(membership.any(-1), decisions, -1).cpu())

    handle = model.register_forward_hook(record)
    try:
        result = predict(model, *args)
    finally:
        handle.remove()
    result["unrestricted_prediction"] = torch.cat(unrestricted)
    return result


def calibrate_all(root: Path):
    protocol, pilot, scratch, training, evaluation = settings(root)
    output = root / "calibration-freeze.json"
    if output.exists():
        raise FileExistsError("Calibration is already frozen")
    selection = json.loads((root / "selection.json").read_text())
    rows = read_examples(root / "data/calibration.jsonl")
    results = {}
    for name in ("pretrained", "token", "head", "scratch"):
        architecture = "token" if name == "pretrained" else name
        tokenizer = get_tokenizer(architecture, pilot, root / "audit")
        sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
        model, weights, load_seconds = load_selected(
            root, name, pilot, scratch, training, selection
        )
        predictions = decision_predictions(
            model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
        )
        fit = fit_temperature(predictions["logits"], labels, evaluation)
        fit |= {
            "weights": str(weights) if weights else None,
            "weights_sha256": file_hash(weights) if weights else None,
            "base_revision": pilot.revision if name == "pretrained" else None,
            "thresholds": abstention_thresholds(
                predictions["logits"], fit["temperature"], evaluation.coverage_targets
            ),
            "load_seconds": load_seconds,
        }
        results[name] = fit
        print(json.dumps({"calibration": name, "temperature": fit["temperature"]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    frozen = {
        "evaluation_code_sha256": evaluation_code_hash(),
        "baseline_freeze_sha256": file_hash(root / "baseline-freeze.json"),
        "protocol_identity": protocol["identity"],
        "models": results,
        "calibration_file_sha256": protocol["input_files"]["data/calibration.jsonl"],
        "test_manifest_sha256": protocol["input_files"]["test-inputs/manifest.json"],
    }
    frozen["identity"] = json_hash(frozen)
    output.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def scored(predictions, calibration, evaluation):
    logits, labels = predictions["logits"], predictions["labels"]
    result = {
        "raw": scores(
            logits, labels, 1.0, evaluation.ece_bins, evaluation.high_confidence_threshold
        ),
        "calibrated": scores(
            logits,
            labels,
            calibration["temperature"],
            evaluation.ece_bins,
            evaluation.high_confidence_threshold,
        ),
        "abstention": abstention_metrics(
            logits, labels, calibration["temperature"], calibration["thresholds"]
        ),
    }
    if "invalid_answer" in predictions:
        result["unrestricted_accuracy"] = (
            (predictions["unrestricted_prediction"] == labels).double().mean().item()
        )
        result["unrestricted_invalid_answer_rate"] = (
            predictions["invalid_answer"].double().mean().item()
        )
        result["mean_probability_outside_answer_tokens"] = (
            predictions["outside_probability"].double().mean().item()
        )
    return result


def store_predictions(path, rows, predictions, calibration, variants=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = predictions["logits"]
    probabilities = (logits.double() / calibration["temperature"]).softmax(-1)
    with path.open("w") as stream:
        for index, row in enumerate(rows):
            value = {
                "example_id": row.example_id,
                "family_id": row.family_id,
                "label": int(predictions["labels"][index]),
                "logits": logits[index].tolist(),
                "probabilities": probabilities[index].tolist(),
                "prediction": int(logits[index].argmax()),
                "proof_depth": row.proof_depth,
                "theory_depth": row.theory_depth,
                "motif": row.motif,
            }
            if variants is not None:
                value |= {
                    "base_example_id": variants[index]["base_example_id"],
                    "label_changed": variants[index]["label_changed"],
                }
            if "invalid_answer" in predictions:
                value |= {
                    "unrestricted_prediction": int(predictions["unrestricted_prediction"][index]),
                    "invalid_answer": bool(predictions["invalid_answer"][index]),
                    "outside_probability": float(predictions["outside_probability"][index]),
                }
            stream.write(json.dumps(value, separators=(",", ":")) + "\n")


def group_metrics(rows, predictions, calibration, evaluation, sequences, vocabularies):
    output = {}
    keys = ("proof_depth", "theory_depth", "motif", "label", "input_token_length", "vocabulary")
    for key in keys:
        groups = {}
        for index, row in enumerate(rows):
            if key == "input_token_length":
                length = len(sequences[index])
                value = next(
                    f"{lower}-{upper}"
                    for lower, upper in ((1, 128), (129, 192), (193, 256), (257, 384), (385, 512))
                    if lower <= length <= upper
                )
            elif key == "vocabulary":
                value = next(
                    str(group)
                    for group, terms in enumerate(vocabularies)
                    if row.query.predicate in terms
                )
            else:
                value = getattr(row, key)
            groups.setdefault(str(value), []).append(index)
        output[key] = {}
        for group, indices in groups.items():
            metrics = scores(
                predictions["logits"][indices],
                predictions["labels"][indices],
                calibration["temperature"],
                evaluation.ece_bins,
                evaluation.high_confidence_threshold,
            )
            output[key][group] = {
                name: value
                for name, value in metrics.items()
                if name not in {"risk_coverage", "ece_bins"}
            }
    return output


def evaluate_all(root: Path):
    protocol, pilot, scratch, training, evaluation = settings(root)
    frozen = json.loads((root / "calibration-freeze.json").read_text())
    if (
        json_hash({key: value for key, value in frozen.items() if key != "identity"})
        != frozen["identity"]
    ):
        raise ValueError("Calibration freeze checksum mismatch")
    if frozen["protocol_identity"] != protocol["identity"]:
        raise ValueError("Calibration belongs to a different protocol")
    if evaluation_code_hash() != frozen["evaluation_code_sha256"]:
        raise ValueError("Evaluation implementation changed after calibration freeze")
    if file_hash(root / "baseline-freeze.json") != frozen["baseline_freeze_sha256"]:
        raise ValueError("Baseline freeze changed after calibration")
    for name in ("token", "head", "scratch"):
        if (
            file_hash(Path(frozen["models"][name]["weights"]))
            != frozen["models"][name]["weights_sha256"]
        ):
            raise ValueError("Selected checkpoint changed after calibration")
    selection = json.loads((root / "selection.json").read_text())
    for name in ("token", "head", "scratch"):
        if Path(selection[name]["selected"]) / "model.safetensors" != Path(
            frozen["models"][name]["weights"]
        ):
            raise ValueError("Checkpoint selection changed after calibration")
    if (root / "evaluation.json").exists():
        raise FileExistsError("This frozen study already has final evaluation results")
    marker = root / "test-evaluation-started.json"
    expected_marker = {
        "protocol_identity": protocol["identity"],
        "calibration_identity": frozen["identity"],
    }
    if marker.exists() and json.loads(marker.read_text()) != expected_marker:
        raise ValueError("Test evaluation was started with another freeze")
    marker.write_text(json.dumps(expected_marker, indent=2) + "\n")
    manifest = json.loads((root / "test-inputs/manifest.json").read_text())
    splits = {
        name: read_examples(root / "data" / f"{name}.jsonl")
        for name in ("test_id", "test_depth", "test_wording", "test_structure")
    }
    robust = {
        name: [json.loads(line) for line in (root / "test-inputs" / file).read_text().splitlines()]
        for name, file in ((name, name + ".jsonl") for name in manifest["transformations"])
    }
    development = read_examples(root / "data/development.jsonl")
    vocabularies = json.loads((root / "data/manifest.json").read_text())["config"]["predicates"]
    result = {
        "protocol_identity": protocol["identity"],
        "calibration_identity": frozen["identity"],
        "models": {},
        "class_labels": pilot.class_labels,
        "robustness_manifest": manifest["transformations"],
    }
    for name in ("pretrained", "token", "head", "scratch"):
        model_path = root / "evaluation" / f"{name}.json"
        if model_path.exists():
            prior = json.loads(model_path.read_text())
            if prior["calibration_identity"] != frozen["identity"]:
                raise ValueError("Partial evaluation uses a different calibration freeze")
            result["models"][name] = prior
            print(json.dumps({"evaluation_resume": name}), flush=True)
            continue
        architecture = "token" if name == "pretrained" else name
        tokenizer = get_tokenizer(architecture, pilot, root / "audit")
        torch.cuda.reset_peak_memory_stats()
        model, _, load_seconds = load_selected(root, name, pilot, scratch, training, selection)
        calibration = frozen["models"][name]
        worked_rows = development[:8]
        worked_sequences, worked_labels = encode_rows(
            worked_rows, tokenizer, pilot.class_labels, pilot.max_length
        )
        worked_predictions = decision_predictions(
            model,
            worked_sequences,
            worked_labels,
            training.evaluation_batch,
            tokenizer.pad_token_id,
            "cuda",
        )
        store_predictions(
            root / "predictions" / name / "worked_development.jsonl",
            worked_rows,
            worked_predictions,
            calibration,
        )
        model_result = {
            "calibration_identity": frozen["identity"],
            "load_seconds": load_seconds,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "splits": {},
            "robustness": {},
        }
        id_predictions = None
        for split, rows in splits.items():
            started = time.perf_counter()
            sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
            tokenize_seconds = time.perf_counter() - started
            predictions = decision_predictions(
                model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
            )
            metrics = scored(predictions, calibration, evaluation)
            metrics["groups"] = group_metrics(
                rows, predictions, calibration, evaluation, sequences, vocabularies
            )
            metrics["tokenization_seconds"] = tokenize_seconds
            model_result["splits"][split] = metrics
            store_predictions(
                root / "predictions" / name / f"{split}.jsonl", rows, predictions, calibration
            )
            if split == "test_id":
                id_predictions = {
                    row.example_id: int(predictions["logits"][index].argmax())
                    for index, row in enumerate(rows)
                }
                model_result["performance"] = benchmark(
                    model,
                    sequences,
                    labels,
                    tokenizer.pad_token_id,
                    evaluation,
                    pilot.hourly_usd,
                    manifest["performance_test_id_indices"],
                )
            print(
                json.dumps(
                    {
                        "evaluation": name,
                        "split": split,
                        "accuracy": metrics["raw"]["accuracy"],
                        "nll": metrics["raw"]["nll"],
                    }
                ),
                flush=True,
            )
        for transformation, group in robust.items():
            rows = [Example.model_validate_json(json.dumps(item["example"])) for item in group]
            sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
            predictions = decision_predictions(
                model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
            )
            metrics = scored(predictions, calibration, evaluation)
            predicted = predictions["logits"].argmax(-1).tolist()
            same = [index for index, item in enumerate(group) if not item["label_changed"]]
            changed = [index for index, item in enumerate(group) if item["label_changed"]]
            metrics["label_preserving_consistency"] = (
                sum(
                    predicted[index] == id_predictions[group[index]["base_example_id"]]
                    for index in same
                )
                / len(same)
                if same
                else None
            )
            metrics["changed_label_questions"] = len(changed)
            metrics["changed_label_accuracy"] = (
                sum(predicted[index] == labels[index] for index in changed) / len(changed)
                if changed
                else None
            )
            model_result["robustness"][transformation] = metrics
            store_predictions(
                root / "predictions" / name / f"robust_{transformation}.jsonl",
                rows,
                predictions,
                calibration,
                group,
            )
        if name != "pretrained":
            directory = export_release(
                root, name, model, tokenizer, development, pilot, scratch, protocol, calibration
            )
            finalize_release(directory, protocol["identity"], model_result)
        result["models"][name] = model_result
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(model_result, indent=2) + "\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (root / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
