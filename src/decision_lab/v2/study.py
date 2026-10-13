import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_model

from decision_lab.data.schema import Label
from decision_lab.data.validate import read_examples
from decision_lab.evaluation.export import export_release
from decision_lab.evaluation.metrics import abstention_thresholds, fit_temperature
from decision_lab.evaluation.study import scored, store_predictions
from decision_lab.model_config import PilotConfig, ScratchConfig
from decision_lab.publication import atomic_json
from decision_lab.release import verify_release
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.config import TrainingConfig
from decision_lab.training.core import (
    build_model,
    encode_rows,
    epoch_order,
    get_tokenizer,
    make_scheduler,
    predict,
    seed_everything,
    synchronize,
    update,
)
from decision_lab.training.protocol import json_hash
from decision_lab.training.runner import verify_inputs
from decision_lab.v2.data import ARMS, TESTS
from decision_lab.v2.protocol import settings, verify_comparison


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()


def run_pilot(root: Path):
    _, pilot, scratch, training, _ = settings(Path("/root/configs"))
    torch.set_num_threads(pilot.cpu_threads)
    results = {}
    for arm in ARMS:
        seed_everything(training.seed, training.deterministic)
        tokenizer = get_tokenizer("head", pilot, root / arm / "audit")
        rows = read_examples(root / arm / "data/train.jsonl")
        sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
        model, optimizer = build_model("head", pilot, scratch, training, "cuda")
        initial_head = hashlib.sha256(
            b"".join(
                parameter.detach().cpu().numpy().tobytes() for parameter in model.head.parameters()
            )
        ).hexdigest()
        scheduler = make_scheduler(optimizer, 1878, training.warmup_fraction)
        order = epoch_order(len(rows), training.seed, 0)
        records = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(20):
            indices = order[step * 32 : (step + 1) * 32]
            synchronize("cuda")
            started = time.perf_counter()
            record = update(
                model,
                optimizer,
                scheduler,
                [sequences[i] for i in indices],
                [labels[i] for i in indices],
                training.pretrained_microbatch,
                tokenizer.pad_token_id,
                training.gradient_clip,
                "cuda",
            )
            synchronize("cuda")
            record["seconds"] = time.perf_counter() - started
            records.append(record)
        results[arm] = {
            "initial_head_sha256": initial_head,
            "seconds_per_update": sum(r["seconds"] for r in records[4:]) / 16,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "updates": records,
            "pilot_weights_discarded": True,
        }
        print(
            json.dumps(
                {"pilot_arm": arm, **{k: v for k, v in results[arm].items() if k != "updates"}}
            ),
            flush=True,
        )
        del model, optimizer, scheduler
        release_memory()
    if results["control"]["initial_head_sha256"] != results["treatment"]["initial_head_sha256"]:
        raise ValueError("Matched arms do not initialize the same head")
    return {"arms": results, "heldout_predictions_inspected": False}


def load_head(root, arm, protocol, device):
    pilot = PilotConfig.model_validate(protocol["pilot"])
    scratch = ScratchConfig.model_validate(protocol["scratch"])
    training = TrainingConfig.model_validate(protocol["training"])
    state = json.loads((root / arm / "head-state.json").read_text())
    if not state["complete"] or state["protocol_identity"] != protocol["identity"]:
        raise ValueError("Arm training has not completed its declared protocol")
    weights = Path(state["selected"]) / "model.safetensors"
    if not weights.is_relative_to(root / arm / "head"):
        raise ValueError("Selected weights are outside the arm")
    seed_everything(training.seed, training.deterministic)
    model, optimizer = build_model("head", pilot, scratch, training, device)
    del optimizer
    load_model(model, weights, strict=True)
    tokenizer = get_tokenizer("head", pilot, root / arm / "audit")
    return model, tokenizer, weights, pilot, scratch, training


def evaluate(root: Path):
    comparison = verify_comparison(root)
    _, _, _, _, evaluation = settings(Path("/root/configs"))
    torch.set_num_threads(2)
    protocols = {}
    for arm in ARMS:
        protocol = json.loads((root / arm / "protocol.json").read_text())
        verify_inputs(root / arm, protocol)
        protocols[arm] = protocol
    frozen_path = root / "calibration-freeze.json"
    if frozen_path.exists():
        raise FileExistsError("V2 calibration/evaluation is already frozen")
    calibrations = {}
    for arm in ARMS:
        model, tokenizer, weights, pilot, _, training = load_head(root, arm, protocols[arm], "cuda")
        rows = read_examples(root / arm / "data/calibration.jsonl")
        sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
        predictions = predict(
            model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
        )
        calibration = fit_temperature(predictions["logits"], labels, evaluation)
        calibration |= {
            "weights": str(weights),
            "weights_sha256": file_hash(weights),
            "thresholds": abstention_thresholds(
                predictions["logits"], calibration["temperature"], evaluation.coverage_targets
            ),
        }
        calibrations[arm] = calibration
        del model
        release_memory()
    frozen = {"comparison_identity": comparison["identity"], "arms": calibrations}
    frozen["identity"] = json_hash(frozen)
    atomic_json(frozen_path, frozen)
    atomic_json(root / "test-evaluation-started.json", {"calibration_identity": frozen["identity"]})
    results = {
        "comparison_identity": comparison["identity"],
        "calibration_identity": frozen["identity"],
        "arms": {},
    }
    for arm in ARMS:
        model, tokenizer, weights, pilot, scratch, training = load_head(
            root, arm, protocols[arm], "cuda"
        )
        if file_hash(weights) != calibrations[arm]["weights_sha256"]:
            raise ValueError("Selected weights changed after calibration")
        result = {"splits": {}, "parameters": sum(p.numel() for p in model.parameters())}
        for split in TESTS:
            rows = read_examples(root / arm / "data" / f"{split}.jsonl")
            sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
            predictions = predict(
                model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
            )
            metric = scored(predictions, calibrations[arm], evaluation)
            unknown = [i for i, row in enumerate(rows) if row.label == Label.UNKNOWN]
            metric["unknown_accuracy"] = (
                (predictions["logits"][unknown].argmax(-1) == 2).double().mean().item()
            )
            by_family = {}
            for i, row in enumerate(rows):
                by_family.setdefault(row.family_id, []).append(
                    int(predictions["logits"][i].argmax()) == labels[i]
                )
            metric["all_questions_in_family_correct"] = sum(
                all(v) for v in by_family.values()
            ) / len(by_family)
            result["splits"][split] = metric
            store_predictions(
                root / "predictions" / arm / f"{split}.jsonl", rows, predictions, calibrations[arm]
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "split": split,
                        "accuracy": metric["raw"]["accuracy"],
                        "unknown_accuracy": metric["unknown_accuracy"],
                    }
                ),
                flush=True,
            )
        model.float().eval()
        rows = read_examples(root / arm / "data/development.jsonl")
        directory = export_release(
            root / arm,
            "head",
            model,
            tokenizer,
            rows,
            pilot,
            scratch,
            protocols[arm],
            calibrations[arm],
            license_directory=Path("/root"),
        )
        card = directory / "README.md"
        card.write_text(
            card.read_text() + f"\nThis is the **v2 {arm}** arm. "
            "It compares original-style queries against changed-premise training "
            "on matched parent theories. The v1 counterfactual failure is motivation, "
            "not a claim that this release resolves it.\n"
        )
        atomic_json(directory / "evaluation.json", result)
        card.write_text(
            card.read_text()
            + "\n## Held-out results\n\n"
            + "| Split | Accuracy | UNKNOWN accuracy | All-family correct |\n"
            + "| --- | ---: | ---: | ---: |\n"
            + "\n".join(
                f"| {split} | {metric['raw']['accuracy']:.2%} | "
                f"{metric['unknown_accuracy']:.2%} | "
                f"{metric['all_questions_in_family_correct']:.2%} |"
                for split, metric in result["splits"].items()
            )
            + "\n"
        )
        atomic_json(
            directory / "artifact-manifest.json",
            {
                "protocol_identity": protocols[arm]["identity"],
                "files": {
                    str(path.relative_to(directory)): file_hash(path)
                    for path in sorted(directory.rglob("*"))
                    if path.is_file()
                },
            },
        )
        results["arms"][arm] = result
        del model
        release_memory()
    atomic_json(root / "evaluation.json", results)
    return {"calibration": frozen, "evaluation": results}


def verify_exports(root: Path):
    verify_comparison(root)
    torch.set_num_threads(2)
    return {arm: verify_release(root / arm / "releases/head", "cuda") for arm in ARMS}
