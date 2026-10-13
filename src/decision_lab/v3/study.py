import gc
import json
import time
from pathlib import Path

import torch

from decision_lab.data.validate import read_examples
from decision_lab.evaluation.export import export_release
from decision_lab.evaluation.metrics import scores
from decision_lab.model_config import load_pilot_config, load_scratch_config
from decision_lab.publication import atomic_json
from decision_lab.release import load_release, verify_release
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.core import encode_rows, predict
from decision_lab.v2.data import TESTS
from decision_lab.v3.protocol import check_seal, seal, verify
from decision_lab.v3.selection import fit_temperature, select_candidates


def predictions(root, name, split, model, tokenizer, spec):
    path = root / "predictions" / name / f"{split}.jsonl"
    if path.exists():
        raise FileExistsError("Refusing to overwrite recorded predictions")
    rows = read_examples(root / "data" / f"{split}.jsonl")
    sequences, labels = encode_rows(rows, tokenizer, spec["class_labels"], spec["max_length"])
    result = predict(
        model, sequences, labels, spec["evaluation_batch"], tokenizer.pad_token_id, "cuda"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "example_id": row.example_id,
                    "family_id": row.family_id,
                    "label": label,
                    "logits": logits,
                },
                separators=(",", ":"),
            )
            + "\n"
            for row, label, logits in zip(rows, labels, result["logits"].tolist(), strict=True)
        )
    )
    print(
        json.dumps({"candidate": name, "scored_split": split, "questions": len(rows)}), flush=True
    )
    return rows, result


def metrics(rows, result, temperature, spec):
    logits, labels = result["logits"], result["labels"]
    output = {}
    for stage, scale in (("raw", 1.0), ("calibrated", temperature)):
        output[stage] = scores(
            logits, labels, scale, spec["ece_bins"], spec["high_confidence_threshold"]
        )
        del output[stage]["risk_coverage"]
    decisions = logits.argmax(-1)
    unknown = labels == spec["class_labels"].index("UNKNOWN")
    grouped = {}
    for row, correct in zip(rows, (decisions == labels).tolist(), strict=True):
        grouped.setdefault(row.family_id, []).append(correct)
    output |= {
        "unknown_accuracy": (decisions[unknown] == labels[unknown]).double().mean().item(),
        "all_questions_in_family_correct": sum(all(values) for values in grouped.values())
        / len(grouped),
        "families": len(grouped),
        "predicted_class_counts": torch.bincount(decisions, minlength=3).tolist(),
    }
    return output


def selection(root, config_directory, commit):
    protocol = verify(root, config_directory)
    spec = protocol["settings"]
    if (root / "selection-freeze.json").exists() or (root / "test-started.json").exists():
        raise FileExistsError("Selection is already frozen or testing has started")
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    candidates = protocol["candidates"]
    fitted = {}
    selection_scores = {}
    calibration_predictions = {}
    prediction_files = {}
    for name in candidates:
        model, tokenizer, _ = load_release(root / "candidates" / name, "cuda")
        _, fit = predictions(root, name, "selection_fit", model, tokenizer, spec)
        fitted[name] = fit_temperature(fit["logits"], fit["labels"], spec)
        _, selected = predictions(root, name, "selection", model, tokenizer, spec)
        raw = scores(selected["logits"], selected["labels"], 1, spec["ece_bins"])
        calibrated = scores(
            selected["logits"], selected["labels"], fitted[name]["temperature"], spec["ece_bins"]
        )
        selection_scores[name] = {
            "raw_nll": raw["nll"],
            "calibrated_nll": calibrated["nll"],
            "accuracy": raw["accuracy"],
        }
        _, calibration_predictions[name] = predictions(
            root, name, "calibration", model, tokenizer, spec
        )
        prediction_files[name] = {
            split: file_hash(root / "predictions" / name / f"{split}.jsonl")
            for split in ("selection_fit", "selection", "calibration")
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        commit()
    choices = select_candidates(candidates, selection_scores, spec["policies"])
    calibrations = {}
    for name, result in calibration_predictions.items():
        calibrations[name] = fit_temperature(result["logits"], result["labels"], spec)
        calibrations[name] |= {
            "weights": str(root / "candidates" / name / "model.safetensors"),
            "weights_sha256": candidates[name]["files"]["model.safetensors"],
        }
    frozen = seal(
        {
            "protocol_identity": protocol["identity"],
            "selection_fit_temperatures": fitted,
            "selection_scores": selection_scores,
            "choices": choices,
            "final_calibrations": calibrations,
            "prediction_files": prediction_files,
            "test_predictions_inspected": False,
            "seconds": time.perf_counter() - started,
        }
    )
    atomic_json(root / "selection-freeze.json", frozen)
    commit()
    return frozen


def evaluate(root, config_directory, commit):
    protocol = verify(root, config_directory)
    spec = protocol["settings"]
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    if frozen["protocol_identity"] != protocol["identity"]:
        raise ValueError("Selection belongs to another protocol")
    for name, files in frozen["prediction_files"].items():
        for split, digest in files.items():
            if file_hash(root / "predictions" / name / f"{split}.jsonl") != digest:
                raise ValueError("Selection predictions changed after freeze")
    if (root / "test-started.json").exists():
        raise FileExistsError("V3 test scoring has already started")
    atomic_json(root / "test-started.json", {"selection_identity": frozen["identity"]})
    commit()
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    pilot = load_pilot_config(config_directory / "pilot.yaml")
    scratch = load_scratch_config(config_directory / "scratch.yaml")
    selected = {name for policies in frozen["choices"].values() for name in policies.values()}
    evaluation = {
        "protocol_identity": protocol["identity"],
        "selection_identity": frozen["identity"],
        "candidates": {},
    }
    checks = {}
    for name in protocol["candidates"]:
        model, tokenizer, _ = load_release(root / "candidates" / name, "cuda")
        calibration = frozen["final_calibrations"][name]
        result = {"splits": {}, "prediction_files": {}}
        for split in TESTS:
            rows, predicted = predictions(root, name, split, model, tokenizer, spec)
            result["splits"][split] = metrics(rows, predicted, calibration["temperature"], spec)
            result["prediction_files"][split] = file_hash(
                root / "predictions" / name / f"{split}.jsonl"
            )
        evaluation["candidates"][name] = result
        atomic_json(root / "candidate-results" / f"{name}.json", result)
        if name in selected:
            directory = export_release(
                root / "exports" / name,
                "head",
                model,
                tokenizer,
                read_examples(root / "data/selection.jsonl"),
                pilot,
                scratch,
                protocol,
                calibration,
                license_directory=Path("/root"),
            )
            atomic_json(directory / "selection-freeze.json", frozen)
            atomic_json(directory / "evaluation.json", result)
            card = [
                "---",
                "license: apache-2.0",
                "library_name: transformers",
                "---",
                "",
                f"# Decision Model Lab V3: {name}",
                "",
                "Private synthetic research artifact; custom inference loader is required.",
                "",
                f"Retained V2 checkpoint: step {protocol['candidates'][name]['step']}. "
                "V3 performs no training. Three selection rules compare "
                "two retained checkpoints per arm.",
                "",
                f"Temperature: {calibration['temperature']:.6g}; "
                "fitted on fresh calibration families.",
                "",
                "| Fresh test | Accuracy | UNKNOWN accuracy | All-family correct "
                "| Calibrated NLL |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for split, score in result["splits"].items():
                card.append(
                    f"| {split} | {score['raw']['accuracy']:.2%} | "
                    f"{score['unknown_accuracy']:.2%} | "
                    f"{score['all_questions_in_family_correct']:.2%} | "
                    f"{score['calibrated']['nll']:.4f} |"
                )
            card += [
                "",
                "Base: HuggingFaceTB/SmolLM2-360M, revision " + pilot.revision + ".",
                "Class order: TRUE, FALSE, UNKNOWN. Maximum 512 tokens. "
                "Run infer.py --input prompts.json with a JSON array of rendered prompts.",
                "",
                "Single training seed, synthetic generator, limited checkpoint pool. "
                "These results do not establish general decision ability. "
                "Calibration may fail on other distributions. "
                "See protocol.json and selection-freeze.json. "
                "The repository implementation remains uncommitted locally; "
                "source hashes are frozen.",
            ]
            (directory / "README.md").write_text("\n".join(card) + "\n")
            atomic_json(
                directory / "artifact-manifest.json",
                {
                    "protocol_identity": protocol["identity"],
                    "files": {
                        str(path.relative_to(directory)): file_hash(path)
                        for path in sorted(directory.rglob("*"))
                        if path.is_file() and path.name != "artifact-manifest.json"
                    },
                },
            )
            checks[name] = verify_release(directory, "cuda")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        commit()
    atomic_json(root / "evaluation.json", evaluation)
    atomic_json(root / "artifact-verification.json", checks)
    commit()
    return {"evaluation": evaluation, "verification": checks}
