import gc
import json
import math
import shutil
import time
from pathlib import Path

import torch

from decision_lab.data.validate import read_examples
from decision_lab.evaluation.export import export_release
from decision_lab.evaluation.metrics import scores
from decision_lab.model_config import load_pilot_config, load_scratch_config
from decision_lab.publication import atomic_json
from decision_lab.release import load_release, verify_release
from decision_lab.training.checkpoint import file_hash, load_checkpoint, save_checkpoint
from decision_lab.training.core import (
    encode_rows,
    epoch_order,
    make_scheduler,
    predict,
    seed_everything,
    synchronize,
    update,
)
from decision_lab.v3.protocol import check_seal, seal
from decision_lab.v3.selection import fit_temperature, select_candidates
from decision_lab.v3.study import metrics
from v7_lab.data import TESTS
from v7_lab.protocol import verify


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score(root, phase, name, split, model, tokenizer, spec, device="cuda"):
    path = root / "predictions" / name / f"{split}.jsonl"
    if path.exists():
        raise FileExistsError("Refusing to overwrite recorded predictions")
    rows = read_examples(root / phase / f"{split}.jsonl")
    sequences, labels = encode_rows(rows, tokenizer, spec["class_labels"], spec["max_length"])
    result = predict(
        model, sequences, labels, spec["evaluation_batch"], tokenizer.pad_token_id, device
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
                }
            )
            + "\n"
            for row, label, logits in zip(rows, labels, result["logits"].tolist(), strict=True)
        )
    )
    print(json.dumps({"candidate": name, "scored": split, "questions": len(rows)}), flush=True)
    return rows, result


def configure(spec):
    torch.set_num_threads(spec["cpu_threads"])
    seed_everything(spec["model_seed"], True)


def optimizer_for(model, spec):
    model.backbone.gradient_checkpointing_enable()
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": spec["backbone_lr"]},
            {"params": model.head.parameters(), "lr": spec["head_lr"]},
        ],
        weight_decay=spec["weight_decay"],
    )


def pilot(root, commit, device="cuda"):
    protocol = verify(root)
    spec = protocol["settings"]
    configure(spec)
    reference = verify_release(root / "parent", device)
    timings = {}
    for arm in ("control", "treatment"):
        model, tokenizer, _ = load_release(root / "parent", device)
        optimizer = optimizer_for(model, spec)
        scheduler = make_scheduler(optimizer, spec["candidate_steps"][-1], spec["warmup_fraction"])
        rows = read_examples(root / "data" / f"train_{arm}.jsonl")
        sequences, labels = encode_rows(rows, tokenizer, spec["class_labels"], spec["max_length"])
        indices = sorted(range(len(rows)), key=lambda i: len(sequences[i]), reverse=True)[
            : spec["effective_batch"]
        ]
        records = []
        for _ in range(20):
            synchronize(device)
            started = time.perf_counter()
            record = update(
                model,
                optimizer,
                scheduler,
                [sequences[i] for i in indices],
                [labels[i] for i in indices],
                spec["microbatch"],
                tokenizer.pad_token_id,
                spec["gradient_clip"],
                device,
            )
            synchronize(device)
            records.append(record | {"seconds": time.perf_counter() - started})
        timings[arm] = {
            "updates": records,
            "seconds_per_update": sum(r["seconds"] for r in records[4:]) / 16,
            "longest_batch_tokens": max(len(sequences[i]) for i in indices),
        }
        del model, optimizer, scheduler
        release_memory()
    result = {
        "protocol_identity": protocol["identity"],
        "timings": timings,
        "parent_reference_check": reference,
        "weights_discarded": True,
        "final_tests_inspected": False,
    }
    atomic_json(root / "pilot.json", result)
    commit()
    return result


def package_candidate(root, arm, step, presentations, checkpoint):
    name = f"{arm}-step{step}"
    target = root / "candidates" / name
    target.mkdir(parents=True, exist_ok=False)
    for source in (root / "parent").iterdir():
        if source.is_file() and source.name not in (
            "model.safetensors",
            "calibration.json",
            "artifact-manifest.json",
        ):
            shutil.copyfile(source, target / source.name)
    shutil.copyfile(checkpoint / "model.safetensors", target / "model.safetensors")
    metadata = {
        "arm": arm,
        "step": step,
        "presentations": presentations,
        "weights_sha256": file_hash(target / "model.safetensors"),
    }
    atomic_json(root / "candidate-records" / f"{name}.json", metadata)


def train_chunk(root, arm, commit, device="cuda"):
    if arm not in ("control", "treatment"):
        raise ValueError("Unknown training arm")
    protocol = verify(root)
    spec = protocol["settings"]
    configure(spec)
    rows = read_examples(root / "data" / f"train_{arm}.jsonl")
    total_steps = math.ceil(len(rows) / spec["effective_batch"]) * spec["epochs"]
    model, tokenizer, _ = load_release(root / "parent", device)
    sequences, labels = encode_rows(rows, tokenizer, spec["class_labels"], spec["max_length"])
    optimizer = optimizer_for(model, spec)
    scheduler = make_scheduler(optimizer, total_steps, spec["warmup_fraction"])
    state_path = root / f"state-{arm}.json"
    progress = {
        "arm": arm,
        "step": 0,
        "epoch": 0,
        "cursor": 0,
        "presentations": 0,
        "total_steps": total_steps,
        "complete": False,
        "updates": [],
    }
    previous = None
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["complete"]:
            raise ValueError("Training arm is already complete")
        previous = root / state["resume"]
        progress = load_checkpoint(
            previous, model, optimizer, scheduler, state["checkpoint_protocol_identity"]
        )
    start_step = progress["step"]
    started = time.perf_counter()

    def save_progress():
        nonlocal previous
        step = progress["step"]
        checkpoint = root / "recovery" / arm / f"step-{step}"
        save_checkpoint(checkpoint, model, optimizer, scheduler, progress, protocol["identity"])
        if step in spec["candidate_steps"]:
            package_candidate(root, arm, step, progress["presentations"], checkpoint)
        atomic_json(
            state_path,
            {k: v for k, v in progress.items() if k != "updates"}
            | {
                "resume": str(checkpoint.relative_to(root)),
                "protocol_identity": protocol["identity"],
                "checkpoint_protocol_identity": protocol["identity"],
            },
        )
        atomic_json(root / f"trajectory-{arm}.json", progress)
        commit()
        if previous is not None and previous != checkpoint:
            shutil.rmtree(previous)
        previous = checkpoint
        commit()

    while progress["step"] < total_steps and progress["step"] - start_step < spec["chunk_steps"]:
        order = epoch_order(len(rows), spec["model_seed"], progress["epoch"])
        indices = order[progress["cursor"] : progress["cursor"] + spec["effective_batch"]]
        synchronize(device)
        tick = time.perf_counter()
        record = update(
            model,
            optimizer,
            scheduler,
            [sequences[i] for i in indices],
            [labels[i] for i in indices],
            spec["microbatch"],
            tokenizer.pad_token_id,
            spec["gradient_clip"],
            device,
        )
        synchronize(device)
        progress["step"] += 1
        progress["presentations"] += len(indices)
        progress["cursor"] += len(indices)
        progress["updates"].append(
            record
            | {
                "step": progress["step"],
                "examples": len(indices),
                "seconds": time.perf_counter() - tick,
            }
        )
        if progress["cursor"] == len(rows):
            progress["epoch"] += 1
            progress["cursor"] = 0
        progress["complete"] = progress["step"] == total_steps
        if progress["step"] in spec["candidate_steps"]:
            save_progress()
        if progress["step"] % 100 == 0 or progress["complete"]:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": progress["step"],
                        "loss": record["loss"],
                        "elapsed": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        if time.perf_counter() - started > spec["training_timeout_seconds"] - 100:
            break
    if previous is None or previous.name != f"step-{progress['step']}":
        save_progress()
    return {k: v for k, v in progress.items() if k != "updates"} | {
        "body_seconds": time.perf_counter() - started,
        "start_step": start_step,
    }


def candidate_pool(root, protocol):
    pool = {}
    for arm in ("control", "treatment"):
        state = json.loads((root / f"state-{arm}.json").read_text())
        if not state["complete"] or state["protocol_identity"] != protocol["identity"]:
            raise ValueError("Both arms must complete before selection")
        for step in protocol["settings"]["candidate_steps"]:
            name = f"{arm}-step{step}"
            if step == 0:
                pool[name] = {
                    "arm": arm,
                    "step": 0,
                    "presentations": 0,
                    "weights_sha256": protocol["parent_files"]["model.safetensors"],
                    "directory": "parent",
                }
            else:
                record = json.loads((root / "candidate-records" / f"{name}.json").read_text())
                if (
                    file_hash(root / "candidates" / name / "model.safetensors")
                    != record["weights_sha256"]
                ):
                    raise ValueError("Candidate weight checksum changed")
                pool[name] = record | {"directory": f"candidates/{name}"}
    return pool


def select(root, commit, device="cuda"):
    protocol = verify(root)
    if (root / "selection-freeze.json").exists():
        raise FileExistsError("Selection is already frozen")
    spec = protocol["settings"]
    configure(spec)
    pool = candidate_pool(root, protocol)
    fitted, selection_scores, prediction_files = {}, {}, {}
    for name, candidate in pool.items():
        model, tokenizer, _ = load_release(root / candidate["directory"], device)
        _, fit = score(root, "data", name, "selection_fit", model, tokenizer, spec, device)
        fitted[name] = fit_temperature(fit["logits"], fit["labels"], spec)
        _, result = score(root, "data", name, "selection", model, tokenizer, spec, device)
        raw = scores(result["logits"], result["labels"], 1, spec["ece_bins"])
        calibrated = scores(
            result["logits"], result["labels"], fitted[name]["temperature"], spec["ece_bins"]
        )
        selection_scores[name] = {
            "raw_nll": raw["nll"],
            "calibrated_nll": calibrated["nll"],
            "accuracy": raw["accuracy"],
        }
        _, development = score(root, "data", name, "development", model, tokenizer, spec, device)
        selection_scores[name]["development_accuracy"] = (
            (development["logits"].argmax(-1) == development["labels"]).double().mean().item()
        )
        prediction_files[name] = {
            s: file_hash(root / "predictions" / name / f"{s}.jsonl")
            for s in ("selection_fit", "selection", "development")
        }
        del model
        release_memory()
        commit()
    choices = select_candidates(pool, selection_scores, ["calibrated_nll", "accuracy"])
    calibrations = {}
    selected = sorted({p["calibrated_nll"] for p in choices.values()})
    for name in selected:
        model, tokenizer, _ = load_release(root / pool[name]["directory"], device)
        _, result = score(root, "data", name, "calibration", model, tokenizer, spec, device)
        calibrations[name] = fit_temperature(result["logits"], result["labels"], spec) | {
            "weights": str(root / pool[name]["directory"] / "model.safetensors"),
            "weights_sha256": pool[name]["weights_sha256"],
        }
        prediction_files[name]["calibration"] = file_hash(
            root / "predictions" / name / "calibration.jsonl"
        )
        del model
        release_memory()
        commit()
    frozen = seal(
        {
            "protocol_identity": protocol["identity"],
            "candidates": pool,
            "selection_scores": selection_scores,
            "selection_fit_temperatures": fitted,
            "choices": choices,
            "calibrations": calibrations,
            "prediction_files": prediction_files,
            "test_predictions_inspected": False,
        }
    )
    atomic_json(root / "selection-freeze.json", frozen)
    commit()
    return frozen


def evaluate(root, commit, config_directory=Path("configs"), device="cuda"):
    protocol = verify(root)
    spec = protocol["settings"]
    configure(spec)
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    if frozen["protocol_identity"] != protocol["identity"]:
        raise ValueError("Selection belongs to another protocol")
    for name, files in frozen["prediction_files"].items():
        for split, digest in files.items():
            if file_hash(root / "predictions" / name / f"{split}.jsonl") != digest:
                raise ValueError("Selection predictions changed")
    if (root / "test-started.json").exists():
        raise FileExistsError("Final test scoring already started")
    atomic_json(root / "test-started.json", {"selection_identity": frozen["identity"]})
    commit()
    results, checks, baseline = {}, {}, {}
    model, tokenizer, _ = load_release(root / "parent", device)
    parent_temperature = json.loads((root / "parent/calibration.json").read_text())["temperature"]
    for split in ("test_multi", "test_original", "test_counterfactual"):
        rows, predicted = score(root, "data", "parent", split, model, tokenizer, spec, device)
        baseline[split] = metrics(rows, predicted, parent_temperature, spec)
    del model
    release_memory()
    pilot_spec = load_pilot_config(config_directory / "pilot.yaml")
    scratch_spec = load_scratch_config(config_directory / "scratch.yaml")
    for arm in ("control", "treatment"):
        name = frozen["choices"][arm]["calibrated_nll"]
        candidate = frozen["candidates"][name]
        model, tokenizer, _ = load_release(root / candidate["directory"], device)
        calibration = frozen["calibrations"][name]
        if file_hash(Path(calibration["weights"])) != calibration["weights_sha256"]:
            raise ValueError("Selected weights changed after freeze")
        result = {"candidate": name, "splits": {}, "prediction_files": {}}
        for split in TESTS:
            rows, predicted = score(root, "data", arm, split, model, tokenizer, spec, device)
            result["splits"][split] = metrics(rows, predicted, calibration["temperature"], spec)
            result["prediction_files"][split] = file_hash(
                root / "predictions" / arm / f"{split}.jsonl"
            )
        model.float().eval()
        directory = export_release(
            root / "exports" / arm,
            "head",
            model,
            tokenizer,
            read_examples(root / "data/selection.jsonl"),
            pilot_spec,
            scratch_spec,
            protocol,
            calibration,
            device=device,
            license_directory=Path("/root") if device == "cuda" else Path.cwd(),
        )
        atomic_json(directory / "selection-freeze.json", frozen)
        atomic_json(directory / "evaluation.json", result)
        card = directory / "README.md"
        card.write_text(
            card.read_text() + f"\n## V7 {arm} arm\n\n"
            f"Additional fine-tuning from V4 mixed; selected update {candidate['step']}. "
            "The comparison changes the training mixture, "
            "with matched initialization and exposure. "
            "Both arms use a fresh optimizer. Control continues the old curriculum; "
            "treatment replaces half its question presentations with multi-entity worlds. "
            "Token exposure differs.\n\n"
            "| Fresh test | Accuracy | UNKNOWN accuracy | Complete-family accuracy |\n"
            "| --- | ---: | ---: | ---: |\n"
            + "\n".join(
                f"| {s} | {m['raw']['accuracy']:.2%} | {m['unknown_accuracy']:.2%} | "
                f"{m['all_questions_in_family_correct']:.2%} |"
                for s, m in result["splits"].items()
            )
            + "\n"
        )
        atomic_json(
            directory / "artifact-manifest.json",
            {
                "protocol_identity": protocol["identity"],
                "files": {
                    str(p.relative_to(directory)): file_hash(p)
                    for p in sorted(directory.rglob("*"))
                    if p.is_file() and p.name != "artifact-manifest.json"
                },
            },
        )
        del model
        release_memory()
        checks[arm] = verify_release(directory, device)
        results[arm] = result
        commit()
    output = {
        "protocol_identity": protocol["identity"],
        "selection_identity": frozen["identity"],
        "arms": results,
        "parent": baseline,
    }
    atomic_json(root / "evaluation.json", output)
    atomic_json(root / "artifact-verification.json", checks)
    commit()
    return {"evaluation": output, "verification": checks}
