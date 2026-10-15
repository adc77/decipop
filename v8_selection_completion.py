from decision_lab.evaluation.metrics import scores
from decision_lab.publication import atomic_json
from decision_lab.release import load_release
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import seal
from decision_lab.v3.selection import fit_temperature, select_candidates
from v8_lab.analysis import predictions
from v8_lab.protocol import verify
from v8_lab.study import candidate_pool, configure, release_memory, score


def finish_selection(root, cache, commit, device="cuda"):
    protocol = verify(root)
    spec = protocol["settings"]
    if cache["protocol_identity"] != protocol["identity"] or len(cache["files"]) != 30:
        raise ValueError("Cached selection belongs to another incomplete protocol")
    if (root / "selection-freeze.json").exists():
        raise FileExistsError("Selection is already frozen")
    for name, digest in cache["files"].items():
        if file_hash(root / name) != digest:
            raise ValueError("Preserved selection predictions changed")
    configure(spec)
    pool = candidate_pool(root, protocol)
    fitted, values, files = {}, {}, {}
    for name in pool:
        _, logits, labels = predictions(root, name, "selection_fit")
        fitted[name] = fit_temperature(logits.float(), labels, spec)
        _, logits, labels = predictions(root, name, "selection")
        raw = scores(logits.float(), labels, 1, spec["ece_bins"])
        calibrated = scores(logits.float(), labels, fitted[name]["temperature"], spec["ece_bins"])
        _, development, targets = predictions(root, name, "development")
        values[name] = dict(
            raw_nll=raw["nll"],
            calibrated_nll=calibrated["nll"],
            accuracy=raw["accuracy"],
            development_accuracy=(development.argmax(-1) == targets).double().mean().item(),
        )
        files[name] = {
            split: file_hash(root / "predictions" / name / f"{split}.jsonl")
            for split in ("selection_fit", "selection", "development")
        }
    choices = select_candidates(pool, values, ["calibrated_nll", "accuracy"])
    calibrations = {}
    for name in sorted({p["calibrated_nll"] for p in choices.values()}):
        model, tokenizer, _ = load_release(root / pool[name]["directory"], device)
        _, result = score(root, "data", name, "calibration", model, tokenizer, spec, device)
        calibrations[name] = fit_temperature(result["logits"], result["labels"], spec) | {
            "weights": str(root / pool[name]["directory"] / "model.safetensors"),
            "weights_sha256": pool[name]["weights_sha256"],
        }
        files[name]["calibration"] = file_hash(root / "predictions" / name / "calibration.jsonl")
        del model
        release_memory()
        commit()
    frozen = seal(
        dict(
            protocol_identity=protocol["identity"],
            candidates=pool,
            selection_scores=values,
            selection_fit_temperatures=fitted,
            choices=choices,
            calibrations=calibrations,
            prediction_files=files,
            test_predictions_inspected=False,
        )
    )
    atomic_json(root / "selection-freeze.json", frozen)
    atomic_json(
        root / "selection-continuation.json",
        dict(
            preserved_prediction_files=cache["files"],
            reused_prediction_rows=cache["prediction_rows"],
            completed_inference_repeated=False,
            calibration_newly_scored=True,
            selection_method_changed=False,
            protocol_identity=protocol["identity"],
            selection_identity=frozen["identity"],
        ),
    )
    commit()
    return frozen
