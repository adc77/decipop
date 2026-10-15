import json
import math
from pathlib import Path

from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json, release_inventory
from decision_lab.studies.v3.protocol import check_seal
from decision_lab.studies.v4.protocol import verify
from decision_lab.training.checkpoint import file_hash

ROOT = Path("runs/study-v4")


def check_predictions(path, inputs):
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    if len(predictions) != len(inputs):
        raise ValueError(f"Prediction count mismatch: {path}")
    ids = set()
    for row, actual in zip(inputs, predictions, strict=True):
        if (
            actual["example_id"] != row.example_id
            or actual["family_id"] != row.family_id
            or actual["label"] != ["TRUE", "FALSE", "UNKNOWN"].index(row.label.value)
        ):
            raise ValueError(f"Prediction/input mismatch: {path}")
        if len(actual["logits"]) != 3 or not all(math.isfinite(v) for v in actual["logits"]):
            raise ValueError(f"Invalid logits: {path}")
        if actual["example_id"] in ids:
            raise ValueError(f"Duplicate prediction: {path}")
        ids.add(actual["example_id"])
    return len(predictions)


def main():
    diagnostic = verify(ROOT, "diagnostic")
    protocol = verify(ROOT, "training")
    frozen = check_seal(json.loads((ROOT / "selection-freeze.json").read_text()))
    if frozen["protocol_identity"] != protocol["identity"]:
        raise ValueError("Selection identity mismatch")
    if (
        file_hash(Path("src/decision_lab/studies/v4/report.py"))
        != protocol["audit"]["analysis_script_sha256"]
    ):
        raise ValueError("Frozen analysis source changed")
    if file_hash(Path("docs/V4_PROTOCOL.md")) != protocol["protocol_markdown_sha256"]:
        raise ValueError("Protocol text changed")
    if file_hash(Path("uv.lock")) != protocol["lock_sha256"]:
        raise ValueError("Dependency lock changed")
    evaluation = json.loads((ROOT / "evaluation.json").read_text())
    if (
        evaluation["protocol_identity"] != protocol["identity"]
        or evaluation["selection_identity"] != frozen["identity"]
    ):
        raise ValueError("Evaluation identity mismatch")
    inputs = {
        p.stem: read_examples(p)
        for p in (ROOT / "training").glob("*.jsonl")
        if p.name != "families.jsonl"
    }
    count = check_predictions(
        ROOT / "predictions/parent/diagnostic.jsonl",
        read_examples(ROOT / "diagnostic/diagnostic.jsonl"),
    )
    for name, files in frozen["prediction_files"].items():
        for split, digest in files.items():
            path = ROOT / "predictions" / name / f"{split}.jsonl"
            if file_hash(path) != digest:
                raise ValueError("Frozen selection prediction changed")
            count += check_predictions(path, inputs[split])
    for arm in ("control", "treatment"):
        trajectory = json.loads((ROOT / f"trajectory-{arm}.json").read_text())
        if (
            not trajectory["complete"]
            or trajectory["step"] != 1878
            or trajectory["presentations"] != 60003
        ):
            raise ValueError("Training exposure is incomplete")
        for split, digest in evaluation["arms"][arm]["prediction_files"].items():
            path = ROOT / "predictions" / arm / f"{split}.jsonl"
            if file_hash(path) != digest:
                raise ValueError("Frozen test prediction changed")
            count += check_predictions(path, inputs[split])
        manifest = release_inventory(Path("artifacts/v4-releases") / arm)
        selected = frozen["choices"][arm]["calibrated_nll"]
        if (
            manifest["files"]["model.safetensors"]
            != frozen["candidates"][selected]["weights_sha256"]
        ):
            raise ValueError("Release weights do not match selected candidate")
    candidates = json.loads((ROOT / "retained-candidates.json").read_text())
    for name, record in candidates.items():
        if file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError("Retained candidate mismatch")
        directory = ROOT / frozen["candidates"][name]["directory"]
        for filename, digest in protocol["parent_files"].items():
            if filename not in ("model.safetensors", "calibration.json"):
                if file_hash(directory / filename) != digest:
                    raise ValueError("Candidate tokenizer/config differs from parent")
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    if any(r["status"] not in ("complete", "failed") for r in ledger):
        raise ValueError("Invocation ledger has an unfinished call")
    if sum(r["allocation_usd"] for r in ledger) > protocol["settings"]["compute_allocation_usd"]:
        raise ValueError("Submitted ceilings exceeded compute cap")
    historic = json.loads(Path("artifacts/v3-source-snapshot/manifest-final.json").read_text())
    for name, digest in historic["files"].items():
        if name.startswith(("src/", "tests/", "configs/")) or name in (
            "docs/V2_PROTOCOL.md",
            "docs/V3_PROTOCOL.md",
        ):
            if file_hash(Path(name)) != digest:
                raise ValueError(f"Historical scientific file changed: {name}")
    result = {
        "valid": True,
        "prediction_rows_checked": count,
        "protocol_identity": protocol["identity"],
        "diagnostic_identity": diagnostic["identity"],
        "selection_identity": frozen["identity"],
        "all_declared_candidates_retained": True,
        "historical_scientific_files_unchanged": True,
        "completed_calls": sum(r["status"] == "complete" for r in ledger),
        "failed_calls": sum(r["status"] == "failed" for r in ledger),
        "submitted_compute_ceilings_usd": sum(r["allocation_usd"] for r in ledger),
    }
    atomic_json(ROOT / "final-audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
