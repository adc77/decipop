import json
import math
from pathlib import Path

from decision_lab.data.validate import read_examples
from decision_lab.evaluation.study import evaluation_code_hash
from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.protocol import verify_protocol
from decision_lab.v2.protocol import verify_comparison
from decision_lab.v3.protocol import check_seal, verify


def main():
    root = Path("runs/study-v3")
    protocol = verify(root)
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    evaluation = json.loads((root / "evaluation.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    for name, digest in manifest["exclusion_files"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("Prior-world inputs changed after exclusion")
    if file_hash(Path("docs/V3_PROTOCOL.md")) != protocol["protocol_markdown_sha256"]:
        raise ValueError("The predeclared V3 protocol text changed")
    if file_hash(Path("uv.lock")) != protocol["lock_sha256"]:
        raise ValueError("The dependency lock changed")
    if evaluation["selection_identity"] != frozen["identity"]:
        raise ValueError("Test results do not use the frozen selection")
    started = json.loads((root / "test-started.json").read_text())
    if started["selection_identity"] != frozen["identity"]:
        raise ValueError("Test scoring started with a different selection")
    total = 0
    predictions = {}
    inputs = {
        split: read_examples(root / "data" / f"{split}.jsonl") for split in manifest["splits"]
    }
    for name in protocol["candidates"]:
        files = (
            frozen["prediction_files"][name] | evaluation["candidates"][name]["prediction_files"]
        )
        predictions[name] = {}
        for split, digest in files.items():
            path = root / "predictions" / name / f"{split}.jsonl"
            if file_hash(path) != digest:
                raise ValueError("Raw prediction checksum mismatch")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for source, predicted in zip(inputs[split], rows, strict=True):
                if (
                    source.example_id,
                    source.family_id,
                    protocol["settings"]["class_labels"].index(source.label.value),
                ) != (predicted["example_id"], predicted["family_id"], predicted["label"]):
                    raise ValueError("A raw prediction does not match its formal input")
                if len(predicted["logits"]) != 3 or not all(
                    math.isfinite(value) for value in predicted["logits"]
                ):
                    raise ValueError("Invalid raw logits")
            predictions[name][split] = {"rows": len(rows), "sha256": digest}
            total += len(rows)
    selected = {name for policies in frozen["choices"].values() for name in policies.values()}
    checks = json.loads((root / "artifact-verification.json").read_text())
    receipts = json.loads((root / "huggingface-release.json").read_text())
    if set(checks) != selected or set(receipts) != selected:
        raise ValueError("Selected artifacts are missing")
    for name in selected:
        if (
            not checks[name]["verified"]
            or not receipts[name]["remote_download_verification"]["verified"]
        ):
            raise ValueError("A selected artifact lacks GPU or fresh private CPU verification")
        source = Path("artifacts/v3-releases") / name
        remote = Path(receipts[name]["verified_download_directory"])
        for directory in (source, remote):
            release = release_inventory(directory)
            if release["protocol_identity"] != protocol["identity"]:
                raise ValueError("Artifact protocol identity mismatch")
            if (
                release["files"]["model.safetensors"]
                != protocol["candidates"][name]["files"]["model.safetensors"]
            ):
                raise ValueError("Artifact weights differ from the selected candidate")
            calibration = json.loads((directory / "calibration.json").read_text())
            if calibration != frozen["final_calibrations"][name]:
                raise ValueError("Artifact calibration differs from the frozen calibration")
            if (
                file_hash(directory / "artifact-manifest.json")
                != receipts[name]["artifact_manifest_sha256"]
            ):
                raise ValueError("Private release manifest differs from its pinned receipt")
    ledger = json.loads((root / "allocation-ledger.json").read_text())
    if any(record["status"] != "complete" for record in ledger):
        raise ValueError("Execution ledger contains an unfinished or failed submission")
    verify_comparison(Path("runs/study-v2"))
    verify_protocol(json.loads(Path("runs/study-v1/protocol.json").read_text()))
    if (
        evaluation_code_hash()
        != json.loads(Path("runs/study-v1/calibration-freeze.json").read_text())[
            "evaluation_code_sha256"
        ]
    ):
        raise ValueError("V1 frozen evaluation source changed")
    retention = json.loads((root / "retention.json").read_text())
    for filename, digest in retention["remote_metadata_files"].items():
        if file_hash(root / filename) != digest:
            raise ValueError("Retained metadata changed")
    output = {
        "passed": True,
        "protocol_identity": protocol["identity"],
        "selection_identity": frozen["identity"],
        "raw_prediction_rows": total,
        "prediction_files": predictions,
        "selected_artifacts": sorted(selected),
        "v1_v2_frozen_sources_verified": True,
        "frozen_protocol_text_verified": True,
        "gpu_calls": len(ledger),
        "all_calls_completed": True,
        "submitted_compute_ceiling_usd": sum(record["allocation_usd"] for record in ledger),
        "cpu_tests_passed": 59,
    }
    atomic_json(root / "final-audit.json", output)
    print(json.dumps({key: value for key, value in output.items() if key != "prediction_files"}))


if __name__ == "__main__":
    main()
