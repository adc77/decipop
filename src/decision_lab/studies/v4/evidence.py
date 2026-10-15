import json
import shutil
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash


def main():
    root = Path("runs/study-v4")
    output = Path("reports/v4-evidence")
    if output.exists():
        raise FileExistsError("Compact evidence already exists")
    output.mkdir()
    names = (
        "diagnostic-protocol.json",
        "diagnostic-audit.json",
        "diagnostics.json",
        "pilot.json",
        "training-protocol.json",
        "training-audit.json",
        "training-preflight.json",
        "selection-freeze.json",
        "test-started.json",
        "evaluation.json",
        "statistics.json",
        "posthoc-error-breakdown.json",
        "posthoc-breakdown-verification.json",
        "posthoc-polarity.json",
        "allocation-ledger.json",
        "modal-ownership.json",
        "artifact-verification.json",
        "retained-records.json",
        "retained-candidates.json",
        "retained-releases.json",
        "worked-example-selection.json",
        "worked-example.json",
        "final-audit.json",
        "final-checks.json",
        "huggingface-release.json",
        "cleanup.json",
        "billing-attribution.json",
        "preflight.json",
    )
    for name in names:
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(root / "cleanup-initial-source.py", output / "cleanup-initial-source.txt")
    shutil.copyfile(root / "cleanup.log", output / "cleanup-initial-log.txt")
    shutil.copyfile(root / "billing-initial-source.py", output / "billing-initial-source.txt")
    shutil.copyfile(root / "billing-monthly-hourly-failure.log", output / "billing-initial-log.txt")
    for phase in ("diagnostic", "training"):
        shutil.copyfile(root / phase / "manifest.json", output / f"{phase}-data-manifest.json")
    for stage in ("before-diagnostics", "before-training"):
        source = Path("artifacts/v4-source-snapshot") / f"manifest-{stage}.json"
        shutil.copyfile(source, output / f"source-{stage}-manifest.json")
    log = (root / "cpu-tests-final.log").read_text()
    if "69 passed" not in log:
        raise ValueError("Expected final CPU test evidence is missing")
    atomic_json(
        output / "cpu-tests.json",
        {
            "tests_passed": 69,
            "raw_log_sha256": file_hash(root / "cpu-tests-final.log"),
            "summary": log.strip(),
        },
    )
    inventory = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic_json(output / "inventory.json", inventory)
    for name, digest in inventory.items():
        if file_hash(output / name) != digest:
            raise ValueError("Compact evidence checksum mismatch")
    print(json.dumps({"compact_evidence_files": len(inventory) + 1, "all_hashes_verified": True}))


if __name__ == "__main__":
    main()
