import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.studies.v7.data import TESTS
from decision_lab.studies.v7.protocol import ROOT, verify_study
from decision_lab.studies.v7.snapshot import snapshot
from decision_lab.training.checkpoint import file_hash


def main():
    closure = ROOT / "closure-receipt.json"
    if closure.exists():
        raise FileExistsError(closure)
    protocol = verify_study()
    cleanup = json.loads((ROOT / "cleanup.json").read_text())
    if not cleanup["owned_volume_absent"] or any(
        r["tasks"] or r["state"] != "stopped" for r in cleanup["apps"]
    ):
        raise ValueError("Cleanup is incomplete")
    audits = {}
    for rep in ("r1", "r2"):
        audits[rep] = json.loads((ROOT / rep / "final-audit.json").read_text())
        raw_check = json.loads((ROOT / rep / "independent-raw-counts.json").read_text())
        if not raw_check["valid"]:
            raise ValueError("Independent raw-count verification is incomplete")
        for arm, splits in raw_check["counts"].items():
            for split, record in splits.items():
                paths = {
                    "input_sha256": ROOT / rep / "data" / f"{split}.jsonl",
                    "prediction_sha256": ROOT / rep / "predictions" / arm / f"{split}.jsonl",
                }
                if any(file_hash(path) != record[key] for key, path in paths.items()):
                    raise ValueError("Independently counted raw files changed")
        releases = json.loads((ROOT / rep / "huggingface-release.json").read_text())
        if not audits[rep]["valid"] or any(
            not r["remote_download_verification"]["verified"] for r in releases.values()
        ):
            raise ValueError("Final audit or fresh private verification is incomplete")
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    failures = [r for r in ledger if r["status"] == "failed"]
    if any(r["status"] not in ("complete", "failed") for r in ledger):
        raise ValueError("Submitted call ledger is incomplete")
    if {(r["repetition"], r["name"]) for r in failures} != {("r1", "pilot")} or not any(
        r["repetition"] == "r1" and r["name"] == "pilot-recovery" and r["status"] == "complete"
        for r in ledger
    ):
        raise ValueError("The preserved pre-scoring failure/recovery pair differs")
    test_logs = [
        Path("artifacts/v7-preparation/cpu-tests-complete.log"),
        Path("artifacts/v7-preparation/cpu-tests-closing.log"),
    ]
    if any("119 passed" not in path.read_text() for path in test_logs):
        raise ValueError("Preparation or closing CPU test evidence differs")
    metrics = json.loads((ROOT / "metric-precheck.json").read_text())
    expected_checks = {
        f"{rep}/{arm}/{split}"
        for rep in ("r1", "r2")
        for arm in ("control", "treatment")
        for split in TESTS
    }
    if (
        not metrics["valid"]
        or metrics["metric_sets_checked"] != 40
        or metrics["study_identity"] != protocol["identity"]
        or set(metrics["checks"]) != expected_checks
    ):
        raise ValueError("Forty-set metric verification is incomplete")
    for name, checked in metrics["checks"].items():
        rep, arm, split = name.split("/")
        path = ROOT / rep / "predictions" / arm / f"{split}.jsonl"
        if file_hash(path) != checked["prediction_sha256"]:
            raise ValueError("Locally recomputed metric input changed")
    figure = json.loads((ROOT / "figure.json").read_text())
    if any(file_hash(Path(name)) != digest for name, digest in figure["files"].items()) or any(
        file_hash(ROOT / rep / "independent-raw-counts.json") != digest
        for rep, digest in figure["input_receipts"].items()
    ):
        raise ValueError("Reviewed figure or its verified counts changed")
    if not json.loads((ROOT / "historical-check.json").read_text())["valid"]:
        raise ValueError("Historical preservation check is incomplete")
    evidence = Path("reports/v7-evidence")
    if evidence.exists():
        raise FileExistsError(evidence)
    evidence.mkdir()
    for path in sorted(ROOT.glob("*.json")):
        if path.name != "billing-final-resources.json":
            shutil.copyfile(path, evidence / path.name)
    for rep in ("r1", "r2"):
        for path in sorted((ROOT / rep).glob("*.json")):
            shutil.copyfile(path, evidence / f"{rep}-{path.name}")
    for path in sorted(Path("artifacts/v7-preparation").glob("*.log")):
        shutil.copyfile(path, evidence / (path.name + ".txt"))
    for path in sorted(ROOT.glob("finish-*.log")):
        if "candidates" not in path.name and "publish" not in path.name:
            shutil.copyfile(path, evidence / (path.name + ".txt"))
    shutil.copyfile(
        Path("artifacts/v7-source-snapshot/manifest-before-execution.json"),
        evidence / "source-before-execution-manifest.json",
    )
    shutil.copyfile(
        Path("artifacts/v7-source-snapshot/manifest-before-pilot-recovery.json"),
        evidence / "source-before-pilot-recovery-manifest.json",
    )
    shutil.copyfile(
        Path("artifacts/v7-preparation/v7_app.before-image-fix.py"),
        evidence / "original-v7-app-before-image-fix.py.txt",
    )
    inventory = {p.name: file_hash(p) for p in sorted(evidence.iterdir())}
    atomic_json(evidence / "inventory.json", inventory)
    for name, digest in inventory.items():
        if file_hash(evidence / name) != digest:
            raise ValueError("Compact evidence checksum differs")
    archive = snapshot("final-with-reports")
    billing = json.loads((ROOT / "billing-attribution.json").read_text())
    result = {
        "closed_at_utc": datetime.now(UTC).isoformat(),
        "study_identity": protocol["identity"],
        "prediction_rows_verified": sum(a["prediction_rows_verified"] for a in audits.values()),
        "cpu_tests_passed": 119,
        "cpu_test_logs": {str(path): file_hash(path) for path in test_logs},
        "independent_raw_count_audits": 2,
        "locally_recomputed_test_metric_sets": 40,
        "retained_failed_pilot_calls": 1,
        "successful_reserved_pilot_recovery_calls": 1,
        "source_archive": archive,
        "evidence_inventory_sha256": file_hash(evidence / "inventory.json"),
        "cleanup_sha256": file_hash(ROOT / "cleanup.json"),
        "observed_owned_metered_usd": billing["owned_resource_metered_usd"],
        "billing_snapshot_utc": billing["snapshot_utc"],
        "workspace_billed_usd": billing["workspace_billed_usd"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_committed_or_pushed": False,
        "new_robustness_or_architecture_training": False,
    }
    atomic_json(closure, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
