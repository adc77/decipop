import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal
from decision_lab.v3.protocol import inventory as file_inventory
from v8_lab.data import TESTS
from v8_lab.protocol import ROOT, verify_study
from v8_snapshot import snapshot


def main():
    closure = ROOT / "closure-receipt.json"
    if closure.exists():
        raise FileExistsError(closure)
    study = verify_study()
    correction = json.loads((ROOT / "position-correction.json").read_text())
    original = Path("runs/study-v8-original")
    if file_hash(original / "cleanup.json") != correction["original_cleanup_sha256"]:
        raise ValueError("Original preparation cleanup changed")
    if (
        file_hash(Path("artifacts/v8-source-snapshot/source-before-execution.tar.gz"))
        != correction["original_source_archive_sha256"]
    ):
        raise ValueError("Original source archive changed")
    for rep in ("r1", "r2"):
        record = check_seal(json.loads((original / rep / "protocol.json").read_text()))
        for folder in ("parent", "data"):
            if file_inventory(original / rep / folder) != record[f"{folder}_files"]:
                raise ValueError("Original frozen preparation input changed")

    cleanup = json.loads((ROOT / "cleanup.json").read_text())
    if not cleanup["owned_volume_absent"] or any(
        r["tasks"] or r["state"] != "stopped" for r in cleanup["apps"]
    ):
        raise ValueError("Owned cleanup is incomplete")
    audits = {}
    for rep in ("r1", "r2"):
        root = ROOT / rep
        audits[rep] = json.loads((root / "final-audit.json").read_text())
        raw = json.loads((root / "independent-raw-counts.json").read_text())
        if not audits[rep]["valid"] or not raw["valid"]:
            raise ValueError("Numerical and independent raw audits must pass")
        for arm, splits in raw["counts"].items():
            for split, record in splits.items():
                for path, key in (
                    (root / "data" / f"{split}.jsonl", "input_sha256"),
                    (root / "predictions" / arm / f"{split}.jsonl", "prediction_sha256"),
                ):
                    if file_hash(path) != record[key]:
                        raise ValueError("Independently audited raw file changed")
        private = json.loads((root / "huggingface-release.json").read_text())
        if set(private) != {"control", "treatment"} or any(
            not r["remote_download_verification"]["verified"]
            or r["remote_download_verification"]["inputs"] != 8
            for r in private.values()
        ):
            raise ValueError("Fresh private CPU reference checks must pass")
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    if any(r["status"] not in ("complete", "failed") for r in ledger):
        raise ValueError("Call ledger is incomplete")
    for rep in ("r1", "r2"):
        failures = [r for r in ledger if r["repetition"] == rep and r["status"] == "failed"]
        recoveries = [r for r in ledger if r["repetition"] == rep and r["stage"] == "retry"]
        expected_failures = [] if rep == "r1" else ["treatment-3", "selection"]
        if [r["name"] for r in failures] != expected_failures:
            raise ValueError("Recorded authentication failure inventory differs")
        if (
            len(recoveries) != 1
            or recoveries[0]["name"] != "position-revalidation"
            or recoveries[0]["status"] != "complete"
        ):
            raise ValueError("Position revalidation evidence differs")
    from v8_permit import permit

    amendment = permit()
    recovery = json.loads((ROOT / "r2/jwt-recovery.json").read_text())
    extras = [r for r in ledger if r["stage"] == "auth-recovery"]
    if (
        len(extras) != 1
        or extras[0]["status"] != "complete"
        or extras[0]["amendment_identity"] != amendment["identity"]
    ):
        raise ValueError("The single authentication recovery did not complete")
    if (
        recovery["start_step"] != 756
        or recovery["end_step"] != 1134
        or not recovery["prior_trajectory_prefix_unchanged"]
        or not recovery["optimizer_scheduler_rng_restored"]
    ):
        raise ValueError("Exact saved-state recovery evidence differs")
    if sum(r["allocation_usd"] for r in ledger) > 8.5:
        raise ValueError("Amended compute reservation exceeded")
    continuation = json.loads((ROOT / "r2/selection-continuation.json").read_text())
    if (
        continuation["reused_prediction_rows"] != 30240
        or continuation["completed_inference_repeated"]
    ):
        raise ValueError("Selection continuation evidence differs")
    logs = [Path("artifacts/v8-cpu-complete.log"), Path("artifacts/v8-cpu-closing.log")]
    if any("126 passed" not in path.read_text() for path in logs):
        raise ValueError("Complete CPU test evidence differs")
    metrics = json.loads((ROOT / "metric-precheck.json").read_text())
    expected = {
        f"{rep}/{arm}/{split}"
        for rep in ("r1", "r2")
        for arm in ("control", "treatment")
        for split in TESTS
    }
    if not metrics["valid"] or set(metrics["checks"]) != expected:
        raise ValueError("Complete metric recomputation is required")
    for name, checked in metrics["checks"].items():
        rep, arm, split = name.split("/")
        if (
            file_hash(ROOT / rep / "predictions" / arm / f"{split}.jsonl")
            != checked["prediction_sha256"]
        ):
            raise ValueError("Metric recomputation input changed")
    figure = json.loads((ROOT / "figure.json").read_text())
    if any(file_hash(Path(name)) != digest for name, digest in figure["files"].items()):
        raise ValueError("Figure changed")
    if not json.loads((ROOT / "historical-check.json").read_text())["valid"]:
        raise ValueError("Historical verification must pass")
    evidence = Path("reports/v8-evidence")
    evidence.mkdir(exist_ok=False)
    for path in sorted(ROOT.glob("*.json")):
        if path.name != "billing-final-resources.json":
            shutil.copyfile(path, evidence / path.name)
    for rep in ("r1", "r2"):
        for path in sorted((ROOT / rep).glob("*.json")):
            shutil.copyfile(path, evidence / f"{rep}-{path.name}")
    for path in sorted(Path("artifacts").glob("v8-cpu-*.log")):
        shutil.copyfile(path, evidence / (path.name + ".txt"))
    for path in sorted(ROOT.glob("finish-*.log")):
        if "candidates" not in path.name and "publish" not in path.name:
            shutil.copyfile(path, evidence / (path.name + ".txt"))
    shutil.copyfile(
        Path("artifacts/v8-source-snapshot/manifest-before-execution.json"),
        evidence / "source-before-execution-manifest.json",
    )
    shutil.copyfile(
        Path("artifacts/v8-preparation/disk-deduplication.json"),
        evidence / "disk-deduplication.json",
    )
    shutil.copyfile(
        Path("artifacts/v8-source-snapshot/manifest-before-position-revalidation.json"),
        evidence / "source-before-position-revalidation-manifest.json",
    )
    inventory = {p.name: file_hash(p) for p in sorted(evidence.iterdir())}
    atomic_json(evidence / "inventory.json", inventory)
    for name, digest in inventory.items():
        if file_hash(evidence / name) != digest:
            raise ValueError("Compact evidence changed")
    archive = snapshot("final-with-reports")
    billing = json.loads((ROOT / "billing-attribution.json").read_text())
    result = dict(
        closed_at_utc=datetime.now(UTC).isoformat(),
        study_identity=study["identity"],
        prediction_rows_verified=sum(a["prediction_rows_verified"] for a in audits.values()),
        cpu_tests_passed=126,
        cpu_test_logs={str(p): file_hash(p) for p in logs},
        independent_raw_count_audits=2,
        locally_recomputed_test_metric_sets=len(expected),
        retained_failed_calls=sum(r["status"] == "failed" for r in ledger),
        source_archive=archive,
        evidence_inventory_sha256=file_hash(evidence / "inventory.json"),
        cleanup_sha256=file_hash(ROOT / "cleanup.json"),
        observed_owned_metered_usd=billing["owned_resource_metered_usd"],
        billing_snapshot_utc=billing["snapshot_utc"],
        workspace_billed_usd=billing["workspace_billed_usd"],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        implementation_committed_or_pushed=False,
        further_training_submitted=False,
    )
    atomic_json(closure, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
