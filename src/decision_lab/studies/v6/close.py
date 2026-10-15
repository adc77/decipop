import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.studies.v6.protocol import verify
from decision_lab.studies.v6.snapshot import snapshot
from decision_lab.training.checkpoint import file_hash


def main():
    root = Path("runs/study-v6")
    receipt = root / "closure-receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    protocol = verify(root)
    for name, field in (
        ("uv.lock", "lock_sha256"),
        ("docs/V6_PROTOCOL.md", "protocol_markdown_sha256"),
    ):
        if file_hash(Path(name)) != protocol[field]:
            raise ValueError(f"Frozen auxiliary file differs: {name}")
    audit = json.loads((root / "final-audit.json").read_text())
    cleanup = json.loads((root / "cleanup.json").read_text())
    if not audit["valid"] or not cleanup["owned_volume_absent"] or not cleanup["zero_owned_tasks"]:
        raise ValueError("Final verification/cleanup is incomplete")
    if any(r["state"] != "stopped" or r["tasks"] for r in cleanup["apps"]):
        raise ValueError("Owned app evidence differs")
    releases = json.loads((root / "huggingface-release.json").read_text())
    if any(not record["remote_download_verification"]["verified"] for record in releases.values()):
        raise ValueError("Private pinned fresh-download verification is incomplete")
    ledger = json.loads((root / "allocation-ledger.json").read_text())
    failures = [r for r in ledger if r["status"] == "failed"]
    if any(r["status"] not in ("complete", "failed") for r in ledger):
        raise ValueError("Submitted call ledger is incomplete")
    recovery = json.loads((root / "checkpoint-recovery.json").read_text())
    if (
        len(failures) != 1
        or failures[0]["name"] != "control-1"
        or recovery["optimizer_and_rng_restoration_pending"]
    ):
        raise ValueError("Failed call recovery is not verified")
    tests = root / "cpu-tests-closing.log"
    if "110 passed" not in tests.read_text():
        raise ValueError("CPU test evidence differs")
    evidence = Path("reports/v6-evidence")
    inventory = json.loads((evidence / "inventory.json").read_text())
    if set(inventory) != {p.name for p in evidence.iterdir() if p.name != "inventory.json"}:
        raise ValueError("Compact evidence inventory differs")
    for name, digest in inventory.items():
        if file_hash(evidence / name) != digest:
            raise ValueError(f"Evidence checksum differs: {name}")
    archive = snapshot("final-with-reports")
    billing = json.loads((root / "billing-attribution.json").read_text())
    result = {
        "closed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_identity": protocol["identity"],
        "prediction_rows_verified": audit["prediction_rows_verified"],
        "cpu_tests_passed": 110,
        "retained_failed_gpu_calls": 1,
        "successful_recovery_calls": 1,
        "historical_scientific_build_files_unchanged": audit[
            "historical_scientific_build_files_unchanged"
        ],
        "analysis_sha256": file_hash(root / "analysis.json"),
        "source_archive": archive,
        "evidence_inventory_sha256": file_hash(evidence / "inventory.json"),
        "cleanup_sha256": file_hash(root / "cleanup.json"),
        "private_release_receipt_sha256": file_hash(root / "huggingface-release.json"),
        "checkpoint_storage_receipt_sha256": file_hash(root / "checkpoint-storage.json"),
        "observed_owned_metered_usd": billing["owned_resource_metered_usd"],
        "billing_snapshot_utc": billing["snapshot_utc"],
        "workspace_billed_usd": billing["workspace_billed_usd"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_committed_or_pushed": False,
        "next_training_submitted": False,
    }
    atomic_json(receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
