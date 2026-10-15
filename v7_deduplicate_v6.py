import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from v7_files import deduplicate_verified_weight
from v7_lab.protocol import ROOT


def main():
    destination = ROOT / "disk-reclamation.json"
    if destination.exists():
        raise FileExistsError(destination)
    receipt_path = Path("runs/study-v6/huggingface-release.json")
    receipts = json.loads(receipt_path.read_text())
    record = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "historical_receipt_sha256": file_hash(receipt_path),
        "free_bytes_before": shutil.disk_usage(Path.cwd()).free,
        "model_bytes_or_paths_removed": False,
        "new_download_or_inference": False,
        "bundles": {},
    }
    atomic_json(destination, record)
    for arm in ("control", "treatment"):
        receipt = receipts[arm]
        checked = receipt["remote_download_verification"]
        if not checked["verified"] or checked["inputs"] != 8 or checked["device"] != "cpu":
            raise ValueError("Historical private-Hub CPU verification is incomplete")
        fresh = Path(receipt["verified_download_directory"])
        retained = Path("artifacts/v6-releases") / arm
        manifests = [release_inventory(directory) for directory in (fresh, retained)]
        digest = manifests[0]["files"]["model.safetensors"]
        if any(m["files"]["model.safetensors"] != digest for m in manifests):
            raise ValueError("Historical verified weight copies differ")
        if any(
            file_hash(p / "artifact-manifest.json") != receipt["artifact_manifest_sha256"]
            for p in (fresh, retained)
        ):
            raise ValueError("Historical verified manifests differ")
        target, source = fresh / "model.safetensors", retained / "model.safetensors"
        already_shared = os.path.samefile(target, source)
        deduplicate_verified_weight(target, source, digest)
        for directory in (fresh, retained):
            release_inventory(directory)
        record["bundles"][arm] = {
            "fresh_directory": str(fresh),
            "retained_directory": str(retained),
            "private_repo": receipt["repo_id"],
            "private_revision": receipt["commit"],
            "weight_sha256": digest,
            "weight_bytes": source.stat().st_size,
            "already_shared": already_shared,
            "now_same_inode": os.path.samefile(target, source),
            "all_bundle_checksums_unchanged": True,
        }
        atomic_json(destination, record)
        print(json.dumps({"historical_verified_duplicate_deduplicated": arm}), flush=True)
    if file_hash(receipt_path) != record["historical_receipt_sha256"]:
        raise ValueError("Historical verification receipt changed")
    record |= {
        "complete": True,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "free_bytes_after": shutil.disk_usage(Path.cwd()).free,
    }
    atomic_json(destination, record)
    print(
        json.dumps(
            {
                "complete": True,
                "observed_free_bytes_change": record["free_bytes_after"]
                - record["free_bytes_before"],
            }
        )
    )


if __name__ == "__main__":
    main()
