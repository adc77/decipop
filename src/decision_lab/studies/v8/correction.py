import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import modal
from modal._utils.async_utils import synchronize_api

from decision_lab.publication import atomic_json
from decision_lab.studies.v3.protocol import check_seal, inventory
from decision_lab.studies.v4.cleanup import inspect_apps
from decision_lab.training.checkpoint import file_hash


def main():
    source = Path("runs/study-v8")
    destination = Path("runs/study-v8-original")
    receipt = Path("artifacts/v8-preparation/position-correction.json")
    if destination.exists() or receipt.exists():
        raise FileExistsError("Original preparation already archived")
    ledger = json.loads((source / "allocation-ledger.json").read_text())
    if len(ledger) != 2 or any(r["stage"] != "pilot" or r["status"] != "complete" for r in ledger):
        raise ValueError("Correction requires exactly the two completed discarded pilots")
    identities = {}
    for rep in ("r1", "r2"):
        root = source / rep
        record = check_seal(json.loads((root / "protocol.json").read_text()))
        for folder in ("data", "parent"):
            if inventory(root / folder) != record[f"{folder}_files"]:
                raise ValueError("Original inputs changed")
        pilot = json.loads((root / "pilot.json").read_text())
        if pilot["protocol_identity"] != record["identity"] or not pilot["weights_discarded"]:
            raise ValueError("Original pilot evidence differs")
        identities[rep] = record["identity"]
    original = check_seal(json.loads((source / "study-freeze.json").read_text()))
    ownership = json.loads((source / "modal-ownership.json").read_text())
    ids = set(ownership["app_ids"])
    for path in source.glob("finish-*.log"):
        ids.update(
            re.findall(
                r"https://modal.com/apps/amandogra2016/main/(ap-[A-Za-z0-9]+)", path.read_text()
            )
        )
    inspected = synchronize_api(inspect_apps)(sorted(ids))
    if ownership["volume"] != "decision-model-lab-v8-runs":
        raise ValueError("Original Volume name differs")
    volume = modal.Volume.from_name(ownership["volume"], create_if_missing=False)
    volume.hydrate()
    if volume.object_id != ownership["volume_id"]:
        raise ValueError("Original Volume ID differs")
    for name in ("v8-cpu-initial.log", "v8-cpu-targeted.log", "v8-cpu-complete.log"):
        shutil.copyfile(Path("artifacts") / name, source / name)
    os.rename(source, destination)
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "volume",
            "delete",
            ownership["volume"],
            "--yes",
            "--profile",
            "amandogra2016",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    volumes = subprocess.check_output(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "volume",
            "list",
            "--json",
            "--profile",
            "amandogra2016",
        ],
        text=True,
    )
    if any(r["name"] == ownership["volume"] for r in json.loads(volumes)):
        raise ValueError("Original Volume remains")
    cleanup = {
        "original_inputs_preserved": True,
        "apps": inspected,
        "original_volume_absent": True,
        "volume_delete_stdout": result.stdout,
        "no_full_training_selection_or_final_tests_submitted": True,
    }
    atomic_json(destination / "cleanup.json", cleanup)
    record = {
        "at_utc": datetime.now(UTC).isoformat(),
        "original_study_identity": original["identity"],
        "original_protocol_identities": identities,
        "reason": "Edit type correlated with target position at three entities",
        "correction": "Rotate target independently using integer group division by three",
        "original_source_archive_sha256": file_hash(
            Path("artifacts/v8-source-snapshot/source-before-execution.tar.gz")
        ),
        "original_resource_ids": sorted(ids | {ownership["volume_id"]}),
        "original_calls": ledger,
        "original_cleanup_sha256": file_hash(destination / "cleanup.json"),
        "new_pilots_consume_reserved_retry_allowance": True,
        "further_recovery_calls_available": 0,
    }
    atomic_json(receipt, record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
