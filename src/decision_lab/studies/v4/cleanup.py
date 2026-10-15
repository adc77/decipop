import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal
from huggingface_hub import HfApi
from modal.client import _Client
from modal_proto import api_pb2

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash


def cli(arguments):
    return subprocess.run(
        ["uv", "run", "--extra", "cloud", "modal", *arguments, "--profile", "amandogra2016"],
        capture_output=True,
        text=True,
    )


async def verify_owned_apps(client, app_ids):
    rows = []
    for app_id in app_ids:
        response = await client._stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))
        tasks = await client._stub.TaskList(api_pb2.TaskListRequest(app_id=app_id))
        if response.lifecycle.app_state != api_pb2.APP_STATE_STOPPED or tasks.tasks:
            raise ValueError(f"Owned app cleanup is not verified: {app_id}")
        rows.append(
            {
                "app_id": app_id,
                "state": "stopped",
                "tasks": len(tasks.tasks),
                "stopped_at": response.lifecycle.stopped_at,
            }
        )
    return rows


async def inspect_apps(app_ids):
    return await verify_owned_apps(await _Client.from_env(), app_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path("runs/study-v4")
    if (root / "cleanup.json").exists():
        raise FileExistsError("Cleanup was already recorded")
    ownership = json.loads((root / "modal-ownership.json").read_text())
    audit = json.loads((root / "final-audit.json").read_text())
    receipt = json.loads((root / "huggingface-release.json").read_text())
    if not audit["valid"] or set(receipt) != {"control", "treatment"}:
        raise ValueError("Retention/audit/publication is incomplete")
    api = HfApi()
    for arm, release in receipt.items():
        verification = release["remote_download_verification"]
        if not verification["verified"] or verification["inputs"] != 8:
            raise ValueError("Independent Hub verification is incomplete")
        info = api.model_info(release["repo_id"], revision=release["commit"])
        if not info.private or info.sha != release["commit"]:
            raise ValueError("Private model revision changed")
        release_inventory(Path("artifacts/v4-releases") / arm)
    if ownership["volume"] != "decision-model-lab-v4-runs":
        raise ValueError("Unexpected owned Volume name")
    for app_id in ownership["app_ids"]:
        if not app_id.startswith("ap-"):
            raise ValueError("Invalid owned app ID")
    if args.verify_only:
        actions = {
            "mode": "read-only verification; no stop or deletion performed by this invocation",
            "earlier_action_receipts_available": False,
            "earlier_source_sha256": file_hash(root / "cleanup-initial-source.py"),
            "earlier_failed_verification_log_sha256": file_hash(root / "cleanup.log"),
        }
    else:
        volume = modal.Volume.from_name(ownership["volume"], create_if_missing=False)
        volume.hydrate()
        if volume.object_id != ownership["volume_id"]:
            raise ValueError("Owned Volume ID differs")
        stops = []
        for app_id in ownership["app_ids"]:
            result = cli(["app", "stop", app_id, "--yes"])
            message = result.stdout + result.stderr
            if result.returncode != 0 and "already stopped" not in message.lower():
                raise RuntimeError(f"Owned app stop failed: {app_id}: {message}")
            stops.append({"app_id": app_id, "exit_code": result.returncode, "message": message})
        deletion = cli(["volume", "delete", ownership["volume"], "--yes"])
        if deletion.returncode != 0:
            raise RuntimeError("Owned Volume deletion failed: " + deletion.stderr)
        actions = {"stop_receipts": stops, "volume_delete_stdout": deletion.stdout}
        atomic_json(root / "cleanup-actions.json", actions)
    volumes = cli(["volume", "list", "--json"])
    if volumes.returncode != 0:
        raise RuntimeError("Post-cleanup resource listing failed")
    volume_rows = json.loads(volumes.stdout)
    if any(row["name"] == ownership["volume"] for row in volume_rows):
        raise ValueError("Owned Volume still exists")
    owned_apps = asyncio.run(inspect_apps(ownership["app_ids"]))
    record = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "owned_volume_absent": True,
        "owned_apps": owned_apps,
        "actions": actions,
        "zero_owned_tasks": True,
        "unrelated_resources_modified": False,
        "optimizer_recovery_states_retained": False,
    }
    atomic_json(root / "cleanup.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("owned_apps", "stop_receipts")}))


if __name__ == "__main__":
    main()
