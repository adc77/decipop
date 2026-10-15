import json
import os
from pathlib import Path

import modal

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.studies.v2.protocol import verify_comparison
from decision_lab.training.checkpoint import file_hash


def read_json(volume, path):
    return json.loads(b"".join(volume.read_file(path)))


def retain_file(volume, remote, local, digest):
    if local.exists() and file_hash(local) == digest:
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(local.suffix + ".partial")
    with temporary.open("wb") as stream:
        for chunk in volume.read_file(remote):
            stream.write(chunk)
    if file_hash(temporary) != digest:
        raise ValueError(f"Retention checksum mismatch: {remote}")
    os.replace(temporary, local)


def main():
    root = Path("runs/study-v2")
    comparison = verify_comparison(root)
    receipt = json.loads((root / "huggingface-release.json").read_text())
    if set(receipt) != {"control", "treatment"}:
        raise ValueError("Both private releases are required before final retention")
    volume = modal.Volume.from_name("decision-model-lab-v2-runs")
    final_weights = {}
    for arm in receipt:
        if not receipt[arm]["remote_download_verification"]["verified"]:
            raise ValueError(f"Private Hub round trip is incomplete: {arm}")
        release_inventory(Path("artifacts/v2-releases") / arm)
        state = json.loads((root / arm / "head-state.json").read_text())
        if not state["complete"] or state["step"] != 1878 or state["examples_seen"] != 60003:
            raise ValueError(f"Training exposure is incomplete: {arm}")
        checkpoint = Path(state["checkpoint"])
        if not checkpoint.is_relative_to(Path("/runs/study-v2") / arm / "head"):
            raise ValueError("Checkpoint is outside the owned study")
        remote = "/" + str(checkpoint.relative_to("/runs"))
        marker = read_json(volume, remote + "/complete.json")
        if marker["identity"] != state["protocol_identity"] + ":head":
            raise ValueError("Retained checkpoint identity differs from training")
        destination = Path("artifacts/v2-final-weights") / arm
        for name in ("model.safetensors", "progress.json"):
            retain_file(volume, remote + "/" + name, destination / name, marker["files"][name])
        atomic_json(destination / "remote-complete.json", marker)
        final_weights[arm] = {
            "source_checkpoint": state["checkpoint"],
            "weights_sha256": marker["files"]["model.safetensors"],
            "progress_sha256": marker["files"]["progress.json"],
            "selected_release": state["selected"],
            "optimizer_and_rng_state_retained": False,
        }
        print(json.dumps({"final_weights_retained": arm}), flush=True)
    records = Path("artifacts/v2-modal-records")
    inventory = {}
    for entry in volume.iterdir("/study-v2", recursive=True):
        path = Path(entry.path)
        if entry.type.name != "FILE" or path.suffix not in {".json", ".md", ".txt", ".yaml"}:
            continue
        if not path.is_relative_to("study-v2") or entry.size > 8 * 1024 * 1024:
            raise ValueError("Unexpected metadata retention path or size")
        local = records / path.relative_to("study-v2")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"".join(volume.read_file("/" + entry.path)))
        inventory[str(local.relative_to(records))] = file_hash(local)
    result = {
        "comparison_identity": comparison["identity"],
        "final_weights": final_weights,
        "metadata_files": inventory,
        "working_volume_deleted_by_this_script": False,
    }
    atomic_json(root / "retention.json", result)
    print(json.dumps({"metadata_files_retained": len(inventory)}), flush=True)


if __name__ == "__main__":
    main()
