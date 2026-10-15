import argparse
import json
import os
from pathlib import Path

import modal

from decision_lab.publication import (
    atomic_json,
    publish_private_releases,
    release_inventory,
    verify_hub_downloads,
)
from decision_lab.training.checkpoint import file_hash
from v8_files import deduplicate_verified_weight, download, ensure_space
from v8_lab.protocol import ROOT, verify


def owned_volume():
    ownership = json.loads((ROOT / "modal-ownership.json").read_text())
    if ownership["volume"] != "decision-model-lab-v8-position-runs":
        raise ValueError("Unexpected owned Volume")
    volume = modal.Volume.from_name(ownership["volume"], create_if_missing=False)
    volume.hydrate()
    if volume.object_id != ownership["volume_id"]:
        raise ValueError("Owned Volume identity differs")
    return volume


def records(root, volume):
    prefix = f"study-v8/{root.name}"
    retained = {}
    for entry in volume.iterdir("/" + prefix, recursive=True):
        if entry.type.name != "FILE":
            continue
        path = Path(entry.path)
        if not path.is_relative_to(prefix):
            raise ValueError("Remote record is outside its repetition")
        relative = path.relative_to(prefix)
        if relative.parts[0] in {"recovery", "exports", "parent", "data"}:
            continue
        if path.suffix not in (".json", ".jsonl"):
            continue
        target = root / relative
        if target.exists():
            if target.read_bytes() != b"".join(volume.read_file("/" + entry.path)):
                raise ValueError(f"Retained local/remote records differ: {relative}")
        else:
            download(volume, "/" + entry.path, target)
        retained[str(relative)] = file_hash(target)
    atomic_json(root / "retained-records.json", retained)
    return retained


def releases(root, volume):
    retained = json.loads((root / "retained-candidates.json").read_text())
    available = {v["sha256"]: Path(v["path"]) for v in retained.values() if v["path"] is not None}
    result = {}
    for arm in ("control", "treatment"):
        remote = f"/study-v8/{root.name}/exports/{arm}/releases/head"
        payload = b"".join(volume.read_file(remote + "/artifact-manifest.json"))
        manifest = json.loads(payload)
        destination = Path("artifacts/v8-releases") / root.name / arm
        destination.mkdir(parents=True, exist_ok=True)
        for name, digest in manifest["files"].items():
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe release member")
            if name == "model.safetensors" and not target.exists():
                os.link(available[digest], target)
            else:
                download(volume, remote + "/" + name, target, digest)
        marker = destination / "artifact-manifest.json"
        if marker.exists() and marker.read_bytes() != payload:
            raise ValueError("Retained release manifest differs")
        marker.write_bytes(payload)
        release_inventory(destination)
        result[arm] = {
            "directory": str(destination),
            "manifest_sha256": file_hash(marker),
            "weights_sha256": manifest["files"]["model.safetensors"],
        }
    atomic_json(root / "retained-releases.json", result)
    return result


def publish(root):
    spec = verify(root)["settings"]
    source = Path("artifacts/v8-releases") / root.name
    receipt = root / "huggingface-release.json"
    names = {
        arm: f"decision-lab-v8-{root.name}-smollm2-360m-{suffix}"
        for arm, suffix in (("control", "paired-ce"), ("treatment", "paired-consistency"))
    }
    records = publish_private_releases(source, receipt, spec["hf_namespace"], model_names=names)
    for arm, record in records.items():
        if "remote_download_verification" in record:
            continue
        ensure_space((source / arm / "model.safetensors").stat().st_size)
        single = root / f"verification-{arm}.json"
        if single.exists():
            raise FileExistsError("Inspect the prior verification attempt before resuming")
        atomic_json(single, {arm: record})
        checked = verify_hub_downloads(single, Path("artifacts/v8-hub-verification") / root.name)
        verified = checked[arm]
        verification = verified["remote_download_verification"]
        if not verification["verified"] or verification["inputs"] != 8:
            raise ValueError("Fresh CPU reference verification is incomplete")
        fresh = Path(verified["verified_download_directory"]) / "model.safetensors"
        retained = source / arm / "model.safetensors"
        manifest = release_inventory(source / arm)
        digest = manifest["files"]["model.safetensors"]
        deduplicate_verified_weight(fresh, retained, digest)
        verified["verified_weight_deduplication"] = {
            "after_fresh_cpu_verification": True,
            "sha256": digest,
            "hardlinked_to": str(retained.resolve()),
            "contents_unchanged": True,
        }
        records[arm] = verified
        atomic_json(receipt, records)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repetition", choices=("r1", "r2"))
    parser.add_argument("action", choices=("records", "candidates", "releases", "publish"))
    args = parser.parse_args()
    root = ROOT / args.repetition
    if args.action == "publish":
        result = publish(root)
    else:
        volume = owned_volume()
        if args.action == "candidates":
            from v8_storage import retain_candidates

            result = retain_candidates(root, volume)
        else:
            result = globals()[args.action](root, volume)
    print(
        json.dumps({"repetition": args.repetition, "action": args.action, "records": len(result)})
    )


if __name__ == "__main__":
    main()
