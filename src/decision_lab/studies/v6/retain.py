import argparse
import json
import os
from pathlib import Path

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash

ROOT = Path("runs/study-v6")


def download(volume, remote, target, expected=None):
    if target.exists():
        if expected is not None and file_hash(target) == expected:
            return
        raise FileExistsError(f"Refusing to overwrite retained file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("wb") as stream:
        for chunk in volume.read_file(remote):
            stream.write(chunk)
    if expected is not None and file_hash(temporary) != expected:
        raise ValueError(f"Retained file checksum mismatch: {remote}")
    os.replace(temporary, target)


def records(volume):
    retained = {}
    for item in volume.iterdir("/study-v6", recursive=True):
        path = Path(item.path)
        if item.type.name != "FILE":
            continue
        if not path.is_relative_to("study-v6"):
            raise ValueError("Record is outside the owned study")
        relative = path.relative_to("study-v6")
        if relative.parts[0] in {"recovery", "exports", "parent", "data"}:
            continue
        failed_assets = relative.parts[0] == "failed-candidate-packaging"
        if path.suffix not in (".json", ".jsonl") and not failed_assets:
            continue
        target = ROOT / relative
        if target.exists():
            remote = b"".join(volume.read_file("/" + item.path))
            if target.read_bytes() != remote:
                raise ValueError(f"Local and remote records differ: {relative}")
        else:
            download(volume, "/" + item.path, target)
        retained[str(relative)] = file_hash(target)
    atomic_json(ROOT / "retained-records.json", retained)
    return retained


def candidates(volume):
    from decision_lab.studies.v6.storage import retain_candidates

    return retain_candidates(volume)


def releases(volume):
    candidates_record = json.loads((ROOT / "retained-candidates.json").read_text())
    available = {
        r["sha256"]: Path(r["path"]) for r in candidates_record.values() if r["path"] is not None
    }
    destination = Path("artifacts/v6-releases")
    output = {}
    for arm in ("control", "treatment"):
        remote = f"/study-v6/exports/{arm}/releases/head"
        payload = b"".join(volume.read_file(remote + "/artifact-manifest.json"))
        manifest = json.loads(payload)
        directory = destination / arm
        directory.mkdir(parents=True, exist_ok=True)
        for name, digest in manifest["files"].items():
            target = directory / name
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe release member")
            if name == "model.safetensors" and not target.exists():
                source = available[digest]
                os.link(source, target)
            else:
                download(volume, remote + "/" + name, target, digest)
        marker = directory / "artifact-manifest.json"
        if marker.exists() and marker.read_bytes() != payload:
            raise ValueError("Existing local release manifest differs")
        marker.write_bytes(payload)
        release_inventory(directory)
        output[arm] = {
            "directory": str(directory),
            "manifest_sha256": file_hash(marker),
            "weights_sha256": manifest["files"]["model.safetensors"],
        }
    atomic_json(ROOT / "retained-releases.json", output)
    return output


def main():
    import modal

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("records", "candidates", "releases"))
    args = parser.parse_args()
    ownership = json.loads((ROOT / "modal-ownership.json").read_text())
    volume = modal.Volume.from_name(ownership["volume"], create_if_missing=False)
    volume.hydrate()
    if volume.object_id != ownership["volume_id"]:
        raise ValueError("Owned Volume ID mismatch")
    operations = {"records": records, "candidates": candidates, "releases": releases}
    result = operations[args.stage](volume)
    print(json.dumps({"retained": args.stage, "records": len(result)}))


if __name__ == "__main__":
    main()
