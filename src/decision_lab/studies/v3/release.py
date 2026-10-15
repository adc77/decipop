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
from decision_lab.studies.v3.protocol import check_seal, settings, verify
from decision_lab.training.checkpoint import file_hash


def download(root, checks):
    protocol = verify(root)
    volume = modal.Volume.from_name(settings()["volume"])
    source = Path("artifacts/v3-releases")
    for name in checks:
        remote = f"/study-v3/exports/{name}/releases/head"
        payload = b"".join(volume.read_file(remote + "/artifact-manifest.json"))
        manifest = json.loads(payload)
        directory = source / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, digest in manifest["files"].items():
            destination = directory / filename
            if not destination.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe artifact path")
            if destination.exists() and file_hash(destination) == digest:
                continue
            if filename == "model.safetensors":
                if digest != protocol["candidates"][name]["files"][filename]:
                    raise ValueError("Exported weights differ from the frozen candidate")
                if destination.exists():
                    raise ValueError("An existing artifact weight file differs")
                os.link(root / "candidates" / name / filename, destination)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(remote + "/" + filename):
                    stream.write(chunk)
            if file_hash(temporary) != digest:
                raise ValueError("Downloaded artifact checksum mismatch")
            os.replace(temporary, destination)
        (directory / "artifact-manifest.json").write_bytes(payload)
        release_inventory(directory)
        print(json.dumps({"downloaded_v3_release": name}), flush=True)
    retained = {}
    for entry in volume.iterdir("/study-v3", recursive=True):
        if entry.type.name != "FILE":
            continue
        relative = Path(entry.path).relative_to("study-v3")
        if relative.parts[0] in {"candidates", "data", "exports"}:
            continue
        if entry.size > 64 * 1024 * 1024:
            raise ValueError("Unexpected large metadata file")
        destination = root / relative
        payload = b"".join(volume.read_file("/" + entry.path))
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"Remote and local evidence disagree: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        retained[str(relative)] = file_hash(destination)
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    evaluation = json.loads((root / "evaluation.json").read_text())
    for name in evaluation["candidates"]:
        files = (
            frozen["prediction_files"][name] | evaluation["candidates"][name]["prediction_files"]
        )
        for split, digest in files.items():
            if file_hash(root / "predictions" / name / f"{split}.jsonl") != digest:
                raise ValueError("Retained raw prediction checksum mismatch")
    atomic_json(
        root / "retention.json",
        {
            "protocol_identity": verify(root)["identity"],
            "remote_metadata_files": retained,
            "releases": {
                name: file_hash(source / name / "artifact-manifest.json") for name in checks
            },
            "candidate_inputs_preserved_locally": True,
            "release_weights_reused_from_verified_local_candidates": True,
        },
    )
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "publish", "verify"))
    args = parser.parse_args()
    root = Path("runs/study-v3")
    verify(root)
    checks = json.loads((root / "artifact-verification.json").read_text())
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    expected = {name for policies in frozen["choices"].values() for name in policies.values()}
    if set(checks) != expected or not all(check["verified"] for check in checks.values()):
        raise ValueError("Every distinct selected candidate must pass GPU artifact reload")
    source = Path("artifacts/v3-releases")
    receipt = root / "huggingface-release.json"
    if args.action == "download":
        download(root, checks)
    elif args.action == "publish":
        names = {name: f"decision-lab-v3-smollm2-360m-{name}" for name in sorted(checks)}
        publish_private_releases(source, receipt, settings()["hf_namespace"], model_names=names)
    else:
        verify_hub_downloads(receipt, Path("artifacts/v3-hub-verification"))


if __name__ == "__main__":
    main()
