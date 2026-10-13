import argparse
import json
import os
from pathlib import Path

import modal

from decision_lab.publication import (
    publish_private_releases,
    release_inventory,
    verify_hub_downloads,
)
from decision_lab.training.checkpoint import file_hash

MODELS = {
    "control": "decision-lab-v2-smollm2-360m-control",
    "treatment": "decision-lab-v2-smollm2-360m-counterfactual",
}


def download(source, root):
    volume = modal.Volume.from_name("decision-model-lab-v2-runs")
    for arm in MODELS:
        remote = f"/study-v2/{arm}/releases/head"
        payload = b"".join(volume.read_file(remote + "/artifact-manifest.json"))
        manifest = json.loads(payload)
        directory = source / arm
        directory.mkdir(parents=True, exist_ok=True)
        for name, digest in manifest["files"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe release manifest path")
            if path.exists() and file_hash(path) == digest:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(remote + "/" + name):
                    stream.write(chunk)
            if file_hash(temporary) != digest:
                raise ValueError("Downloaded file checksum mismatch")
            os.replace(temporary, path)
        (directory / "artifact-manifest.json").write_bytes(payload)
        release_inventory(directory)
        print(json.dumps({"downloaded": arm}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "publish", "verify"))
    args = parser.parse_args()
    root = Path("runs/study-v2")
    checks = json.loads((root / "artifact-verification.json").read_text())
    if set(checks) != set(MODELS) or not all(v["verified"] for v in checks.values()):
        raise ValueError("Both v2 GPU export checks must pass before release")
    source = Path("artifacts/v2-releases")
    receipt = root / "huggingface-release.json"
    if args.action == "download":
        download(source, root)
    elif args.action == "publish":
        publish_private_releases(source, receipt, "axondendriteplus", model_names=MODELS)
    else:
        verify_hub_downloads(receipt, Path("artifacts/v2-hub-verification"))


if __name__ == "__main__":
    main()
