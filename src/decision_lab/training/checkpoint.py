import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, model, optimizer, scheduler, progress: dict, identity: str):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".checkpoint-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        save_model(model, temporary / "model.safetensors")
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(state, temporary / "training.pt")
        (temporary / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        marker = {
            "identity": identity,
            "files": {
                name: file_hash(temporary / name)
                for name in ("model.safetensors", "training.pt", "progress.json")
            },
        }
        (temporary / "complete.json").write_text(json.dumps(marker, indent=2) + "\n")
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def verify_checkpoint(path: Path, identity: str):
    marker = json.loads((path / "complete.json").read_text())
    if marker["identity"] != identity:
        raise ValueError("Checkpoint protocol identity mismatch")
    expected = {"model.safetensors", "training.pt", "progress.json"}
    if set(marker["files"]) != expected:
        raise ValueError("Checkpoint file inventory mismatch")
    for name, digest in marker["files"].items():
        if file_hash(path / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")


def load_checkpoint(path: Path, model, optimizer, scheduler, identity: str) -> dict:
    verify_checkpoint(path, identity)
    load_model(model, path / "model.safetensors", strict=True)
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA random state cannot be restored on this worker")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return json.loads((path / "progress.json").read_text())
