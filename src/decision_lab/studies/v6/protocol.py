import json
import math
from pathlib import Path

import yaml

from decision_lab.publication import atomic_json
from decision_lab.studies.v3.protocol import check_seal, inventory, seal
from decision_lab.training.checkpoint import file_hash


def settings(directory=Path("configs")):
    spec = yaml.safe_load((directory / "v6.yaml").read_text())
    if spec["class_labels"] != ["TRUE", "FALSE", "UNKNOWN"]:
        raise ValueError("Explicit class ordering is required")
    rows = 3 * spec["train_old_families"]
    if rows // 2 != 72 * spec["train_multi_groups_per_size"] or rows % 2:
        raise ValueError("Training arms must have exactly matched balanced exposure")
    steps = math.ceil(rows / spec["effective_batch"]) * spec["epochs"]
    if spec["candidate_steps"][-1] != steps:
        raise ValueError("Final candidate must match total updates")
    return spec


def code_inventory():
    import decision_lab

    package = Path(decision_lab.__file__).parent
    return {
        f"decision_lab/{path.relative_to(package)}": file_hash(path)
        for path in sorted(package.rglob("*.py"))
    }


def budget(spec):
    hourly = spec["gpu_hourly_usd"] + spec["cpu_cores"] * spec["cpu_core_hourly_usd"]
    hourly += spec["memory_mib"] / 1024 * spec["memory_gib_hourly_usd"]
    overhead = spec["startup_timeout_seconds"] + spec["scaledown_seconds"]
    calls = {
        stage: hourly * (spec[f"{stage}_timeout_seconds"] + overhead) / 3600
        for stage in ("pilot", "training", "selection", "test", "retry")
    }
    maximum = sum(calls[s] for s in ("pilot", "selection", "test"))
    maximum += 2 * spec["max_training_calls_per_arm"] * calls["training"]
    maximum += spec["max_retry_calls"] * calls["retry"]
    if maximum > spec["compute_allocation_usd"]:
        raise ValueError("Call ceilings exceed compute allocation")
    return {
        "hourly_requested_usd": hourly,
        "call_ceiling_usd": calls,
        "maximum_compute_usd": maximum,
    }


def freeze(root, audit):
    path = root / "protocol.json"
    if path.exists():
        raise FileExistsError(path)
    spec = settings()
    record = seal(
        {
            "study": spec["study"],
            "settings": spec,
            "budget": budget(spec),
            "code_files": code_inventory(),
            "parent_files": inventory(root / "parent"),
            "data_files": inventory(root / "data"),
            "audit": audit,
            "lock_sha256": file_hash(Path("uv.lock")),
            "protocol_markdown_sha256": file_hash(Path("docs/V6_PROTOCOL.md")),
            "primary": "Treatment-minus-control complete-world accuracy on test_multi",
            "selector": "Minimum calibrated selection NLL; earlier-step exact ties",
            "test_predictions_inspected": False,
        }
    )
    atomic_json(path, record)
    return record


def verify(root):
    original = check_seal(json.loads((root / "protocol.json").read_text()))
    amended = root / "amended-protocol.json"
    record = check_seal(json.loads(amended.read_text())) if amended.exists() else original
    if amended.exists():
        if record["parent_protocol_identity"] != original["identity"]:
            raise ValueError("Amendment belongs to another original protocol")
        if record["initial_protocol_sha256"] != file_hash(root / "protocol.json"):
            raise ValueError("Original frozen protocol changed")
        for field in (
            "settings",
            "parent_files",
            "data_files",
            "audit",
            "budget",
            "primary",
            "selector",
        ):
            if record[field] != original[field]:
                raise ValueError(f"Operational amendment changed scientific field: {field}")
    if record["code_files"] != code_inventory():
        raise ValueError("Scientific code changed after freeze")
    if inventory(root / "parent") != record["parent_files"]:
        raise ValueError("Parent files changed")
    if inventory(root / "data") != record["data_files"]:
        raise ValueError("Frozen data changed")
    if settings() != record["settings"]:
        raise ValueError("Frozen settings changed")
    return record
