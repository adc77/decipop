import json
from pathlib import Path

import yaml

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal, inventory, seal


def settings(directory=Path("configs")):
    spec = yaml.safe_load((directory / "v5.yaml").read_text())
    if spec["class_labels"] != ["TRUE", "FALSE", "UNKNOWN"]:
        raise ValueError("Explicit class ordering is required")
    if any(count % 96 for count in spec["families"].values()):
        raise ValueError("Diagnostic counts must preserve exact label/name/polarity balance")
    return spec


def scientific_files():
    package = Path(__file__).parent
    import decision_lab

    historic = Path(decision_lab.__file__).parent
    return {
        **{f"v5_lab/{p.name}": file_hash(p) for p in sorted(package.glob("*.py"))},
        **{
            f"decision_lab/{p.relative_to(historic)}": file_hash(p)
            for p in sorted(historic.rglob("*.py"))
        },
    }


def budget(spec):
    hourly = (
        spec["gpu_hourly_usd"]
        + spec["cpu_cores"] * spec["cpu_core_hourly_usd"]
        + spec["memory_mib"] / 1024 * spec["memory_gib_hourly_usd"]
    )
    amounts = {
        phase: hourly
        * (
            spec[f"{phase}_timeout_seconds"]
            + spec["startup_timeout_seconds"]
            + spec["scaledown_seconds"]
        )
        / 3600
        for phase in ("pilot", "diagnostic", "retry")
    }
    maximum = amounts["pilot"] + amounts["diagnostic"] + amounts["retry"] * spec["max_retry_calls"]
    if maximum > spec["compute_allocation_usd"]:
        raise ValueError("Compute reservations exceed allocation")
    return {
        "hourly_requested_usd": hourly,
        "call_ceiling_usd": amounts,
        "maximum_compute_usd": maximum,
    }


def freeze(root, candidates, audit):
    path = root / "protocol.json"
    if path.exists():
        raise FileExistsError(path)
    spec = settings()
    record = seal(
        {
            "study": spec["study"],
            "settings": spec,
            "models": candidates,
            "data_files": inventory(root / "data"),
            "audit": audit,
            "scientific_files": scientific_files(),
            "budget": budget(spec),
            "lock_sha256": file_hash(Path("uv.lock")),
            "protocol_markdown_sha256": file_hash(Path("docs/V5_PROTOCOL.md")),
            "predictions_inspected": False,
            "primary": "V4 mixed complete-base-world accuracy on conjunction suite",
            "uncertainty": "parent-family sampling for fixed models; one training seed",
        }
    )
    atomic_json(path, record)
    return record


def verify(root, config_directory=Path("configs")):
    record = check_seal(json.loads((root / "protocol.json").read_text()))
    if record["scientific_files"] != scientific_files():
        raise ValueError("Scientific source changed after freeze")
    if record["settings"] != settings(config_directory):
        raise ValueError("Settings changed after freeze")
    if record["data_files"] != inventory(root / "data"):
        raise ValueError("Frozen data changed")
    for name, candidate in record["models"].items():
        if candidate["files"] != inventory(root / "models" / name):
            raise ValueError(f"Frozen model files changed: {name}")
    return record
