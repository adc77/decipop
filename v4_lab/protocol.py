import json
from pathlib import Path

import yaml

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.protocol import json_hash
from decision_lab.v3.protocol import check_seal, inventory, seal


def settings(directory=Path("configs")):
    spec = yaml.safe_load((directory / "v4.yaml").read_text())
    if spec["class_labels"] != ["TRUE", "FALSE", "UNKNOWN"]:
        raise ValueError("V4 requires explicit class ordering")
    if not 0 < spec["temperature_min"] < 1 < spec["temperature_max"]:
        raise ValueError("Invalid temperature bounds")
    return spec


def code_inventory():
    package = Path(__file__).parent
    historic = package.parent / "src/decision_lab"
    if not historic.exists():
        import decision_lab

        historic = Path(decision_lab.__file__).parent
    return {
        **{f"v4_lab/{p.name}": file_hash(p) for p in sorted(package.glob("*.py"))},
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
    overhead = spec["startup_timeout_seconds"] + spec["scaledown_seconds"]
    calls = {
        stage: hourly * (spec[f"{stage}_timeout_seconds"] + overhead) / 3600
        for stage in ("diagnostic", "pilot", "training", "selection", "test", "retry")
    }
    maximum = (
        sum(calls[s] for s in ("diagnostic", "pilot", "selection", "test"))
        + 2 * spec["max_training_calls_per_arm"] * calls["training"]
        + spec["max_retry_calls"] * calls["retry"]
    )
    if maximum > spec["compute_allocation_usd"]:
        raise ValueError("Call ceilings exceed V4 compute allocation")
    return {
        "hourly_requested_usd": hourly,
        "call_ceiling_usd": calls,
        "maximum_compute_usd": maximum,
    }


def freeze(root, phase, audit):
    path = root / f"{phase}-protocol.json"
    if path.exists():
        raise FileExistsError(path)
    spec = settings()
    record = seal(
        {
            "phase": phase,
            "settings": spec,
            "budget": budget(spec),
            "code_files": code_inventory(),
            "parent_files": inventory(root / "parent"),
            "data_files": inventory(root / phase),
            "audit": audit,
            "lock_sha256": file_hash(Path("uv.lock")),
            "test_predictions_inspected": False,
        }
    )
    if phase == "training":
        record = seal(
            {k: v for k, v in record.items() if k != "identity"}
            | {
                "diagnostic_protocol_identity": json.loads(
                    (root / "diagnostic-protocol.json").read_text()
                )["identity"],
                "diagnostic_results_sha256": file_hash(root / "diagnostics.json"),
                "protocol_markdown_sha256": file_hash(Path("docs/V4_PROTOCOL.md")),
                "primary": "treatment-minus-control unsupported-query UNKNOWN accuracy",
                "selector": "minimum calibrated selection NLL; earlier-step exact ties",
            }
        )
    atomic_json(path, record)
    return record


def verify(root, phase):
    protocol = check_seal(json.loads((root / f"{phase}-protocol.json").read_text()))
    if protocol["code_files"] != code_inventory():
        raise ValueError("V4 scientific source changed after freeze")
    if inventory(root / "parent") != protocol["parent_files"]:
        raise ValueError("Parent artifact changed after freeze")
    if inventory(root / phase) != protocol["data_files"]:
        raise ValueError("Frozen V4 inputs changed")
    if phase == "training" and settings() != protocol["settings"]:
        raise ValueError("Training settings changed after freeze")
    return protocol


def canonical_identity(value):
    return json_hash(value)
