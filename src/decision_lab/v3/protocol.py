import json
from pathlib import Path

import yaml

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.protocol import json_hash


def settings(directory=Path("configs")):
    spec = yaml.safe_load((directory / "v3.yaml").read_text())
    if spec["class_labels"] != ["TRUE", "FALSE", "UNKNOWN"]:
        raise ValueError("V3 requires explicit TRUE/FALSE/UNKNOWN class ordering")
    if not 0 < spec["temperature_min"] < 1 < spec["temperature_max"]:
        raise ValueError("Invalid temperature bounds")
    return spec


def code_hash():
    package = Path(__file__).parent.parent
    return json_hash(
        {str(path.relative_to(package)): file_hash(path) for path in sorted(package.rglob("*.py"))}
    )


def inventory(directory):
    return {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def budget(spec):
    hourly = (
        spec["gpu_hourly_usd"]
        + spec["cpu_cores"] * spec["cpu_core_hourly_usd"]
        + spec["memory_mib"] / 1024 * spec["memory_gib_hourly_usd"]
    )
    overhead = spec["startup_timeout_seconds"] + spec["scaledown_seconds"]
    amounts = {
        stage: hourly * (spec[f"{stage}_timeout_seconds"] + overhead) / 3600
        for stage in ("selection", "test", "retry")
    }
    maximum = amounts["selection"] + amounts["test"] + spec["max_retry_calls"] * amounts["retry"]
    if maximum > spec["compute_allocation_usd"]:
        raise ValueError("GPU call ceilings exceed the compute allocation")
    return {
        "hourly_requested_usd": hourly,
        "call_ceiling_usd": amounts,
        "maximum_compute_usd": maximum,
    }


def seal(value):
    return {**value, "identity": json_hash(value)}


def check_seal(value):
    if (
        json_hash({key: item for key, item in value.items() if key != "identity"})
        != value["identity"]
    ):
        raise ValueError("Frozen record identity mismatch")
    return value


def verify(root, config_directory=Path("configs")):
    protocol = check_seal(json.loads((root / "protocol.json").read_text()))
    if protocol["code_sha256"] != code_hash():
        raise ValueError("V3 scientific source changed after freeze")
    if settings(config_directory) != protocol["settings"]:
        raise ValueError("V3 settings changed after freeze")
    if inventory(root / "data") != protocol["data_files"]:
        raise ValueError("V3 inputs changed after freeze")
    for name, candidate in protocol["candidates"].items():
        if inventory(root / "candidates" / name) != candidate["files"]:
            raise ValueError(f"Candidate files changed: {name}")
    return protocol


def freeze(root, candidates, prior_identity, audit):
    spec = settings()
    protocol = seal(
        {
            "study": spec["study"],
            "settings": spec,
            "candidates": candidates,
            "parent_comparison_identity": prior_identity,
            "data_files": inventory(root / "data"),
            "input_audit": audit,
            "code_sha256": code_hash(),
            "protocol_markdown_sha256": file_hash(Path("docs/V3_PROTOCOL.md")),
            "lock_sha256": file_hash(Path("uv.lock")),
            "budget": budget(spec),
            "candidate_pool": "two retained checkpoints per arm; one training seed",
            "test_predictions_inspected": False,
            "primary_comparison": "treatment calibrated_nll minus raw_nll UNKNOWN accuracy",
        }
    )
    path = root / "protocol.json"
    if path.exists():
        raise FileExistsError("V3 is already frozen")
    atomic_json(path, protocol)
    return protocol
