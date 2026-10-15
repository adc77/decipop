import json
import math
from pathlib import Path

import yaml

from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.model_config import load_pilot_config, load_scratch_config
from decision_lab.publication import atomic_json
from decision_lab.studies.v2.budget import allocations
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.config import load_training_config
from decision_lab.training.protocol import json_hash, make_protocol, verify_protocol


def settings(directory=Path("configs")):
    spec = yaml.safe_load((directory / "v2.yaml").read_text())
    pilot = load_pilot_config(directory / "pilot.yaml").model_copy(
        update={
            "volume_name": spec["volume"],
            "app_name": "decision-model-lab-v2",
        }
    )
    training = load_training_config(directory / "training.yaml").model_copy(
        update={
            "epochs": spec["epochs"],
            "seed": spec["model_seed"],
            "chunk_steps": spec["chunk_steps"],
            "chunk_training_seconds": spec["chunk_training_seconds"],
            "timeout_seconds": spec["training_timeout_seconds"],
        }
    )
    scratch = load_scratch_config(directory / "scratch.yaml")
    evaluation = load_evaluation_config(directory / "evaluation.yaml")
    return spec, pilot, scratch, training, evaluation


def code_hash():
    package = Path(__file__).parents[2]
    paths = [
        *sorted((package / "studies/v2").glob("*.py")),
        *sorted((package / "data").glob("*.py")),
        *(
            package / "evaluation" / name
            for name in ("config.py", "metrics.py", "export.py", "study.py")
        ),
        package / "publication.py",
        package / "release.py",
    ]
    return json_hash({str(path.relative_to(package)): file_hash(path) for path in paths})


def freeze(root: Path, data: Path, audit: Path, pilot_report: dict):
    spec, pilot, scratch, training, evaluation = settings()
    prices, maximum = allocations(spec, pilot)
    steps = math.ceil(3 * spec["train_families"] / training.effective_batch) * training.epochs
    forecasts = {}
    for arm in ("control", "treatment"):
        measured = pilot_report["arms"][arm]["seconds_per_update"]
        usable = training.chunk_training_seconds / (measured * 1.2)
        calls = math.ceil(steps / min(training.chunk_steps, usable))
        if calls > spec["max_training_calls_per_arm"]:
            raise ValueError(f"Throughput forecast exceeds the call budget for {arm}: {calls}")
        forecasts[arm] = calls
    arms = {}
    for arm in ("control", "treatment"):
        local = root / arm
        local.mkdir(parents=True, exist_ok=True)
        tests = local / "test-inputs"
        tests.mkdir(exist_ok=True)
        atomic_json(tests / "manifest.json", {"v2": True, "shared_tests": True})
        protocol = make_protocol(
            data / arm,
            audit / arm,
            pilot,
            scratch,
            training,
            evaluation.model_dump(),
            Path("uv.lock"),
            tests,
        )
        path = local / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Existing arm protocol differs")
        atomic_json(path, protocol)
        arms[arm] = protocol["identity"]
    comparison = {
        "settings": spec,
        "arms": arms,
        "v2_code_sha256": code_hash(),
        "pilot_sha256": file_hash(root / "pilot.json"),
        "primary_metric": "treatment minus control UNKNOWN accuracy on test_counterfactual",
        "secondary_paired_metric": "treatment minus control raw NLL on test_counterfactual",
        "forecast_training_calls": forecasts,
        "maximum_allocation_usd": maximum,
        "call_allocations_usd": prices,
        "test_predictions_inspected": False,
        "training_protocol_markdown_sha256": file_hash(Path("docs/V2_PROTOCOL.md")),
    }
    comparison["identity"] = json_hash(comparison)
    path = root / "comparison.json"
    if path.exists() and json.loads(path.read_text()) != comparison:
        raise ValueError("Comparison is already frozen with different contents")
    atomic_json(path, comparison)
    return comparison


def verify_comparison(root: Path):
    comparison = json.loads((root / "comparison.json").read_text())
    if (
        json_hash({k: v for k, v in comparison.items() if k != "identity"})
        != comparison["identity"]
    ):
        raise ValueError("Comparison identity mismatch")
    if comparison["v2_code_sha256"] != code_hash():
        raise ValueError("V2 scientific implementation changed after freeze")
    for arm, identity in comparison["arms"].items():
        protocol = json.loads((root / arm / "protocol.json").read_text())
        verify_protocol(protocol)
        if protocol["identity"] != identity:
            raise ValueError("Arm protocol mismatch")
    return comparison
