import hashlib
import json
from pathlib import Path

from decision_lab.training.checkpoint import file_hash


def json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def training_code_hash() -> str:
    root = Path(__file__).parent.parent
    paths = [
        *sorted((root / "models").glob("*.py")),
        *(
            root / "training" / name
            for name in ("core.py", "checkpoint.py", "config.py", "runner.py")
        ),
        root / "model_config.py",
        root / "tokenization.py",
    ]
    return json_hash({str(path.relative_to(root)): file_hash(path) for path in paths})


def make_protocol(
    dataset: Path, audit: Path, pilot, scratch, training, evaluation: dict, lock: Path, tests: Path
):
    paths = {}
    for prefix, directory in (("data", dataset), ("audit", audit), ("test-inputs", tests)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                paths[str(Path(prefix) / path.relative_to(directory))] = file_hash(path)
    value = {
        "schema_version": 1,
        "input_files": paths,
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "training_code_sha256": training_code_hash(),
        "lock_sha256": file_hash(lock),
        "pilot": pilot.model_dump(),
        "scratch": scratch.model_dump(),
        "training": training.model_dump(),
        "evaluation": evaluation,
        "test_predictions_inspected": False,
        "checkpoint_selection": "development restricted three-class NLL; exact ties by accuracy",
        "class_labels": pilot.class_labels,
        "primary_seed": training.seed,
        "second_seed": "not included in v1",
    }
    return value | {"identity": json_hash(value)}


def verify_protocol(protocol: dict):
    if (
        json_hash({key: value for key, value in protocol.items() if key != "identity"})
        != protocol["identity"]
    ):
        raise ValueError("Protocol identity checksum mismatch")
    if training_code_hash() != protocol["training_code_sha256"]:
        raise ValueError("Training code changed after protocol freeze")
