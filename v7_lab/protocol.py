import json
import math
from pathlib import Path

import yaml

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal, inventory, seal
from v6_lab.protocol import budget
from v6_lab.protocol import code_inventory as historical_inventory

ROOT = Path("runs/study-v7")


def settings(repetition, directory=Path("configs")):
    spec = yaml.safe_load((directory / "v7.yaml").read_text())
    repetitions = spec.pop("repetitions")
    if set(repetitions) != {"r1", "r2"} or len(set(repetitions.values())) != 2:
        raise ValueError("Two distinct repetition seeds are required")
    seed = repetitions[repetition]
    spec |= {
        "study": f"v7-{repetition}",
        "dataset_seed": seed,
        "model_seed": seed,
        "bootstrap_seed": seed,
    }
    if spec["class_labels"] != ["TRUE", "FALSE", "UNKNOWN"]:
        raise ValueError("Explicit class ordering is required")
    rows = 3 * spec["train_old_families"]
    if rows // 2 != 72 * spec["train_multi_groups_per_size"] or rows % 2:
        raise ValueError("Matched balanced exposure is required")
    steps = math.ceil(rows / spec["effective_batch"]) * spec["epochs"]
    if spec["candidate_steps"][-1] != steps:
        raise ValueError("Final candidate must match total updates")
    return spec


def code_inventory():
    return historical_inventory() | {
        str(path): file_hash(path) for path in sorted(Path("v7_lab").glob("*.py"))
    }


def reservation(ledger, repetition, stage, name):
    spec = settings(repetition)
    caps = {
        "pilot": 1,
        "training": 2 * spec["max_training_calls_per_arm"],
        "selection": 1,
        "test": 1,
        "retry": spec["max_retry_calls"],
    }
    if stage not in caps:
        raise ValueError("Unknown reservation stage")
    own = [r for r in ledger if r["repetition"] == repetition]
    if any(r["name"] == name for r in own):
        raise ValueError("Call name already submitted")
    if sum(r["stage"] == stage for r in own) >= caps[stage]:
        raise ValueError("Stage call allowance exhausted, including failed calls")
    ceiling = budget(spec)["call_ceiling_usd"][stage]
    if sum(r["allocation_usd"] for r in own) + ceiling > spec["compute_allocation_usd"]:
        raise ValueError("Per-repetition allocation exceeded")
    if sum(r["allocation_usd"] for r in ledger) + ceiling > 8:
        raise ValueError("Joint allocation exceeded")
    return ceiling


def freeze(root, repetition, audit):
    target = root / "protocol.json"
    if target.exists():
        raise FileExistsError(target)
    spec = settings(repetition)
    record = seal(
        {
            "repetition": repetition,
            "settings": spec,
            "budget": budget(spec),
            "code_files": code_inventory(),
            "parent_files": inventory(root / "parent"),
            "data_files": inventory(root / "data"),
            "audit": audit,
            "config_sha256": file_hash(Path("configs/v7.yaml")),
            "lock_sha256": file_hash(Path("uv.lock")),
            "protocol_markdown_sha256": file_hash(Path("docs/V7_PROTOCOL.md")),
            "primary": "Treatment-minus-control complete-world accuracy on test_multi",
            "selector": "Minimum calibrated selection NLL; earlier-step exact ties",
            "test_predictions_inspected": False,
        }
    )
    atomic_json(target, record)
    return record


def verify(root):
    record = check_seal(json.loads((root / "protocol.json").read_text()))
    if record["settings"] != settings(record["repetition"]):
        raise ValueError("Frozen settings changed")
    if record["code_files"] != code_inventory():
        raise ValueError("Frozen scientific source changed")
    for folder in ("parent", "data"):
        if inventory(root / folder) != record[f"{folder}_files"]:
            raise ValueError(f"Frozen {folder} files changed")
    for name, field in (
        ("configs/v7.yaml", "config_sha256"),
        ("uv.lock", "lock_sha256"),
        ("docs/V7_PROTOCOL.md", "protocol_markdown_sha256"),
    ):
        if file_hash(Path(name)) != record[field]:
            raise ValueError(f"Frozen auxiliary file changed: {name}")
    return record


def freeze_study():
    path = ROOT / "study-freeze.json"
    if path.exists():
        raise FileExistsError(path)
    records = {rep: verify(ROOT / rep) for rep in ("r1", "r2")}
    hashes = {}
    for rep in records:
        registry = ROOT / rep / "data/families.jsonl"
        hashes[rep] = {
            digest
            for line in registry.read_text().splitlines()
            for digest in json.loads(line)["world_hashes"]
        }
    if hashes["r1"] & hashes["r2"]:
        raise ValueError("Repetitions share canonical worlds")
    record = seal(
        {
            "protocol_identities": {rep: r["identity"] for rep, r in records.items()},
            "joint_compute_allocation_usd": 8.0,
            "joint_storage_reserve_usd": 1.0,
            "maximum_compute_usd": sum(
                r["budget"]["maximum_compute_usd"] for r in records.values()
            ),
            "cross_repetition_world_overlap": 0,
            "replication_check": "Both primary lower bounds exceed zero",
            "engineering_replication_check": "Both repetitions pass all per-seed checks",
            "uncertainty": "Per-seed world-sampling intervals; no pooled training-seed interval",
        }
    )
    atomic_json(path, record)
    return record


def verify_study():
    record = check_seal(json.loads((ROOT / "study-freeze.json").read_text()))
    for rep, identity in record["protocol_identities"].items():
        if verify(ROOT / rep)["identity"] != identity:
            raise ValueError("Repetition no longer matches the study freeze")
    return record
