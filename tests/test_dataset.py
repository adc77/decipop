import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from decision_lab.config import DatasetConfig, load_dataset_config
from decision_lab.data.generate import generate_dataset
from decision_lab.data.schema import Atom, Rule, Theory
from decision_lab.data.split import canonical_hash, content_hash
from decision_lab.data.validate import read_examples, validate_dataset


@pytest.fixture
def config():
    design = load_dataset_config(Path("configs/smoke.yaml"))
    return design.model_copy(
        update={
            "splits": {
                name: spec.model_copy(update={"families": 4})
                for name, spec in design.splits.items()
            }
        }
    )


def test_generation_reproducibility_and_validation(tmp_path, config):
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert generate_dataset(config, first) == generate_dataset(config, second)
    summary = validate_dataset(first)
    assert summary["valid"]
    assert not summary["token_length_verified"]
    assert all(row["questions"] == 12 for row in summary["splits"].values())
    for path in first.glob("*.jsonl"):
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_entity_and_variable_renaming_is_canonicalized():
    first = Theory(
        entities=("Aman", "Bella"),
        facts=(Atom(predicate="ready", arguments=("Aman",), negative=False),),
        rules=(
            Rule(
                conditions=(Atom(predicate="ready", arguments=("?x",), negative=False),),
                conclusion=Atom(predicate="active", arguments=("?x",), negative=False),
            ),
        ),
    )
    second = Theory(
        entities=("Dalia", "Chen"),
        facts=(Atom(predicate="ready", arguments=("Chen",), negative=False),),
        rules=(
            Rule(
                conditions=(Atom(predicate="ready", arguments=("?renamed",), negative=False),),
                conclusion=Atom(predicate="active", arguments=("?renamed",), negative=False),
            ),
        ),
    )
    assert canonical_hash(first) == canonical_hash(second)
    assert canonical_hash(first) != canonical_hash(first.model_copy(update={"facts": ()}))


def test_content_tampering_is_detected(tmp_path, config):
    directory = tmp_path / "dataset"
    generate_dataset(config, directory)
    path = directory / "train.jsonl"
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["label"] = "UNKNOWN" if row["label"] != "UNKNOWN" else "TRUE"
    rows[0] = json.dumps(row)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="Content hash mismatch"):
        validate_dataset(directory)


def test_solver_catches_rehashed_false_labels(tmp_path, config):
    directory = tmp_path / "dataset"
    manifest = generate_dataset(config, directory)
    path = directory / "train.jsonl"
    rows = read_examples(path)
    first = rows[0]
    wrong_depth = 99 if first.proof_depth is not None else 0
    rows[0] = first.model_copy(update={"proof_depth": wrong_depth})
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    manifest["splits"]["train"]["sha256"] = content_hash(
        [row.model_dump(mode="json") for row in rows]
    )
    manifest["splits"]["train"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incorrect theory hash, answer, or proof depth"):
        validate_dataset(directory)


def test_no_overwrite_and_invalid_configuration(tmp_path, config):
    generate_dataset(config, tmp_path / "dataset")
    with pytest.raises(FileExistsError):
        generate_dataset(config, tmp_path / "dataset")
    raw = config.model_dump()
    raw["splits"]["train"]["depths"] = [4]
    with pytest.raises(ValueError, match="Training cannot contain"):
        DatasetConfig.model_validate(raw)


def test_relational_generation_uses_two_variable_rules(tmp_path, config):
    spec = config.splits["train"].model_copy(update={"depths": [2], "motifs": ["relational"]})
    relational = config.model_copy(update={"splits": {"train": spec}})
    directory = tmp_path / "relational"
    generate_dataset(relational, directory)
    validate_dataset(directory)
    examples = read_examples(directory / "train.jsonl")
    assert all(len(rule.conditions) == 2 for row in examples for rule in row.theory.rules)
    assert any(
        len(atom.arguments) == 2
        for row in examples
        for rule in row.theory.rules
        for atom in rule.conditions
    )


def test_reproducibility_across_python_hash_seeds(tmp_path, config):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()))
    for seed in ("111", "222"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "decision_lab",
                "generate",
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / seed),
            ],
            env=os.environ | {"PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
    for path in (tmp_path / "111").iterdir():
        assert path.read_bytes() == (tmp_path / "222" / path.name).read_bytes()
