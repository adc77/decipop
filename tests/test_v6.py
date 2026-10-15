import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch
from test_v4 import tiny_parent

from decision_lab.config import load_dataset_config
from decision_lab.data.schema import Label
from decision_lab.data.validate import read_examples
from decision_lab.studies.v6 import study
from decision_lab.studies.v6.analysis import complete_worlds, world_contrast
from decision_lab.studies.v6.data import VARIANTS, generate, multi_bundle, validate
from decision_lab.studies.v6.protocol import budget, settings
from decision_lab.training.checkpoint import file_hash


def small_spec():
    return settings() | {
        "train_old_families": 48,
        "train_multi_groups_per_size": 1,
        "auxiliary_old_families": 24,
        "auxiliary_multi_groups_per_size": 1,
        "test_old_families": 2,
        "test_multi_groups_per_size": 1,
        "transfer_old_families": 2,
        "transfer_multi_groups_per_size": 1,
    }


def test_local_edits_preserve_other_entities_and_fact_counts():
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rng = random.Random(631)
    for size in (3, 4, 5):
        for sign in (False, True):
            for label in Label:
                for signed in (False, True):
                    rows, record = multi_bundle(
                        config,
                        rng,
                        "test",
                        tuple(config.train_entities[:size]),
                        [label] * size,
                        sign,
                        2,
                        signed,
                        VARIANTS,
                        size - 1,
                    )
                    grouped = {
                        r.example_id.rsplit("-", 2)[-2] + "/" + r.example_id.rsplit("-", 1)[-1]: r
                        for r in rows
                    }
                    for variant in ("base", "support_edit", "support_edit_inverse"):
                        for slot in range(size):
                            row = grouped[f"{variant}/{slot}"]
                            assert Counter(f.arguments[0] for f in row.theory.facts) == Counter(
                                {e: 3 for e in record["entities"]}
                            )
                            if slot != size - 1:
                                assert row.label == label
                    assert (
                        len(
                            {
                                grouped[f"base/{slot}"].prompt.split("Statement:")[0]
                                for slot in range(size)
                            }
                        )
                        == 1
                    )


def test_matching_balance_exclusion_and_serialized_corruption(tmp_path):
    directory = tmp_path / "data"
    audit = generate(directory, small_spec(), set(), {})
    assert audit["valid"]
    control, treatment = (
        read_examples(directory / f"train_{arm}.jsonl") for arm in ("control", "treatment")
    )
    assert len(control) == len(treatment) == 144
    assert [r.label for r in control] == [r.label for r in treatment]
    assert sum(a.prompt == b.prompt for a, b in zip(control, treatment, strict=True)) == 72
    composition = [r for r in read_examples(directory / "test_multi.jsonl")]
    grouped = {}
    for row in composition:
        grouped.setdefault(row.family_id, set()).add(row.label)
    assert any(len(values) == 1 for values in grouped.values())
    records = [json.loads(line) for line in (directory / "families.jsonl").read_text().splitlines()]
    excluded = {h for record in records for h in record["world_hashes"]}
    with pytest.raises(ValueError, match="overlap"):
        validate(directory, excluded)
    path = directory / "selection.jsonl"
    path.write_text(path.read_text().replace('"label":"TRUE"', '"label":"UNKNOWN"', 1))
    with pytest.raises(ValueError, match="checksum"):
        validate(directory, set())


def test_budget_reserves_recovery_and_rejects_underfunding():
    result = budget(settings())
    assert 3.8 < result["maximum_compute_usd"] < 4
    with pytest.raises(ValueError, match="exceed"):
        budget(settings() | {"compute_allocation_usd": 3})


def test_complete_world_and_rotation_block_contrasts():
    spec = settings() | {"bootstrap_repetitions": 100}
    records = {
        f"family-{size}-{cycle}": {"entities": list(range(size)), "sampling_block": f"block-{size}"}
        for size in (3, 4, 5)
        for cycle in range(3)
    }
    left, right = dict.fromkeys(records, 1), dict.fromkeys(records, 0)
    result = world_contrast(left, right, records, spec)
    assert result["estimate"] == 1 and result["interval"] == [1, 1]
    assert result["blocks"] == 3 and result["families"] == 9
    assert world_contrast(left, right, records, spec) == result
    with pytest.raises(ValueError, match="incomplete"):
        world_contrast(dict(list(left.items())[:-1]), dict(list(right.items())[:-1]), records, spec)
    rows = [type("Row", (), {"family_id": "one"})() for _ in range(3)]
    assert complete_worlds(rows, torch.zeros(3, 3), torch.tensor([0, 1, 2])) == {"one": 0}


def test_cpu_training_resume_selection_and_fresh_export(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    spec = small_spec() | {
        "epochs": 1,
        "effective_batch": 24,
        "microbatch": 12,
        "evaluation_batch": 32,
        "candidate_steps": [0, 2, 4, 6],
        "chunk_steps": 3,
    }
    first = tmp_path / "first"
    tiny_parent(first / "parent")
    (first / "parent/code").mkdir()
    (first / "parent/code/vendor.py").write_text("retained = True\n")
    generate(first / "data", spec, set(), {})
    protocol = {
        "identity": "test-v6",
        "settings": spec,
        "parent_files": {"model.safetensors": file_hash(first / "parent/model.safetensors")},
    }
    monkeypatch.setattr(study, "verify", lambda root: protocol)
    for arm in ("control", "treatment"):
        assert not study.train_chunk(first, arm, lambda: None, "cpu")["complete"]
        assert study.train_chunk(first, arm, lambda: None, "cpu")["complete"]
    second = tmp_path / "second"
    tiny_parent(second / "parent")
    generate(second / "data", spec, set(), {})
    spec["chunk_steps"] = 6
    study.train_chunk(second, "control", lambda: None, "cpu")
    assert file_hash(first / "candidates/control-step6/model.safetensors") == file_hash(
        second / "candidates/control-step6/model.safetensors"
    )
    selected = study.select(first, lambda: None, "cpu")
    assert len(selected["candidates"]) == 8
    evaluated = study.evaluate(first, lambda: None, Path("configs"), "cpu")
    assert set(evaluated["evaluation"]["parent"]) == {
        "test_multi",
        "test_original",
        "test_counterfactual",
    }
    assert all(check["verified"] for check in evaluated["verification"].values())
    with pytest.raises(FileExistsError, match="already started"):
        study.evaluate(first, lambda: None, Path("configs"), "cpu")


def test_packaging_failure_recovers_original_optimizer_and_rng(tmp_path, monkeypatch):
    from decision_lab.studies.v6.recovery import prepare_resume

    torch.set_num_threads(2)
    spec = small_spec() | {
        "epochs": 1,
        "effective_batch": 24,
        "microbatch": 12,
        "evaluation_batch": 32,
        "candidate_steps": [0, 2, 4, 6],
        "chunk_steps": 6,
    }
    root = tmp_path / "interrupted"
    tiny_parent(root / "parent")
    generate(root / "data", spec, set(), {})
    protocol = {"identity": "original", "settings": spec}
    monkeypatch.setattr(study, "verify", lambda path: protocol)
    original = study.package_candidate

    def interrupted(*args):
        raise RuntimeError("Injected candidate packaging failure after checkpoint saved")

    monkeypatch.setattr(study, "package_candidate", interrupted)
    with pytest.raises(RuntimeError, match="Injected"):
        study.train_chunk(root, "control", lambda: None, "cpu")
    monkeypatch.setattr(study, "package_candidate", original)
    protocol["identity"] = "amended"
    partial = root / "candidates/control-step2"
    partial.mkdir(parents=True)
    (partial / "LICENSE").write_text("Retained partial asset\n")
    recovery = prepare_resume(root, "control", "original", "amended", 2)
    assert recovery["checkpoint_hashes_verified"] and recovery["presentations"] == 48
    assert (root / "failed-candidate-packaging/control-step2/LICENSE").exists()
    assert study.train_chunk(root, "control", lambda: None, "cpu")["complete"]
    uninterrupted = tmp_path / "uninterrupted"
    tiny_parent(uninterrupted / "parent")
    generate(uninterrupted / "data", spec, set(), {})
    study.train_chunk(uninterrupted, "control", lambda: None, "cpu")
    assert file_hash(root / "candidates/control-step6/model.safetensors") == file_hash(
        uninterrupted / "candidates/control-step6/model.safetensors"
    )
