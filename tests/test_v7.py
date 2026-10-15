import json
from pathlib import Path

import pytest
import torch
from test_v4 import tiny_parent

from decision_lab.data.validate import read_examples
from decision_lab.studies.v7 import study
from decision_lab.studies.v7.data import generate, validate
from decision_lab.studies.v7.protocol import budget, reservation, settings
from decision_lab.training.checkpoint import file_hash


def small_spec(repetition):
    return settings(repetition) | {
        "train_old_families": 48,
        "train_multi_groups_per_size": 1,
        "auxiliary_old_families": 24,
        "auxiliary_multi_groups_per_size": 1,
        "test_old_families": 2,
        "test_multi_groups_per_size": 1,
        "transfer_old_families": 2,
        "transfer_multi_groups_per_size": 1,
    }


def test_repetitions_are_fresh_namespaced_and_label_matched(tmp_path):
    excluded, previous_ids = set(), set()
    for rep in ("r1", "r2"):
        directory = tmp_path / rep
        audit = generate(directory, small_spec(rep), excluded, {})
        assert audit["valid"]
        records = [
            json.loads(line) for line in (directory / "families.jsonl").read_text().splitlines()
        ]
        worlds = {h for r in records for h in r["world_hashes"]}
        ids = {r["family_id"] for r in records}
        assert not worlds & excluded and not ids & previous_ids
        assert all(name.startswith(f"v7-{rep}-") for name in ids)
        control, treatment = (
            read_examples(directory / f"train_{arm}.jsonl") for arm in ("control", "treatment")
        )
        assert [r.label for r in control] == [r.label for r in treatment]
        assert sum(a.prompt == b.prompt for a, b in zip(control, treatment, strict=True)) == 72
        excluded |= worlds
        previous_ids |= ids
        with pytest.raises(ValueError, match="overlap"):
            validate(directory, excluded)


def test_same_curriculum_settings_and_joint_reservation():
    from decision_lab.studies.v6.protocol import settings as v6_settings

    previous = v6_settings()
    replicas = [settings(rep) for rep in ("r1", "r2")]
    changed = {"study", "dataset_seed", "model_seed", "bootstrap_seed", "volume", "app_name"}
    for spec in replicas:
        assert {k: v for k, v in spec.items() if k not in changed} == {
            k: v for k, v in previous.items() if k not in changed
        }
    assert replicas[0]["model_seed"] != replicas[1]["model_seed"] != previous["model_seed"]
    assert sum(budget(r)["maximum_compute_usd"] for r in replicas) < 8
    with pytest.raises(ValueError, match="exceed"):
        budget(replicas[0] | {"compute_allocation_usd": 3})


def test_failed_calls_consume_the_stage_and_joint_allowances():
    amount = budget(settings("r1"))["call_ceiling_usd"]["retry"]
    ledger = [
        {
            "repetition": "r1",
            "stage": "retry",
            "name": "first",
            "status": "failed",
            "allocation_usd": amount,
        }
    ]
    with pytest.raises(ValueError, match="exhausted"):
        reservation(ledger, "r1", "retry", "second")
    with pytest.raises(ValueError, match="already submitted"):
        reservation(ledger, "r1", "retry", "first")
    assert reservation(ledger, "r2", "retry", "first") == amount
    with pytest.raises(ValueError, match="Joint"):
        reservation(
            ledger
            + [{"repetition": "r2", "stage": "training", "name": "used", "allocation_usd": 7.5}],
            "r1",
            "pilot",
            "pilot",
        )


@pytest.mark.parametrize("rep", ["r1", "r2"])
def test_namespacing_does_not_change_the_v6_curriculum(tmp_path, rep):
    from decision_lab.studies.v6.data import generate as reference_generate

    spec = small_spec(rep)
    reference_generate(tmp_path / "reference", spec, set(), {})
    generate(tmp_path / "replica", spec, set(), {})
    for path in sorted((tmp_path / "reference").glob("*.jsonl")):
        if path.stem == "families":
            continue
        before, after = read_examples(path), read_examples(tmp_path / "replica" / path.name)
        assert [(r.prompt, r.label, r.theory_hash, r.proof_depth) for r in before] == [
            (r.prompt, r.label, r.theory_hash, r.proof_depth) for r in after
        ]


def test_fresh_seed_training_resume_selection_and_reload(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    spec = small_spec("r1") | {
        "epochs": 1,
        "effective_batch": 24,
        "microbatch": 12,
        "candidate_steps": [0, 2, 4, 6],
        "chunk_steps": 3,
    }
    root = tmp_path / "resumed"
    tiny_parent(root / "parent")
    (root / "parent/code").mkdir()
    (root / "parent/code/vendor.py").write_text("retained = True\n")
    generate(root / "data", spec, set(), {})
    protocol = {
        "identity": "test-v7-r1",
        "settings": spec,
        "parent_files": {"model.safetensors": file_hash(root / "parent/model.safetensors")},
    }
    monkeypatch.setattr(study, "verify", lambda path: protocol)
    for arm in ("control", "treatment"):
        assert not study.train_chunk(root, arm, lambda: None, "cpu")["complete"]
        assert study.train_chunk(root, arm, lambda: None, "cpu")["complete"]
    other = tmp_path / "uninterrupted"
    tiny_parent(other / "parent")
    generate(other / "data", spec, set(), {})
    spec["chunk_steps"] = 6
    study.train_chunk(other, "control", lambda: None, "cpu")
    assert file_hash(root / "candidates/control-step6/model.safetensors") == file_hash(
        other / "candidates/control-step6/model.safetensors"
    )
    selection = study.select(root, lambda: None, "cpu")
    assert len(selection["candidates"]) == 8
    results = study.evaluate(root, lambda: None, Path("configs"), "cpu")
    assert all(r["verified"] for r in results["verification"].values())
    with pytest.raises(FileExistsError, match="already started"):
        study.evaluate(root, lambda: None, Path("configs"), "cpu")
