import json
from pathlib import Path

import pytest
import torch
from test_v4 import tiny_parent
from torch.nn import functional as F

from decision_lab.data.validate import read_examples
from decision_lab.studies.v8 import study
from decision_lab.studies.v8.data import generate, validate
from decision_lab.studies.v8.objective import paired_loss
from decision_lab.studies.v8.protocol import budget, reservation, settings
from decision_lab.training.checkpoint import file_hash


def small_spec(rep):
    return settings(rep) | dict(
        train_old_families=12,
        train_multi_groups_per_size=1,
        auxiliary_old_families=2,
        auxiliary_multi_groups_per_size=1,
        test_old_families=2,
        test_multi_groups_per_size=3,
        transfer_old_families=2,
        transfer_multi_groups_per_size=1,
    )


def test_alignment_and_cross_entropy_gradients():
    logits = torch.tensor([[4.0, 1.0, -1.0], [-1.0, 1.0, 4.0]], requires_grad=True)
    labels = torch.tensor([0, 2])
    loss, ce, js = paired_loss(logits, labels, 0.5)
    assert abs(js.item()) < 1e-7
    assert torch.allclose(loss, F.cross_entropy(logits, labels))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    other = torch.tensor([[4.0, 1.0, -1.0], [4.0, 1.0, -1.0]], requires_grad=True)
    loss, ce, js = paired_loss(other, labels, 0.5)
    assert js.item() > 0 and loss > ce
    loss.backward()
    assert torch.isfinite(other.grad).all()


def test_identity_symmetry_extreme_logits_and_wrong_consistency():
    torch.manual_seed(31)
    logits = torch.randn(18, 3, requires_grad=True)
    labels = torch.randint(0, 3, (18,))
    value, _, _ = paired_loss(logits, labels, 0.5)
    flipped = logits.reshape(-1, 2, 3).flip(1).reshape(-1, 3)
    targets = labels.reshape(-1, 2).flip(1).reshape(-1)
    assert torch.allclose(value, paired_loss(flipped, targets, 0.5)[0])
    zero = paired_loss(logits, labels, 0)[0]
    assert torch.allclose(zero, F.cross_entropy(logits, labels))
    extreme = torch.tensor([[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 0.0]], requires_grad=True)
    loss, _, _ = paired_loss(extreme, torch.tensor([2, 2]), 0.5)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    wrong = torch.tensor([[-10.0, 10.0, -10.0], [-10.0, 10.0, -10.0]])
    loss, _, js = paired_loss(wrong, torch.tensor([0, 2]), 0.5)
    assert js.abs() < 1e-7 and loss > 10


def test_fresh_data_pairs_and_intervention_invariants(tmp_path):
    excluded = set()
    for rep in ("r1", "r2"):
        root = tmp_path / rep
        audit = generate(root, small_spec(rep), excluded, {})
        assert audit["shared_training_questions"] == 144
        control, treatment = (
            read_examples(root / f"train_{arm}.jsonl") for arm in ("control", "treatment")
        )
        assert [r.prompt for r in control] == [r.prompt for r in treatment]
        pairs = list(zip(control[::2], control[1::2], strict=True))
        assert all(a.family_id == b.family_id and a.query == b.query for a, b in pairs)
        records = [json.loads(line) for line in (root / "families.jsonl").read_text().splitlines()]
        lookup = {r["family_id"]: r for r in records}
        primary = read_examples(root / "test_paired.jsonl")
        by_family = {}
        for row in primary:
            by_family.setdefault(row.family_id, []).append(row)
        assert {r["pair_variant"] for r in records if r["partition"] == "test_multi"} == {
            "support_removal",
            "support_reversal",
            "irrelevant_edit",
        }
        for family, rows in by_family.items():
            record = lookup[family]
            size = len(record["entities"])
            assert len(rows) == 2 * size
            for before, after in zip(rows[:size], rows[size:], strict=True):
                assert len(before.theory.facts) == len(after.theory.facts)
                target = before.query.arguments[0] == record["entities"][record["target_slot"]]
                if not target or record["pair_variant"] == "irrelevant_edit":
                    assert before.label == after.label
                elif record["pair_variant"] == "support_removal":
                    assert after.label.value == "UNKNOWN"
                elif before.label.value != "UNKNOWN":
                    assert before.label != after.label
        hashes = {h for r in records for h in r["world_hashes"]}
        assert not hashes & excluded
        excluded |= hashes
        with pytest.raises(ValueError, match="overlap"):
            validate(root, excluded)


def test_budget_failure_reservation():
    assert sum(budget(settings(rep))["maximum_compute_usd"] for rep in ("r1", "r2")) < 8
    ledger = [
        {
            "repetition": "r1",
            "stage": "retry",
            "name": "failed",
            "status": "failed",
            "allocation_usd": 0.1,
        }
    ]
    with pytest.raises(ValueError, match="exhausted"):
        reservation(ledger, "r1", "retry", "again")


def test_paired_training_exact_resume_selection_export(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    spec = small_spec("r1") | dict(
        epochs=1, effective_batch=24, microbatch=12, candidate_steps=[0, 2, 4, 6], chunk_steps=3
    )
    root = tmp_path / "resumed"
    tiny_parent(root / "parent")
    generate(root / "data", spec, set(), {})
    protocol = {
        "identity": "test-v8",
        "settings": spec,
        "parent_files": {"model.safetensors": file_hash(root / "parent/model.safetensors")},
    }
    monkeypatch.setattr(study, "verify", lambda path: protocol)
    for arm in ("control", "treatment"):
        assert not study.train_chunk(root, arm, lambda: None, "cpu")["complete"]
        assert study.train_chunk(root, arm, lambda: None, "cpu")["complete"]
        other = tmp_path / f"uninterrupted-{arm}"
        tiny_parent(other / "parent")
        generate(other / "data", spec, set(), {})
        spec["chunk_steps"] = 6
        study.train_chunk(other, arm, lambda: None, "cpu")
        assert file_hash(root / f"candidates/{arm}-step6/model.safetensors") == file_hash(
            other / f"candidates/{arm}-step6/model.safetensors"
        )
        spec["chunk_steps"] = 3
    assert len(study.select(root, lambda: None, "cpu")["candidates"]) == 8
    result = study.evaluate(root, lambda: None, Path("configs"), "cpu")
    assert all(r["verified"] for r in result["verification"].values())


def test_blocked_contrast_and_target_subsets(tmp_path):
    from decision_lab.studies.v8.analysis import complete_worlds, entity_checks, world_contrast

    root = tmp_path / "data"
    spec = small_spec("r1")
    generate(root, spec, set(), {})
    records = {
        r["family_id"]: r
        for r in (json.loads(line) for line in (root / "families.jsonl").read_text().splitlines())
    }
    paired = read_examples(root / "test_paired.jsonl")
    families = {r.family_id for r in paired}
    contrast = world_contrast(dict.fromkeys(families, 1), dict.fromkeys(families, 0), records, spec)
    assert contrast["estimate"] == 1 and contrast["interval"] == [1.0, 1.0]
    rows = read_examples(root / "test_multi_transformations.jsonl")
    labels = torch.tensor([spec["class_labels"].index(r.label.value) for r in rows])
    logits = F.one_hot(labels, 3).double() * 10
    checks = entity_checks(rows, logits, labels, records)
    assert all(r["pairs"] == r["both_correct"] for r in checks.values())
    assert checks["support_removal/target_known"]["pairs"] > 0
    assert checks["support_removal/target_unknown"]["pairs"] > 0
    assert all(complete_worlds(rows, logits, labels).values())


def test_every_edit_covers_every_target_position(tmp_path):
    root = tmp_path / "balanced"
    audit = generate(root, small_spec("r1") | {"train_multi_groups_per_size": 15}, set(), {})
    for key, counts in audit["target_position_counts"].items():
        if key.startswith("train/"):
            size = int(key.split("/")[1])
            values = [counts[i] for i in range(size)]
            assert all(values) and max(values) - min(values) <= 3
