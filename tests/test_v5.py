import json
import random
from pathlib import Path

import pytest

from decision_lab.config import load_dataset_config
from decision_lab.data.schema import Atom, Label, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash
from decision_lab.studies.v5.analysis import interval, pair_metrics
from decision_lab.studies.v5.data import (
    atom,
    audit_family,
    independent_closure,
    make_family,
    reference_label,
    validate_dataset,
)
from decision_lab.studies.v5.protocol import budget, settings


@pytest.mark.parametrize("suite", ["conjunction", "deeper", "signed_chain"])
@pytest.mark.parametrize("index", [0, 8, 16, 24, 32, 40, 48, 88])
def test_multi_entity_semantics_and_interventions(suite, index):
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rows, record = make_family(config, random.Random(1200 + index), index, "test", suite)
    audit_family(rows, record)
    base = [r for r in rows if r["variant"] == "base"]
    assert {r["label"] for r in base} == {"TRUE", "FALSE", "UNKNOWN"}
    assert len({r["prompt"].split("Statement:")[0] for r in base}) == 1
    assert all(
        sum(f.arguments[0] == entity for f in Theory.model_validate(base[0]["theory"]).facts) == 3
        for entity in record["entities"]
    )
    renamed = next(r for r in rows if r["variant"] == "renamed")
    assert canonical_hash(Theory.model_validate(renamed["theory"])) == base[0]["world_hash"]
    assert all(
        r["label"]
        == reference_label(
            independent_closure(Theory.model_validate(r["theory"])), Atom.model_validate(r["query"])
        ).value
        for r in rows
    )


def test_independent_oracle_keeps_entities_separate_and_rejects_contradiction():
    world = Theory(
        entities=("Hana", "Evan", "Mina"),
        facts=(atom("ready", "Hana"), atom("ready", "Evan", True)),
        rules=(),
    )
    closure = independent_closure(world)
    assert [reference_label(closure, atom("ready", name)) for name in world.entities] == list(Label)
    conflict = world.model_copy(update={"facts": (*world.facts, atom("ready", "Hana", True))})
    with pytest.raises(ValueError, match="contradiction"):
        independent_closure(conflict)
    with pytest.raises(InconsistentTheoryError):
        solve(conflict)


def test_constant_world_prediction_cannot_pass_query_binding_or_intervention():
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rows, record = make_family(config, random.Random(77), 0, "world", "conjunction")
    summary = pair_metrics(rows, ["TRUE"] * len(rows), {"world": record})
    assert summary["worlds"]["base"]["complete_world_correct"] == 0
    assert summary["pairs"]["query_change"]["expected_change_or_invariance"] == 0
    assert summary["pairs"]["remove_support/target"]["both_correct"] == 0
    oracle = pair_metrics(rows, [r["label"] for r in rows], {"world": record})
    assert oracle["worlds"]["base"]["complete_world_correct"] == 1
    for pair in oracle["pairs"].values():
        assert pair["pairs"] == pair["both_correct"] == pair["expected_change_or_invariance"]


def test_paired_invariance_does_not_mean_correctness():
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rows, record = make_family(config, random.Random(77), 0, "world", "conjunction")
    summary = pair_metrics(rows, ["UNKNOWN"] * len(rows), {"world": record})
    pair = summary["pairs"]["reordered"]
    assert pair["expected_change_or_invariance"] == 3 and pair["both_correct"] == 1


def test_corrupted_intervention_is_rejected():
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rows, record = make_family(config, random.Random(77), 0, "world", "conjunction")
    corrupted = [r.copy() for r in rows]
    row = next(
        r
        for r in corrupted
        if r["variant"] == "remove_support" and r["slot"] == record["target_slot"]
    )
    row["label"] = "TRUE"
    with pytest.raises(ValueError, match="unintended"):
        audit_family(corrupted, record)


def test_budget_and_family_bootstrap_are_deterministic():
    spec = settings()
    assert budget(spec)["maximum_compute_usd"] < spec["compute_allocation_usd"]
    assert interval([0, 0, 1, 1], spec) == interval([0, 0, 1, 1], spec)
    assert interval([0] * 10, spec)["interval"] == [0.0, 0.0]


def test_json_roundtrip_audit_and_corrupt_prompt(tmp_path):
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rows, record = make_family(config, random.Random(77), 0, "world", "conjunction")
    record["partition"] = "pilot"
    (tmp_path / "families.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "pilot.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "diagnostic.jsonl").write_text("")
    assert validate_dataset(tmp_path, set())["questions"] == 27
    rows[0]["prompt"] += "Injected answer: TRUE"
    (tmp_path / "pilot.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="changed premises|rendered input"):
        validate_dataset(tmp_path, set())
