import json
from pathlib import Path

import pytest

from decision_lab.training.config import load_training_config
from decision_lab.training.ledger import finish_invocation, reserve_invocation


def test_ledger_reserves_failed_invocations_and_enforces_stage_cap(tmp_path):
    training = load_training_config(Path("configs/training.yaml"))
    path = tmp_path / "ledger.json"
    index = reserve_invocation(path, "gate", 0.4, training, 1.0)
    finish_invocation(path, index, {"status": "failed"})
    reserve_invocation(path, "gate", 0.4, training, 1.0)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exhausted"):
        reserve_invocation(path, "gate", 0.4, training, 1.0)
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["invocations"][0]["status"] == "failed"


def test_ledger_blocks_spending_into_project_cap_or_credit_reserve(tmp_path):
    config = load_training_config(Path("configs/training.yaml"))
    with pytest.raises(ValueError, match="compute allocation"):
        reserve_invocation(tmp_path / "limit.json", "token", 20, config, 1.0)
    larger_reserve = config.model_copy(update={"credits_reserved_usd": 29.0})
    with pytest.raises(ValueError, match="reserved credits"):
        reserve_invocation(tmp_path / "reserve.json", "token", 0.4, larger_reserve, 1.0)
    assert list(tmp_path.iterdir()) == []


def test_study_definition_import_does_not_start_gpu_work():
    import runpy

    definition = runpy.run_path("study_app.py")
    assert definition["app"].name == "decision-model-lab-study"
    assert definition["resources"]["cpu"] == (4, 4)
    assert definition["resources"]["memory"] == (32768, 32768)


def test_explicit_operational_amendment_keeps_spend_and_provenance_guards(tmp_path):
    training = load_training_config(Path("configs/training.yaml"))
    path = tmp_path / "ledger.json"
    for _ in range(training.max_gate_calls):
        index = reserve_invocation(path, "gate", 0.4, training, 1.0)
        finish_invocation(path, index, {"status": "failed"})
    (tmp_path / "protocol.json").write_text(json.dumps({"identity": "fixture"}))
    amendment = {
        "protocol_identity": "fixture",
        "additional_calls": {"gate": 1},
        "reason": "Recover an interrupted job without changing training settings",
    }
    amendment_path = tmp_path / "operational-invocation-amendment.json"
    amendment_path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match="compute allocation"):
        reserve_invocation(path, "gate", 20, training, 1.0)
    reserve_invocation(path, "gate", 0.4, training, 1.0)
    assert json.loads(path.read_text())["operational_amendment"] == amendment
    amendment["additional_calls"]["gate"] = 2
    amendment_path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match="amendment changed"):
        reserve_invocation(path, "gate", 0.4, training, 1.0)
    amendment_path.unlink()
    with pytest.raises(ValueError, match="amendment is missing"):
        reserve_invocation(path, "gate", 0.4, training, 1.0)
