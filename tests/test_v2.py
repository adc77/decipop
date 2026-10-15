import json
from pathlib import Path

import pytest

from decision_lab.data.schema import Label
from decision_lab.data.validate import read_examples
from decision_lab.studies.v2.budget import allocations, finish, reserve
from decision_lab.studies.v2.data import generate, validate
from decision_lab.studies.v2.protocol import settings


def test_v2_calibration_evaluation_and_two_head_exports(tmp_path, monkeypatch):
    import shutil

    import torch
    from safetensors.torch import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast

    from decision_lab.evaluation.export import export_release
    from decision_lab.models.decision_head import HeadDecisionModel
    from decision_lab.publication import atomic_json
    from decision_lab.release import verify_release
    from decision_lab.studies.v2 import study
    from decision_lab.studies.v2.protocol import code_hash
    from decision_lab.studies.v2.report import write_report
    from decision_lab.training.core import predict
    from decision_lab.training.protocol import json_hash, make_protocol

    spec, pilot, scratch, training, evaluation = settings()
    spec = spec | {
        "train_families": 2,
        "development_families": 2,
        "calibration_families": 2,
        "test_families": 2,
        "transfer_families": 2,
    }
    evaluation = evaluation.model_copy(update={"calibration_max_iterations": 5})
    data = tmp_path / "dataset"
    generate(data, spec)
    root = tmp_path / "study"
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )

    def build(*_args):
        torch.manual_seed(17)
        return HeadDecisionModel(LlamaModel(config), 16, 3), None

    arms = {}
    for arm in ("control", "treatment"):
        local = root / arm
        shutil.copytree(data / arm, local / "data")
        (local / "audit").mkdir()
        (local / "test-inputs").mkdir()
        atomic_json(local / "test-inputs/manifest.json", {})
        protocol = make_protocol(
            local / "data",
            local / "audit",
            pilot,
            scratch,
            training,
            evaluation.model_dump(),
            Path("uv.lock"),
            local / "test-inputs",
        )
        atomic_json(local / "protocol.json", protocol)
        selected = local / "head/selected-1"
        selected.mkdir(parents=True)
        save_model(build()[0], selected / "model.safetensors")
        atomic_json(
            local / "head-state.json",
            {
                "complete": True,
                "protocol_identity": protocol["identity"],
                "selected": str(selected),
                "examples_seen": 6,
                "input_tokens": 100,
                "step": 1,
            },
        )
        arms[arm] = protocol["identity"]
    comparison = {"v2_code_sha256": code_hash(), "arms": arms}
    comparison["identity"] = json_hash(comparison)
    atomic_json(root / "comparison.json", comparison)
    monkeypatch.setattr(
        study, "settings", lambda _path: (spec, pilot, scratch, training, evaluation)
    )
    monkeypatch.setattr(study, "build_model", build)
    monkeypatch.setattr(study, "get_tokenizer", lambda *_args: tokenizer)
    monkeypatch.setattr(study, "predict", lambda *args: predict(*args[:-1], "cpu"))
    monkeypatch.setattr(study, "release_memory", lambda: None)
    monkeypatch.setattr(
        study,
        "export_release",
        lambda *args, **_kwargs: export_release(*args, device="cpu", license_directory=Path.cwd()),
    )
    result = study.evaluate(root)
    assert set(result["calibration"]["arms"]) == {"control", "treatment"}
    assert (root / "test-evaluation-started.json").exists()
    for arm in ("control", "treatment"):
        assert verify_release(root / arm / "releases/head")["verified"]
    statistics = write_report(root, tmp_path / "reports/V2_RESULTS.md")
    assert statistics["primary_unknown_accuracy"]["left_minus_right"] == 0
    with pytest.raises(FileExistsError, match="already frozen"):
        study.evaluate(root)


def test_matched_worlds_are_balanced_isolated_and_oracle_checked(tmp_path):
    spec, *_ = settings()
    spec = spec | {
        "train_families": 4,
        "development_families": 2,
        "calibration_families": 2,
        "test_families": 3,
        "transfer_families": 2,
    }
    output = tmp_path / "data"
    report = generate(output, spec)
    assert report["valid"]
    control = read_examples(output / "control/train.jsonl")
    treatment = read_examples(output / "treatment/train.jsonl")
    assert {r.family_id for r in control} == {r.family_id for r in treatment}
    for family in {r.family_id for r in treatment}:
        rows = [r for r in treatment if r.family_id == family]
        assert {r.label for r in rows} == set(Label)
        assert len({r.query for r in rows}) == 1
        assert len({r.question for r in rows}) == 1
    shared = read_examples(output / "control/test_counterfactual.jsonl")
    assert not {r.family_id for r in shared} & {r.family_id for r in treatment}
    path = output / "control/train.jsonl"
    path.write_text(path.read_text().replace('"label":"TRUE"', '"label":"UNKNOWN"', 1))
    with pytest.raises(ValueError, match="checksum"):
        validate(output)


def test_v2_allocation_keeps_failed_calls_and_limits_retries(tmp_path):
    spec, pilot, *_ = settings()
    _, maximum = allocations(spec, pilot)
    assert maximum < 5
    path = tmp_path / "ledger.json"
    i = reserve(path, "pilot", spec, pilot)
    finish(path, i, "complete")
    with pytest.raises(ValueError, match="exhausted"):
        reserve(path, "pilot", spec, pilot)
    for _ in range(4):
        i = reserve(path, "control", spec, pilot)
        finish(path, i, "complete")
    with pytest.raises(ValueError, match="successful calls"):
        reserve(path, "control", spec, pilot)
    ledger = json.loads(path.read_text())
    ledger["invocations"][1]["status"] = "failed"
    path.write_text(json.dumps(ledger))
    i = reserve(path, "control", spec, pilot)
    finish(path, i, "complete")
    assert json.loads(path.read_text())["invocations"][1]["status"] == "failed"


def test_orphan_selection_is_preserved_without_touching_durable_selection(tmp_path):
    from decision_lab.studies.v2.recovery import preserve_orphan_selections
    from decision_lab.training.checkpoint import file_hash

    parent = tmp_path / "head/attempt/checkpoint-001000"
    parent.mkdir(parents=True)
    (parent / "progress.json").write_text(json.dumps({"step": 1000}))
    (parent / "complete.json").write_text(
        json.dumps(
            {
                "identity": "study:head",
                "files": {"progress.json": file_hash(parent / "progress.json")},
            }
        )
    )
    keep = tmp_path / "head/selected-000750"
    keep.mkdir()
    orphan = tmp_path / "head/selected-001250"
    orphan.mkdir()
    (orphan / "model.safetensors").write_bytes(b"diagnostic")
    result = preserve_orphan_selections(tmp_path, str(parent), "study:head")
    assert keep.exists() and not orphan.exists() and len(result) == 1
    assert (Path(result[0]["preserved"]) / "model.safetensors").read_bytes() == b"diagnostic"
