import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from decision_lab.data.schema import Label
from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from v4_lab import study
from v4_lab.data import generate, validate
from v4_lab.protocol import budget, settings


def small_spec():
    return settings() | {
        "diagnostic_families": 3,
        "train_families": 4,
        "selection_fit_families": 2,
        "selection_families": 2,
        "calibration_families": 2,
        "test_families": 2,
        "transfer_families": 2,
    }


def tiny_parent(directory):
    from safetensors.torch import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast

    from decision_lab.models.decision_head import HeadDecisionModel

    directory.mkdir(parents=True)
    config = LlamaConfig(
        vocab_size=2,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
    )
    torch.manual_seed(317)
    model = HeadDecisionModel(LlamaModel(config), 16, 3)
    save_model(model, directory / "model.safetensors")
    atomic_json(
        directory / "config.json",
        {
            "architecture": "head",
            "backbone": config.to_dict(),
            "class_labels": ["TRUE", "FALSE", "UNKNOWN"],
            "max_length": 512,
        },
    )
    atomic_json(directory / "calibration.json", {"temperature": 4.79268})
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    tokenizer.save_pretrained(directory)
    return model


def test_diagnostic_isolates_entity_and_invariance_interventions(tmp_path):
    directory = tmp_path / "diagnostic"
    audit = generate(directory, small_spec(), set(), {}, diagnostic=True)
    assert audit["questions"] == 24
    groups = {}
    for row in read_examples(directory / "diagnostic.jsonl"):
        groups.setdefault(row.family_id, {})[row.example_id.rsplit("-", 1)[1]] = row
    for rows in groups.values():
        for base, changed in (
            ("base_true", "query_unknown_true"),
            ("base_false", "query_unknown_false"),
        ):
            assert rows[base].theory == rows[changed].theory
            assert rows[base].facts == rows[changed].facts
            assert rows[base].rules == rows[changed].rules
            assert rows[base].query.predicate == rows[changed].query.predicate
            assert rows[base].query.negative == rows[changed].query.negative
            assert rows[changed].label == Label.UNKNOWN
        assert rows["renamed_true"].theory_hash == rows["base_true"].theory_hash
        assert rows["renamed_unknown"].label == rows["query_unknown_true"].label
        assert rows["distractor_true"].label == rows["base_true"].label
        assert len(rows["distractor_true"].theory.facts) == len(rows["base_true"].theory.facts) + 1
        assert len(rows["premise_unknown"].theory.facts) == len(rows["base_true"].theory.facts)


def test_training_is_matched_and_excludes_diagnostic_worlds(tmp_path):
    diagnostic = tmp_path / "diagnostic"
    generate(diagnostic, small_spec(), set(), {}, diagnostic=True)
    records = [json.loads(s) for s in (diagnostic / "families.jsonl").read_text().splitlines()]
    excluded = {h for r in records for h in r["world_hashes"]}
    training = tmp_path / "training"
    audit = generate(training, small_spec(), excluded, {})
    assert audit["valid"]
    control = read_examples(training / "train_control.jsonl")
    treatment = read_examples(training / "train_treatment.jsonl")
    assert len(control) == len(treatment) == 12
    assert [(r.family_id, r.label) for r in control] == [(r.family_id, r.label) for r in treatment]
    assert Counter(r.label for r in control) == Counter({label: 4 for label in Label})
    assert sum(a.prompt == b.prompt for a, b in zip(control, treatment, strict=True)) == 8
    path = training / "selection.jsonl"
    path.write_text(path.read_text().replace('"label":"TRUE"', '"label":"UNKNOWN"', 1))
    with pytest.raises(ValueError, match="checksum"):
        validate(training, excluded)


def test_compute_ceiling_includes_all_calls_and_recovery():
    spec = settings()
    allocation = budget(spec)
    assert 3.9 < allocation["maximum_compute_usd"] < 4.0
    assert allocation["hourly_requested_usd"] == pytest.approx(1.2452)
    with pytest.raises(ValueError, match="exceed"):
        budget(spec | {"compute_allocation_usd": 3.0})


def test_cpu_diagnostics_and_training_resume_match_uninterrupted(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    base_spec = small_spec() | {
        "epochs": 2,
        "effective_batch": 4,
        "microbatch": 2,
        "evaluation_batch": 8,
        "candidate_steps": [0, 2, 4, 6],
        "chunk_steps": 3,
    }
    first = tmp_path / "first"
    first.mkdir()
    tiny_parent(first / "parent")
    generate(first / "diagnostic", base_spec, set(), {}, diagnostic=True)
    generate(first / "training", base_spec, set(), {})

    def verified(root, phase):
        return {
            "identity": "test-v4",
            "settings": base_spec,
            "parent_files": {"model.safetensors": file_hash(root / "parent/model.safetensors")},
        }

    monkeypatch.setattr(study, "verify", verified)
    diagnostic = study.diagnose(first, lambda: None, "cpu")
    assert len(diagnostic["roles"]) == 8
    assert all(p["families"] == 3 for p in diagnostic["pairs"].values())
    assert diagnostic["pairs"]["base_true-to-query_unknown_true"]["oracle_label_changes"] == 3
    parent_hash = file_hash(first / "parent/model.safetensors")
    part = study.train_chunk(first, "control", lambda: None, "cpu")
    assert part["step"] == 3 and not part["complete"] and part["presentations"] == 12
    complete = study.train_chunk(first, "control", lambda: None, "cpu")
    assert complete["complete"] and complete["step"] == 6 and complete["presentations"] == 24
    assert file_hash(first / "parent/model.safetensors") == parent_hash
    assert sorted(p.name for p in (first / "recovery/control").iterdir()) == ["step-6"]
    final = first / "candidates/control-step6/model.safetensors"
    assert file_hash(final) != parent_hash

    second = tmp_path / "second"
    second.mkdir()
    tiny_parent(second / "parent")
    generate(second / "training", base_spec, set(), {})
    base_spec["chunk_steps"] = 6
    whole = study.train_chunk(second, "control", lambda: None, "cpu")
    assert whole["complete"]
    assert file_hash(final) == file_hash(second / "candidates/control-step6/model.safetensors")
    with pytest.raises(ValueError, match="already complete"):
        study.train_chunk(first, "control", lambda: None, "cpu")


def test_cpu_selection_calibration_scoring_and_export_round_trip(tmp_path, monkeypatch):
    from decision_lab.release import verify_release
    from decision_lab.v3.protocol import check_seal

    torch.set_num_threads(2)
    spec = small_spec() | {
        "epochs": 2,
        "effective_batch": 4,
        "microbatch": 2,
        "evaluation_batch": 8,
        "candidate_steps": [0, 2, 4, 6],
        "chunk_steps": 6,
    }
    root = tmp_path / "study"
    root.mkdir()
    tiny_parent(root / "parent")
    generate(root / "training", spec, set(), {})
    protocol = {
        "identity": "test-v4",
        "settings": spec,
        "parent_files": {"model.safetensors": file_hash(root / "parent/model.safetensors")},
    }
    monkeypatch.setattr(study, "verify", lambda *args: protocol)
    for arm in ("control", "treatment"):
        assert study.train_chunk(root, arm, lambda: None, "cpu")["complete"]
    selected = study.select(root, lambda: None, "cpu")
    assert check_seal(selected) == selected
    assert len(selected["candidates"]) == 8
    assert set(selected["calibrations"]) == {
        choices["calibrated_nll"] for choices in selected["choices"].values()
    }
    result = study.evaluate(root, lambda: None, Path("configs"), "cpu")
    assert set(result["evaluation"]["arms"]) == {"control", "treatment"}
    assert (
        json.loads((root / "test-started.json").read_text())["selection_identity"]
        == selected["identity"]
    )
    for arm in ("control", "treatment"):
        assert result["verification"][arm]["verified"]
        assert verify_release(root / "exports" / arm / "releases/head", "cpu")["inputs"] == 6
    with pytest.raises(FileExistsError, match="already started"):
        study.evaluate(root, lambda: None, Path("configs"), "cpu")
