import json
from pathlib import Path

import pytest
import torch

from decision_lab.publication import atomic_json
from decision_lab.studies.v3.data import generate, validate
from decision_lab.studies.v3.protocol import (
    budget,
    check_seal,
    code_hash,
    inventory,
    seal,
    settings,
)
from decision_lab.studies.v3.selection import fit_temperature, select_candidates
from decision_lab.training.checkpoint import file_hash


def small_spec():
    return settings() | {
        "selection_fit_families": 2,
        "selection_families": 2,
        "calibration_families": 2,
        "test_families": 2,
        "transfer_families": 2,
    }


def test_convex_temperature_fit_matches_grid_and_keeps_decisions():
    spec = settings()
    logits = torch.tensor([[8.0, 0.0, -1.0], [8.0, 0.0, -1.0], [0.0, 6.0, -1.0], [0.0, -1.0, 8.0]])
    labels = torch.tensor([0, 1, 1, 2])
    fitted = fit_temperature(logits, labels, spec)
    grid = torch.logspace(-1, 3, 2000, dtype=torch.float64)
    losses = torch.stack(
        [
            torch.nn.functional.cross_entropy(logits.double() / temperature, labels)
            for temperature in grid
        ]
    )
    assert fitted["calibrated_nll"] <= losses.min().item() + 1e-8
    assert fitted["calibrated_nll"] < fitted["raw_nll"]
    assert torch.equal(logits.argmax(-1), (logits / fitted["temperature"]).argmax(-1))
    uniform = fit_temperature([[4.0, 0.0, 0.0]] * 3, [0, 1, 2], spec)
    assert uniform["temperature"] == spec["temperature_max"] and uniform["at_bound"]
    perfect = fit_temperature([[5.0, 0.0, 0.0]], [0], spec)
    assert perfect["temperature"] == spec["temperature_min"] and perfect["at_bound"]


def test_policy_pool_and_early_tie_break():
    candidates = {
        "early": {"arm": "treatment", "step": 250},
        "final": {"arm": "treatment", "step": 1878},
    }
    metrics = {
        "early": {"raw_nll": 1.1, "calibrated_nll": 1.09, "accuracy": 1 / 3},
        "final": {"raw_nll": 1.8, "calibrated_nll": 0.6, "accuracy": 0.8},
    }
    chosen = select_candidates(candidates, metrics, settings()["policies"])["treatment"]
    assert chosen == {"raw_nll": "early", "calibrated_nll": "final", "accuracy": "final"}
    assert (
        select_candidates(candidates, dict.fromkeys(candidates, metrics["early"]), ["accuracy"])[
            "treatment"
        ]["accuracy"]
        == "early"
    )


def test_fresh_families_exclude_prior_worlds_and_detect_tampering(tmp_path):
    spec = small_spec()
    first = tmp_path / "first"
    generate(first, spec, set(), {})
    records = [json.loads(line) for line in (first / "families.jsonl").read_text().splitlines()]
    excluded = {digest for record in records for digest in record["world_hashes"]}
    second = tmp_path / "second"
    audit = generate(second, spec, excluded, {})
    assert audit["valid"] and audit["families"] == 14
    fresh = [json.loads(line) for line in (second / "families.jsonl").read_text().splitlines()]
    assert not excluded & {digest for record in fresh for digest in record["world_hashes"]}
    path = second / "selection.jsonl"
    path.write_text(path.read_text().replace('"label":"TRUE"', '"label":"UNKNOWN"', 1))
    with pytest.raises(ValueError, match="changed"):
        validate(second, excluded)


def test_sealed_selection_and_compute_budget():
    record = seal({"choices": {"accuracy": "final"}})
    assert check_seal(record) == record
    record["choices"]["accuracy"] = "early"
    with pytest.raises(ValueError, match="identity"):
        check_seal(record)
    allocation = budget(settings())
    assert allocation["maximum_compute_usd"] < settings()["compute_allocation_usd"]
    with pytest.raises(ValueError, match="exceed"):
        budget(settings() | {"compute_allocation_usd": 0.01})


def test_cpu_selection_freeze_test_scoring_and_verified_exports(tmp_path, monkeypatch):
    import yaml
    from safetensors.torch import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast

    from decision_lab.evaluation.export import export_release
    from decision_lab.models.decision_head import HeadDecisionModel
    from decision_lab.release import load_release, verify_release
    from decision_lab.studies.v3 import study
    from decision_lab.training.core import predict

    torch.set_num_threads(2)
    root = tmp_path / "study"
    root.mkdir()
    spec = small_spec()
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "v3.yaml").write_text(yaml.safe_dump(spec))
    for name in ("pilot.yaml", "scratch.yaml"):
        (configs / name).write_text((Path("configs") / name).read_text())
    generate(root / "data", spec, set(), {})
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
    candidates = {}
    for arm in ("control", "treatment"):
        for stage, step in (("early", 250), ("final", 1878)):
            name = f"{arm}-{stage}"
            directory = root / "candidates" / name
            directory.mkdir(parents=True)
            torch.manual_seed(17)
            model = HeadDecisionModel(LlamaModel(config), 16, 3)
            save_model(model, directory / "model.safetensors")
            tokenizer.save_pretrained(directory)
            atomic_json(
                directory / "config.json",
                {
                    "architecture": "head",
                    "backbone": config.to_dict(),
                    "class_labels": spec["class_labels"],
                    "max_length": spec["max_length"],
                },
            )
            candidates[name] = {"arm": arm, "step": step, "files": inventory(directory)}
    protocol = seal(
        {
            "settings": spec,
            "code_sha256": code_hash(),
            "data_files": inventory(root / "data"),
            "candidates": candidates,
        }
    )
    atomic_json(root / "protocol.json", protocol)
    monkeypatch.setattr(
        study, "load_release", lambda directory, _device: load_release(directory, "cpu")
    )
    monkeypatch.setattr(study, "predict", lambda *args: predict(*args[:-1], "cpu"))
    monkeypatch.setattr(
        study,
        "export_release",
        lambda *args, **_kwargs: export_release(*args, device="cpu", license_directory=Path.cwd()),
    )
    monkeypatch.setattr(
        study, "verify_release", lambda directory, _device: verify_release(directory, "cpu")
    )
    with pytest.raises(FileNotFoundError):
        study.evaluate(root, configs, lambda: None)
    frozen = study.selection(root, configs, lambda: None)
    assert not (root / "test-started.json").exists()
    assert all(
        value == f"{arm}-early"
        for arm, policies in frozen["choices"].items()
        for value in policies.values()
    )
    result = study.evaluate(root, configs, lambda: None)
    assert len(result["evaluation"]["candidates"]) == 4
    assert set(result["verification"]) == {"control-early", "treatment-early"}
    for name in result["verification"]:
        assert verify_release(root / "exports" / name / "releases/head")["verified"]
        assert (
            file_hash(root / "exports" / name / "releases/head/model.safetensors")
            == candidates[name]["files"]["model.safetensors"]
        )
    with pytest.raises(FileExistsError, match="already"):
        study.selection(root, configs, lambda: None)
    with pytest.raises(FileExistsError, match="already"):
        study.evaluate(root, configs, lambda: None)
