import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import generate_dataset
from decision_lab.evaluation import study
from decision_lab.evaluation.baselines import evaluate_baselines, prepare_baselines
from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.evaluation.export import export_release
from decision_lab.evaluation.learning import write_learning
from decision_lab.evaluation.performance import benchmark
from decision_lab.evaluation.report import write_report
from decision_lab.model_config import load_pilot_config, load_scratch_config
from decision_lab.models.decision_head import HeadDecisionModel
from decision_lab.models.scratch_transformer import ScratchDecisionModel
from decision_lab.models.token_model import TokenDecisionModel
from decision_lab.release import verify_release
from decision_lab.training.config import load_training_config
from decision_lab.training.core import predict


def test_calibration_evaluation_export_and_standalone_inference(tmp_path, monkeypatch):
    pilot = load_pilot_config(Path("configs/pilot.yaml"))
    scratch = load_scratch_config(Path("configs/scratch.yaml")).model_copy(
        update={
            "vocab_size": 32,
            "hidden_size": 16,
            "heads": 2,
            "layers": 1,
            "intermediate_size": 32,
        }
    )
    training = load_training_config(Path("configs/training.yaml"))
    evaluation = load_evaluation_config(Path("configs/evaluation.yaml")).model_copy(
        update={
            "performance_requests": 4,
            "performance_warmup": 1,
            "performance_batch": 2,
            "calibration_max_iterations": 5,
        }
    )
    config = load_dataset_config(Path("configs/smoke.yaml"))
    config = config.model_copy(
        update={
            "splits": {
                name: value.model_copy(update={"families": 2})
                for name, value in config.splits.items()
            }
        }
    )
    generate_dataset(config, tmp_path / "data")
    backend = Tokenizer(
        models.WordLevel({"[PAD]": 0, "[UNK]": 1, "A": 7, "B": 8, "C": 9}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    backbone = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )

    def build(architecture, *_args):
        torch.manual_seed(17)
        base = LlamaForCausalLM(backbone)
        if architecture == "scratch":
            model = ScratchDecisionModel(scratch)
        elif architecture == "head":
            model = HeadDecisionModel(base.model, 16, 3)
        else:
            model = TokenDecisionModel(base, [7, 8, 9])
        return model, None

    selected = {}
    for name in ("token", "head", "scratch"):
        directory = tmp_path / name / "selected"
        directory.mkdir(parents=True)
        save_model(build(name)[0], directory / "model.safetensors")
        selected[name] = {"selected": str(directory)}
    (tmp_path / "selection.json").write_text(json.dumps(selected))
    prepare_baselines(tmp_path / "data", tmp_path / "baselines", pilot.class_labels, evaluation)
    (tmp_path / "baseline-freeze.json").write_text((tmp_path / "baselines/freeze.json").read_text())
    inputs = tmp_path / "test-inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_text(
        json.dumps({"transformations": [], "performance_test_id_indices": [0, 1, 2, 3]})
    )
    protocol = {
        "identity": "fixture",
        "primary_seed": 17,
        "evaluation": evaluation.model_dump(),
        "input_files": {
            "data/calibration.jsonl": "fixture",
            "test-inputs/manifest.json": "fixture",
        },
    }
    monkeypatch.setattr(
        study, "settings", lambda _root: (protocol, pilot, scratch, training, evaluation)
    )
    monkeypatch.setattr(study, "get_tokenizer", lambda *_args: tokenizer)
    monkeypatch.setattr(study, "build_model", build)
    monkeypatch.setattr(study, "synchronize", lambda _device: None)
    monkeypatch.setattr(study, "predict", lambda *args: predict(*args[:-1], "cpu"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(study, "benchmark", lambda *args: benchmark(*args, device="cpu"))
    monkeypatch.setattr(
        study,
        "export_release",
        lambda *args: export_release(*args, device="cpu", license_directory=Path.cwd()),
    )
    frozen = study.calibrate_all(tmp_path)
    assert set(frozen["models"]) == {"pretrained", "token", "head", "scratch"}
    result = study.evaluate_all(tmp_path)
    baselines = evaluate_baselines(tmp_path / "data", tmp_path / "baselines", inputs, evaluation)
    assert set(baselines["models"]) == {"majority", "lexical"}
    assert result["calibration_identity"] == frozen["identity"]
    assert (tmp_path / "test-evaluation-started.json").exists()
    for name in ("token", "head", "scratch"):
        release = tmp_path / "releases" / name
        assert verify_release(release)["verified"]
        reference = json.loads((release / "verification.json").read_text())
        (tmp_path / "prompts.json").write_text(json.dumps(reference["prompts"]))
        output = subprocess.run(
            [sys.executable, str(release / "infer.py"), "--input", str(tmp_path / "prompts.json")],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(output.stdout)["decisions"] == reference["decisions"]
        state = {
            "history": [{"step": 1, "nll": 1.0, "accuracy": 0.33}],
            "parameters": 100,
            "examples_seen": 6,
            "input_tokens": 100,
            "step": 1,
            "worker_seconds": 1,
            "peak_allocated_gpu_bytes": 512,
        }
        (tmp_path / f"{name}-state.json").write_text(json.dumps(state))
        (tmp_path / f"{name}-attempt-fixture.json").write_text(json.dumps(state))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "ledger.json").write_text(
        json.dumps({"initial_pilot_allocation_usd": 1, "invocations": []})
    )
    write_report(tmp_path, tmp_path / "reports/RESULTS.md", baselines)
    write_learning(
        tmp_path, tmp_path / "releases", tmp_path / "data", tmp_path / "reports/LEARNING.md"
    )
    assert (tmp_path / "reports/figures/accuracy.svg").exists()
    with pytest.raises(FileExistsError, match="already frozen"):
        study.calibrate_all(tmp_path)
    with pytest.raises(FileExistsError, match="final evaluation"):
        study.evaluate_all(tmp_path)
