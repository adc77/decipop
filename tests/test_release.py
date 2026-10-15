import json

import pytest
import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from decision_lab.model_config import ScratchConfig
from decision_lab.models.decision_head import HeadDecisionModel
from decision_lab.models.scratch_transformer import ScratchDecisionModel
from decision_lab.models.token_model import TokenDecisionModel
from decision_lab.release import predict_release, verify_release
from decision_lab.training.checkpoint import file_hash


@pytest.mark.parametrize("architecture", ["token", "head", "scratch"])
def test_custom_release_loads_exact_architecture_and_detects_tampering(tmp_path, architecture):
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    scratch = ScratchConfig(
        vocab_size=32,
        hidden_size=16,
        heads=2,
        layers=1,
        intermediate_size=32,
        max_length=16,
        dropout=0.1,
        classes=3,
        pad_id=0,
        initializer_std=0.02,
    )
    base = LlamaForCausalLM(config)
    if architecture == "token":
        model = TokenDecisionModel(base, [7, 8, 9])
    elif architecture == "head":
        model = HeadDecisionModel(base.model, 16, 3)
    else:
        model = ScratchDecisionModel(scratch)
    model.eval()
    backend = Tokenizer(
        models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "red": 4, "blue": 5, "green": 6}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    tokenizer.save_pretrained(tmp_path)
    prompts = ["red blue", "green"]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = model.class_logits(model(encoded.input_ids, encoded.attention_mask))
    save_model(model, tmp_path / "model.safetensors")
    metadata = {
        "architecture": architecture,
        "backbone": config.to_dict(),
        "scratch": scratch.model_dump(),
        "answer_ids": [7, 8, 9],
        "class_labels": ["TRUE", "FALSE", "UNKNOWN"],
        "max_length": 16,
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    (tmp_path / "calibration.json").write_text(json.dumps({"temperature": 2.0}))
    actual = predict_release(tmp_path, prompts)
    torch.testing.assert_close(torch.tensor(actual["logits"]), expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.tensor(actual["probabilities"]), (expected / 2).softmax(-1))
    reference = actual | {"prompts": prompts, "logit_atol": 0, "logit_rtol": 0}
    (tmp_path / "verification.json").write_text(json.dumps(reference))
    inventory = {path.name: file_hash(path) for path in tmp_path.iterdir() if path.is_file()}
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps({"protocol_identity": "test", "files": inventory})
    )
    assert verify_release(tmp_path)["verified"]
    with (tmp_path / "model.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_release(tmp_path)


def test_evaluation_app_import_does_not_launch_gpu_work():
    import runpy

    assert (
        runpy.run_path("src/decision_lab/studies/v1/evaluate.py")["app"].name
        == "decision-model-lab-evaluation"
    )
