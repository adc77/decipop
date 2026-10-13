import json
from pathlib import Path

import torch
from safetensors.torch import load_model
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel, PreTrainedTokenizerFast

from decision_lab.model_config import ScratchConfig
from decision_lab.models.decision_head import HeadDecisionModel
from decision_lab.models.scratch_transformer import ScratchDecisionModel
from decision_lab.models.token_model import TokenDecisionModel


def load_release(directory: Path, device: str = "cpu"):
    metadata = json.loads((directory / "config.json").read_text())
    architecture = metadata["architecture"]
    if architecture == "scratch":
        model = ScratchDecisionModel(ScratchConfig.model_validate(metadata["scratch"]))
    elif architecture == "token":
        model = TokenDecisionModel(
            LlamaForCausalLM(LlamaConfig(**metadata["backbone"])), metadata["answer_ids"]
        )
    elif architecture == "head":
        config = LlamaConfig(**metadata["backbone"])
        model = HeadDecisionModel(LlamaModel(config), config.hidden_size, 3)
    else:
        raise ValueError("Unsupported release architecture")
    load_model(model, directory / "model.safetensors", strict=True)
    model.to(device).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
    tokenizer.padding_side = "right"
    return model, tokenizer, metadata


def predict_release(directory: Path, prompts: list[str], device: str = "cpu"):
    model, tokenizer, metadata = load_release(directory, device)
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(device)
    if encoded.input_ids.shape[1] > metadata["max_length"]:
        raise ValueError("Release inference input exceeds the length limit")
    with torch.no_grad():
        logits = model.class_logits(model(encoded.input_ids, encoded.attention_mask)).float().cpu()
    temperature = json.loads((directory / "calibration.json").read_text())["temperature"]
    probabilities = (logits / temperature).softmax(-1)
    return {
        "class_labels": metadata["class_labels"],
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "decisions": [metadata["class_labels"][index] for index in logits.argmax(-1).tolist()],
        "precision": "FP32 artifact verification inference",
    }


def verify_release(directory: Path, device: str = "cpu"):
    from decision_lab.training.checkpoint import file_hash

    inventory = json.loads((directory / "artifact-manifest.json").read_text())
    for name, digest in inventory["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Release file checksum mismatch: {name}")
    reference = json.loads((directory / "verification.json").read_text())
    actual = predict_release(directory, reference["prompts"], device)
    torch.testing.assert_close(
        torch.tensor(actual["logits"]),
        torch.tensor(reference["logits"]),
        atol=reference["logit_atol"],
        rtol=reference["logit_rtol"],
    )
    if actual["decisions"] != reference["decisions"]:
        raise ValueError("Artifact reload changed a fixed verification decision")
    return {
        "verified": True,
        "inputs": len(reference["prompts"]),
        "device": device,
        "artifact_identity": inventory["protocol_identity"],
    }
