import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from decision_lab.models.decision_head import HeadDecisionModel
from decision_lab.models.scratch_transformer import ScratchDecisionModel
from decision_lab.models.token_model import TokenDecisionModel
from decision_lab.tokenization import answer_ids, load_tokenizer


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)


def build_model(architecture: str, pilot, scratch, training, device: str):
    if architecture not in {"token", "head", "scratch"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    if architecture == "scratch":
        model = ScratchDecisionModel(scratch)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            pilot.model_id,
            revision=pilot.revision,
            dtype=torch.float32,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        if base.config.model_type != "llama":
            raise ValueError("Only the pinned Llama backbone is supported")
        if architecture == "token":
            model = TokenDecisionModel(base, answer_ids(load_tokenizer(pilot), pilot))
        else:
            model = HeadDecisionModel(base.model, base.config.hidden_size, 3)
        del base
        model.backbone.gradient_checkpointing_enable()
    model.to(device)
    if architecture == "head":
        groups = [
            {"params": model.backbone.parameters(), "lr": training.backbone_lr},
            {"params": model.head.parameters(), "lr": training.head_lr},
        ]
    else:
        rate = training.scratch_lr if architecture == "scratch" else training.backbone_lr
        groups = [{"params": model.parameters(), "lr": rate}]
    optimizer = torch.optim.AdamW(groups, weight_decay=training.weight_decay)
    return model, optimizer


def make_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).tolist()


def encode_rows(rows, tokenizer, class_labels: list[str], max_length: int):
    encoded = tokenizer([row.prompt for row in rows], truncation=False, padding=False)
    result = encoded["input_ids"]
    if any(not ids or len(ids) > max_length for ids in result):
        raise ValueError("Encoded prompt is empty or exceeds the length limit")
    labels = [class_labels.index(row.label.value) for row in rows]
    return result, labels


def get_tokenizer(architecture: str, pilot, directory: Path):
    if architecture == "scratch":
        tokenizer = PreTrainedTokenizerFast.from_pretrained(directory / "scratch-tokenizer")
        tokenizer.padding_side = "right"
        return tokenizer
    return load_tokenizer(pilot)


def padded_batch(sequences: list[list[int]], labels: list[int], pad_id: int, device: str):
    width = max(map(len, sequences))
    inputs = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for index, sequence in enumerate(sequences):
        inputs[index, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[index, : len(sequence)] = 1
    return inputs, mask, torch.tensor(labels, dtype=torch.long, device=device)


def update(model, optimizer, scheduler, sequences, labels, microbatch, pad_id, clip, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    token_count = 0
    for offset in range(0, len(sequences), microbatch):
        subset = sequences[offset : offset + microbatch]
        batch = padded_batch(subset, labels[offset : offset + microbatch], pad_id, device)
        inputs, mask, targets = batch
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(inputs, mask)
            loss = model.loss(logits, targets) * len(subset) / len(sequences)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        loss_value += loss.item()
        token_count += int(mask.sum())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite training gradients")
    optimizer.step()
    scheduler.step()
    return {"loss": loss_value, "input_tokens": token_count, "gradient_norm": norm.item()}


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def predict(model, sequences, labels, batch_size, pad_id, device):
    model.eval()
    scores = []
    outside = []
    invalid = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for start in range(0, len(sequences), batch_size):
            inputs, mask, _ = padded_batch(
                sequences[start : start + batch_size],
                labels[start : start + batch_size],
                pad_id,
                device,
            )
            logits = model(inputs, mask)
            scores.append(model.class_logits(logits).float().cpu())
            if isinstance(model, TokenDecisionModel):
                outside.append(model.outside_probability(logits).cpu())
                invalid.append(~torch.isin(logits.argmax(-1), model.answer_ids).cpu())
    result = {"logits": torch.cat(scores), "labels": torch.tensor(labels)}
    if outside:
        result["outside_probability"] = torch.cat(outside)
        result["invalid_answer"] = torch.cat(invalid)
    return result


def validation_metrics(predictions: dict) -> dict:
    logits, labels = predictions["logits"], predictions["labels"]
    return {
        "nll": torch.nn.functional.cross_entropy(logits, labels).item(),
        "accuracy": (logits.argmax(-1) == labels).float().mean().item(),
        "questions": len(labels),
    }
