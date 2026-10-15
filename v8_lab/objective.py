import torch
from torch.nn import functional as F

from decision_lab.training.core import padded_batch


def paired_loss(logits, labels, coefficient):
    if logits.ndim != 2 or logits.shape[1] != 3 or len(logits) % 2:
        raise ValueError("Expected consecutive pairs of three-class logits")
    if labels.shape != (len(logits),) or coefficient < 0:
        raise ValueError("Invalid labels or loss coefficient")
    if not torch.isfinite(logits).all() or not torch.all((labels >= 0) & (labels < 3)):
        raise ValueError("Invalid logits or class indices")
    logp, logq = logits.float().log_softmax(-1).reshape(-1, 2, 3).unbind(1)
    y, other = labels.reshape(-1, 2).unbind(1)
    axes = torch.arange(3, device=logits.device).expand(len(y), 3).clone()
    indices = torch.arange(len(y), device=logits.device)
    axes[indices, y], axes[indices, other] = other, y
    aligned = logp.gather(1, axes)
    mixture = torch.logaddexp(aligned, logq) - torch.log(
        logits.new_tensor(2.0, dtype=torch.float32)
    )
    js = (
        (aligned.exp() * (aligned - mixture)).sum(-1) + (logq.exp() * (logq - mixture)).sum(-1)
    ).mean() / 2
    ce = F.cross_entropy(logits.float(), labels)
    return ce + coefficient * js, ce, js


def update(
    model, optimizer, scheduler, sequences, labels, microbatch, pad_id, clip, device, coefficient
):
    if len(sequences) % 2 or microbatch % 2:
        raise ValueError("Microbatches must preserve complete pairs")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = dict(loss=0.0, cross_entropy=0.0, consistency=0.0, input_tokens=0)
    for offset in range(0, len(sequences), microbatch):
        subset = sequences[offset : offset + microbatch]
        inputs, mask, targets = padded_batch(
            subset, labels[offset : offset + microbatch], pad_id, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model.class_logits(model(inputs, mask))
        loss, ce, js = paired_loss(logits, targets, coefficient)
        weight = len(subset) / len(sequences)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite paired loss")
        (loss * weight).backward()
        for name, value in (("loss", loss), ("cross_entropy", ce), ("consistency", js)):
            totals[name] += value.item() * weight
        totals["input_tokens"] += int(mask.sum())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite paired gradients")
    optimizer.step()
    scheduler.step()
    return totals | {"gradient_norm": norm.item()}
