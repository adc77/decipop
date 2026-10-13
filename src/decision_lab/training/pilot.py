import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from decision_lab.model_config import PilotConfig
from decision_lab.models.decision_head import HeadDecisionModel
from decision_lab.models.token_model import TokenDecisionModel
from decision_lab.tokenization import answer_ids, load_tokenizer


def run_gpu_pilot(config: PilotConfig, examples: list[dict], output: Path) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The pilot requires a CUDA GPU with bfloat16 support")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(config.cpu_threads)
    tokenizer = load_tokenizer(config)
    mapping = answer_ids(tokenizer, config)
    batches = []
    for start in range(0, len(examples), config.microbatch):
        rows = examples[start : start + config.microbatch]
        encoded = tokenizer(
            [row["prompt"] for row in rows], padding=True, truncation=False, return_tensors="pt"
        )
        if encoded.input_ids.shape[1] > config.max_length:
            raise ValueError("Pilot input exceeds the declared token limit")
        labels = torch.tensor(
            [config.class_labels.index(row["label"]) for row in rows], device="cuda"
        )
        batches.append((encoded.to("cuda"), labels))
    reports = {}
    for architecture in ("token", "head"):
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        loaded_at = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        if base.config.model_type != "llama":
            raise ValueError("This pilot explicitly supports the pinned Llama backbone")
        if architecture == "token":
            model = TokenDecisionModel(base, mapping)
            groups = [{"params": model.parameters(), "lr": config.backbone_lr}]
        else:
            model = HeadDecisionModel(base.model, base.config.hidden_size, len(config.class_labels))
            groups = [
                {"params": model.backbone.parameters(), "lr": config.backbone_lr},
                {"params": model.head.parameters(), "lr": config.head_lr},
            ]
        del base
        model = model.to("cuda")
        model.backbone.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - loaded_at
        losses = []
        step_seconds = []
        tokens = []
        model.train()
        for step in range(config.steps):
            encoded, labels = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(encoded.input_ids, encoded.attention_mask)
                loss = model.loss(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Pilot loss is nonfinite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("Pilot gradients are nonfinite")
            optimizer.step()
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - started)
            losses.append(loss.item())
            tokens.append(int(encoded.attention_mask.sum()))
        model.eval()
        correct = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for encoded, labels in batches:
                logits = model(encoded.input_ids, encoded.attention_mask)
                correct += int((model.class_logits(logits).argmax(-1) == labels).sum())
        measured = step_seconds[config.warmup_steps :]
        report = {
            "architecture": architecture,
            "steps": config.steps,
            "examples": len(examples),
            "losses": losses,
            "step_seconds": step_seconds,
            "load_seconds": load_seconds,
            "mean_step_seconds_after_warmup": sum(measured) / len(measured),
            "tokens_per_second_after_warmup": sum(tokens[config.warmup_steps :]) / sum(measured),
            "tiny_set_accuracy": correct / len(examples),
            "parameters": sum(p.numel() for p in model.parameters()),
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "precision": "fp32 parameters and optimizer; bfloat16 autocast",
        }
        reports[architecture] = report
        (output / f"{architecture}.json").write_text(json.dumps(report, indent=2) + "\n")
        del model, optimizer, groups, logits, loss, norm
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "config": config.model_dump(),
        "gpu": torch.cuda.get_device_name(),
        "models": reports,
        "status": "smoke-and-throughput-only",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
