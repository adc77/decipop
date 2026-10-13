import gc
import json
import math
import random
import shutil
import time

import torch

from decision_lab.data.validate import read_examples
from decision_lab.training.checkpoint import load_checkpoint, save_checkpoint
from decision_lab.training.core import (
    build_model,
    encode_rows,
    get_tokenizer,
    make_scheduler,
    predict,
    seed_everything,
    synchronize,
    update,
    validation_metrics,
)


def run_gpu_gate(root, pilot, scratch, training, output):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The training gate requires a BF16-capable CUDA GPU")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(pilot.cpu_threads)
    all_rows = read_examples(root / "data/train.jsonl")
    longest = sorted(all_rows, key=lambda row: len(row.prompt), reverse=True)[:2]
    reserved = {row.example_id for row in longest}
    rows = longest + random.Random(training.seed).sample(
        [row for row in all_rows if row.example_id not in reserved], training.gate_examples - 2
    )
    development = read_examples(root / "data/development.jsonl")
    reports = {}
    for architecture in ("token", "head", "scratch"):
        seed_everything(training.seed, training.deterministic)
        tokenizer = get_tokenizer(architecture, pilot, root / "audit")
        sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
        dev_sequences, dev_labels = encode_rows(
            development, tokenizer, pilot.class_labels, pilot.max_length
        )
        microbatch = (
            training.scratch_microbatch
            if architecture == "scratch"
            else training.pretrained_microbatch
        )
        steps = training.scratch_gate_steps if architecture == "scratch" else training.gate_steps
        model, optimizer = build_model(architecture, pilot, scratch, training, "cuda")
        scheduler = make_scheduler(optimizer, steps + 2, training.warmup_fraction)
        torch.cuda.reset_peak_memory_stats()
        records = []
        for step in range(steps):
            synchronize("cuda")
            started = time.perf_counter()
            record = update(
                model,
                optimizer,
                scheduler,
                sequences,
                labels,
                microbatch,
                tokenizer.pad_token_id,
                training.gradient_clip,
                "cuda",
            )
            synchronize("cuda")
            record["seconds"] = time.perf_counter() - started
            records.append(record)
            if (step + 1) % 100 == 0:
                print(
                    json.dumps({"gate": architecture, "step": step + 1, "loss": record["loss"]}),
                    flush=True,
                )
        tiny = validation_metrics(
            predict(
                model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
            )
        )
        synchronize("cuda")
        started = time.perf_counter()
        checkpoint = output / f"{architecture}-checkpoint"
        identity = "gate:" + architecture
        save_checkpoint(checkpoint, model, optimizer, scheduler, {"step": steps}, identity)
        save_seconds = time.perf_counter() - started
        expected = update(
            model,
            optimizer,
            scheduler,
            sequences,
            labels,
            microbatch,
            tokenizer.pad_token_id,
            training.gradient_clip,
            "cuda",
        )
        expected_logits = predict(
            model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
        )["logits"]
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        model, optimizer = build_model(architecture, pilot, scratch, training, "cuda")
        scheduler = make_scheduler(optimizer, steps + 2, training.warmup_fraction)
        started = time.perf_counter()
        load_checkpoint(checkpoint, model, optimizer, scheduler, identity)
        synchronize("cuda")
        load_seconds = time.perf_counter() - started
        actual = update(
            model,
            optimizer,
            scheduler,
            sequences,
            labels,
            microbatch,
            tokenizer.pad_token_id,
            training.gradient_clip,
            "cuda",
        )
        actual_logits = predict(
            model, sequences, labels, training.evaluation_batch, tokenizer.pad_token_id, "cuda"
        )["logits"]
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        if actual != expected:
            raise RuntimeError("CUDA checkpoint resume changed the next training update")
        synchronize("cuda")
        started = time.perf_counter()
        dev = validation_metrics(
            predict(
                model,
                dev_sequences,
                dev_labels,
                training.evaluation_batch,
                tokenizer.pad_token_id,
                "cuda",
            )
        )
        synchronize("cuda")
        dev_seconds = time.perf_counter() - started
        measured = records[training.gate_warmup_steps :]
        mean_step = sum(row["seconds"] for row in measured) / len(measured)
        total_steps = math.ceil(len(all_rows) / training.effective_batch) * training.epochs
        reports[architecture] = {
            "steps": steps,
            "effective_batch": len(sequences),
            "microbatch": microbatch,
            "mean_optimizer_step_seconds": mean_step,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "tiny": tiny,
            "development": dev,
            "development_seconds": dev_seconds,
            "checkpoint_save_seconds": save_seconds,
            "checkpoint_load_seconds": load_seconds,
            "cuda_resume_next_update_identical": True,
            "preliminary_training_seconds": total_steps * mean_step,
            "updates": records,
        }
        (output / f"{architecture}.json").write_text(
            json.dumps(reports[architecture], indent=2) + "\n"
        )
        print(
            json.dumps({"gate": architecture, "tiny": tiny, "optimizer_step_seconds": mean_step}),
            flush=True,
        )
        shutil.rmtree(checkpoint)
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "models": reports,
        "scratch_fit_passed": reports["scratch"]["tiny"]["accuracy"] >= training.gate_accuracy,
        "gpu": torch.cuda.get_device_name(),
        "training_config": training.model_dump(),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
