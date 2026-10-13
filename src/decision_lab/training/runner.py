import json
import math
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_model

from decision_lab.data.validate import read_examples
from decision_lab.training.checkpoint import file_hash, load_checkpoint, save_checkpoint
from decision_lab.training.core import (
    build_model,
    encode_rows,
    epoch_order,
    get_tokenizer,
    make_scheduler,
    predict,
    seed_everything,
    synchronize,
    update,
    validation_metrics,
)


def verify_inputs(root: Path, protocol: dict):
    for name, digest in protocol["input_files"].items():
        path = root / name
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError(f"Study input hash mismatch: {name}")


def prune_recovery(root: Path):
    checkpoints = []
    for path in root.glob("attempt-*/checkpoint-*"):
        if (path / "complete.json").exists():
            progress = json.loads((path / "progress.json").read_text())
            checkpoints.append((progress["step"], path, progress["selected"]))
    checkpoints.sort(key=lambda item: (item[0], str(item[1])))
    retained = checkpoints[-2:]
    keep_selected = {item[2] for item in retained}
    for _, path, _ in checkpoints[:-2]:
        shutil.rmtree(path)
    for path in root.glob("selected-*"):
        if str(path) not in keep_selected:
            shutil.rmtree(path)


def run_training_chunk(
    root, architecture, pilot, scratch, training, protocol, attempt, parent, device="cuda"
):
    if architecture not in {"token", "head", "scratch"}:
        raise ValueError("Invalid training architecture")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Unsupported training device")
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Training requires a BF16-capable CUDA GPU")
    verify_inputs(root, protocol)
    directory = root / architecture / attempt
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(pilot.cpu_threads)
    seed_everything(training.seed, training.deterministic)
    tokenizer = get_tokenizer(architecture, pilot, root / "audit")
    rows = read_examples(root / "data/train.jsonl")
    development = read_examples(root / "data/development.jsonl")
    sequences, labels = encode_rows(rows, tokenizer, pilot.class_labels, pilot.max_length)
    dev_sequences, dev_labels = encode_rows(
        development, tokenizer, pilot.class_labels, pilot.max_length
    )
    steps_per_epoch = math.ceil(len(rows) / training.effective_batch)
    total_steps = steps_per_epoch * training.epochs
    model, optimizer = build_model(architecture, pilot, scratch, training, device)
    scheduler = make_scheduler(optimizer, total_steps, training.warmup_fraction)
    identity = protocol["identity"] + ":" + architecture
    progress = {
        "epoch": 0,
        "offset": 0,
        "step": 0,
        "examples_seen": 0,
        "input_tokens": 0,
        "selected": None,
        "best_nll": None,
        "best_accuracy": None,
        "history": [],
    }
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_relative_to(root / architecture):
            raise ValueError("Parent checkpoint is outside this architecture's study directory")
        progress = load_checkpoint(parent_path, model, optimizer, scheduler, identity)
    (directory / "parent.json").write_text(json.dumps({"checkpoint": parent}) + "\n")
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    microbatch = (
        training.scratch_microbatch if architecture == "scratch" else training.pretrained_microbatch
    )
    records = []
    trained_at = time.perf_counter()
    initial_step = progress["step"]
    evaluated_step = -1

    def evaluate_and_select():
        nonlocal evaluated_step
        synchronize(device)
        before = time.perf_counter()
        metrics = validation_metrics(
            predict(
                model,
                dev_sequences,
                dev_labels,
                training.evaluation_batch,
                tokenizer.pad_token_id,
                device,
            )
        )
        synchronize(device)
        metrics |= {"step": progress["step"], "seconds": time.perf_counter() - before}
        progress["history"].append(metrics)
        best = progress["best_nll"]
        if (
            best is None
            or metrics["nll"] < best
            or (metrics["nll"] == best and metrics["accuracy"] > progress["best_accuracy"])
        ):
            selected = root / architecture / f"selected-{progress['step']:06d}"
            selected.mkdir(exist_ok=False)
            save_model(model, selected / "model.safetensors")
            (selected / "selection.json").write_text(json.dumps(metrics, indent=2) + "\n")
            progress["selected"] = str(selected)
            progress["best_nll"] = metrics["nll"]
            progress["best_accuracy"] = metrics["accuracy"]
        evaluated_step = progress["step"]
        print(json.dumps({"architecture": architecture, "development": metrics}), flush=True)

    while progress["epoch"] < training.epochs:
        order = epoch_order(len(rows), training.seed, progress["epoch"])
        if "order" in progress and progress["order"] != order:
            raise ValueError("Checkpoint data order disagrees with the declared seed")
        indices = order[progress["offset"] : progress["offset"] + training.effective_batch]
        if not indices:
            raise ValueError("Training cursor does not point to an effective batch")
        synchronize(device)
        before = time.perf_counter()
        record = update(
            model,
            optimizer,
            scheduler,
            [sequences[i] for i in indices],
            [labels[i] for i in indices],
            microbatch,
            tokenizer.pad_token_id,
            training.gradient_clip,
            device,
        )
        synchronize(device)
        record |= {"seconds": time.perf_counter() - before, "examples": len(indices)}
        records.append(record)
        progress["step"] += 1
        progress["offset"] += len(indices)
        progress["examples_seen"] += len(indices)
        progress["input_tokens"] += record["input_tokens"]
        if progress["offset"] == len(rows):
            progress["epoch"] += 1
            progress["offset"] = 0
            progress.pop("order", None)
        if progress["step"] % training.evaluation_interval == 0:
            evaluate_and_select()
        if progress["step"] % 50 == 0:
            print(
                json.dumps(
                    {"architecture": architecture, "step": progress["step"], "loss": record["loss"]}
                ),
                flush=True,
            )
        if (
            progress["step"] - initial_step >= training.chunk_steps
            or time.perf_counter() - trained_at >= training.chunk_training_seconds
        ):
            break
    complete = progress["epoch"] == training.epochs
    if complete and evaluated_step != progress["step"]:
        evaluate_and_select()
    if not complete:
        progress["order"] = epoch_order(len(rows), training.seed, progress["epoch"])
    checkpoint = directory / f"checkpoint-{progress['step']:06d}"
    checkpoint_started = time.perf_counter()
    save_checkpoint(checkpoint, model, optimizer, scheduler, progress, identity)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    if complete:
        final = root / architecture / "final"
        final.mkdir(exist_ok=False)
        save_model(model, final / "model.safetensors")
    result = {
        "architecture": architecture,
        "attempt": attempt,
        "parent": parent,
        "checkpoint": str(checkpoint),
        "selected": progress["selected"],
        "complete": complete,
        "step": progress["step"],
        "total_steps": total_steps,
        "epoch": progress["epoch"],
        "examples_seen": progress["examples_seen"],
        "input_tokens": progress["input_tokens"],
        "history": progress["history"],
        "updates": records,
        "setup_seconds": setup_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "worker_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "protocol_identity": protocol["identity"],
    }
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
