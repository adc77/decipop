import statistics
import time

import torch

from decision_lab.training.core import padded_batch, synchronize


def benchmark(model, sequences, labels, pad_id, config, hourly_usd, indices, device="cuda"):
    if len(indices) != config.performance_requests:
        raise ValueError("Insufficient requests for the benchmark")
    chosen = [sequences[index] for index in indices]
    targets = [labels[index] for index in indices]
    model.eval()
    elapsed = []
    preparation = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for index in range(config.performance_warmup + len(chosen)):
            preparation_started = time.perf_counter()
            inputs, mask, _ = padded_batch(
                [chosen[index % len(chosen)]], [targets[index % len(chosen)]], pad_id, device
            )
            synchronize(device)
            if index >= config.performance_warmup:
                preparation.append(time.perf_counter() - preparation_started)
            before = time.perf_counter()
            model(inputs, mask)
            synchronize(device)
            if index >= config.performance_warmup:
                elapsed.append(time.perf_counter() - before)
        batch_size = config.performance_batch
        width = max(map(len, chosen))
        batches = []
        preparation_started = time.perf_counter()
        for offset in range(0, len(chosen) - batch_size + 1, batch_size):
            inputs, mask, _ = padded_batch(
                chosen[offset : offset + batch_size],
                targets[offset : offset + batch_size],
                pad_id,
                device,
            )
            inputs = torch.nn.functional.pad(inputs, (0, width - inputs.shape[1]), value=pad_id)
            mask = torch.nn.functional.pad(mask, (0, width - mask.shape[1]))
            batches.append((inputs, mask))
        synchronize(device)
        batched_preparation_seconds = time.perf_counter() - preparation_started
        for index in range(config.performance_warmup):
            model(*batches[index % len(batches)])
        synchronize(device)
        before = time.perf_counter()
        for inputs, mask in batches:
            model(inputs, mask)
        synchronize(device)
        batch_seconds = time.perf_counter() - before
    sorted_times = sorted(elapsed)
    p95 = sorted_times[min(len(sorted_times) - 1, int(0.95 * len(sorted_times)))]
    requests = len(batches) * batch_size
    return {
        "forward_only_excludes_tokenization_and_transfers": True,
        "batch_one_mean_preparation_ms": statistics.mean(preparation) * 1000,
        "batched_preparation_seconds": batched_preparation_seconds,
        "preparation_includes_allocation_padding_and_host_to_device_copy": True,
        "batch_one_requests": len(elapsed),
        "batch_one_median_ms": statistics.median(elapsed) * 1000,
        "batch_one_p95_ms": p95 * 1000,
        "batch_one_cost_per_1000_usd": sum(elapsed) / len(elapsed) * hourly_usd / 3600 * 1000,
        "throughput_batch": batch_size,
        "throughput_requests": requests,
        "throughput_seconds": batch_seconds,
        "decisions_per_second": requests / batch_seconds,
        "batch_cost_per_1000_usd": batch_seconds / requests * hourly_usd / 3600 * 1000,
        "token_lengths": {"min": min(map(len, chosen)), "max": width},
        "fixed_batch_padding_length": width,
        "selected_input_indices": indices,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    }
