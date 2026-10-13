import math

import torch
from torch.nn import functional as F


def scores(
    logits, labels, temperature: float, bins: int, high_confidence_threshold: float = 0.9
) -> dict:
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if logits.shape != (len(labels), 3) or len(labels) == 0 or temperature <= 0:
        raise ValueError("Metrics require nonempty three-class logits and positive temperature")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite evaluation logits")
    scaled = logits / temperature
    probabilities = scaled.softmax(-1)
    confidence, prediction = probabilities.max(-1)
    correct = prediction == labels
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), prediction.tolist(), strict=True):
        confusion[actual, predicted] += 1
    f1s = []
    for index in range(3):
        true_positive = confusion[index, index].item()
        denominator = confusion[index].sum().item() + confusion[:, index].sum().item()
        f1s.append(2 * true_positive / denominator if denominator else 0.0)
    reliability = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & (
            (confidence <= upper) if index == bins - 1 else (confidence < upper)
        )
        count = int(selected.sum())
        mean_confidence = confidence[selected].mean().item() if count else None
        accuracy = correct[selected].double().mean().item() if count else None
        if count:
            ece += count / len(labels) * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    order = confidence.argsort(descending=True, stable=True)
    risks = (~correct[order]).double().cumsum(0) / torch.arange(1, len(labels) + 1)
    return {
        "questions": len(labels),
        "class_counts": torch.bincount(labels, minlength=3).tolist(),
        "accuracy": correct.double().mean().item(),
        "macro_f1": sum(f1s) / 3,
        "nll": F.cross_entropy(scaled, labels).item(),
        "brier_sum_over_classes": ((probabilities - F.one_hot(labels, 3)) ** 2)
        .sum(-1)
        .mean()
        .item(),
        "ece": ece,
        "ece_bins": reliability,
        "confusion_matrix_actual_rows": confusion.tolist(),
        "aurc": risks.mean().item(),
        "aurc_convention": (
            "mean risk at retained counts 1..N; stable input order breaks confidence ties"
        ),
        "risk_coverage": [
            {"retained": index + 1, "coverage": (index + 1) / len(labels), "risk": risk}
            for index, risk in enumerate(risks.tolist())
        ],
        "high_confidence_wrong": int(((confidence >= high_confidence_threshold) & ~correct).sum()),
        "high_confidence_threshold": high_confidence_threshold,
    }


def fit_temperature(logits, labels, config) -> dict:
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=config.calibration_learning_rate,
        max_iter=config.calibration_max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(
            config.calibration_min_temperature, config.calibration_max_temperature
        )
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    candidate = (
        log_temperature.detach()
        .exp()
        .clamp(config.calibration_min_temperature, config.calibration_max_temperature)
        .item()
    )
    candidates = [
        1.0,
        candidate,
        config.calibration_min_temperature,
        config.calibration_max_temperature,
    ]
    temperature = min(candidates, key=lambda value: F.cross_entropy(logits / value, labels).item())
    return {
        "temperature": temperature,
        "calibration_questions": len(labels),
        "raw_nll": F.cross_entropy(logits, labels).item(),
        "calibrated_nll": F.cross_entropy(logits / temperature, labels).item(),
        "bounds": [config.calibration_min_temperature, config.calibration_max_temperature],
    }


def abstention_thresholds(logits, temperature: float, targets: list[float]) -> dict:
    confidence = (
        (torch.as_tensor(logits, dtype=torch.float64) / temperature).softmax(-1).max(-1).values
    )
    descending = confidence.sort(descending=True).values
    return {
        str(target): descending[math.ceil(len(confidence) * target) - 1].item()
        for target in targets
    }


def abstention_metrics(logits, labels, temperature: float, thresholds: dict) -> dict:
    confidence, predicted = (
        (torch.as_tensor(logits, dtype=torch.float64) / temperature).softmax(-1).max(-1)
    )
    labels = torch.as_tensor(labels)
    result = {}
    for target, threshold in thresholds.items():
        retained = confidence >= threshold
        count = int(retained.sum())
        result[target] = {
            "threshold_from_calibration": threshold,
            "retained": count,
            "coverage": count / len(labels),
            "risk": (predicted[retained] != labels[retained]).double().mean().item()
            if count
            else None,
        }
    return result


def paired_family_interval(left, right, labels, families, repetitions, confidence, seed, metric):
    import numpy as np

    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if metric == "accuracy":
        differences = (
            (left.argmax(-1) == labels).double() - (right.argmax(-1) == labels).double()
        ).numpy()
    elif metric == "nll":
        differences = (
            F.cross_entropy(left, labels, reduction="none")
            - F.cross_entropy(right, labels, reduction="none")
        ).numpy()
    else:
        raise ValueError("Unsupported paired bootstrap metric")
    grouped = {}
    for family, difference in zip(families, differences, strict=True):
        grouped.setdefault(family, []).append(difference)
    sums = np.array([sum(values) for values in grouped.values()])
    counts = np.array([len(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(grouped), len(grouped))
        draws.append(sums[selected].sum() / counts[selected].sum())
    tail = (1 - confidence) / 2
    return {
        "left_minus_right": float(differences.mean()),
        "interval": np.quantile(draws, [tail, 1 - tail]).tolist(),
        "confidence": confidence,
        "families": len(grouped),
        "repetitions": repetitions,
        "uncertainty": "test-family sampling only; one training seed",
    }
