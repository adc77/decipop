import torch
from torch.nn import functional as F


def fit_temperature(logits, labels, spec):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if logits.shape != (len(labels), 3) or not len(labels) or not torch.isfinite(logits).all():
        raise ValueError("Temperature fitting requires finite, nonempty three-class logits")
    lower = 1 / spec["temperature_max"]
    upper = 1 / spec["temperature_min"]
    truth = logits.gather(1, labels[:, None]).squeeze(1)

    def derivative(beta):
        return ((logits * (logits * beta).softmax(-1)).sum(-1) - truth).mean().item()

    if derivative(lower) >= 0:
        beta = lower
    elif derivative(upper) <= 0:
        beta = upper
    else:
        for _ in range(spec["temperature_iterations"]):
            midpoint = (lower + upper) / 2
            if derivative(midpoint) > 0:
                upper = midpoint
            else:
                lower = midpoint
        beta = (lower + upper) / 2
    temperature = 1 / beta
    return {
        "temperature": temperature,
        "bounds": [spec["temperature_min"], spec["temperature_max"]],
        "at_bound": temperature in (spec["temperature_min"], spec["temperature_max"]),
        "calibration_questions": len(labels),
        "raw_nll": F.cross_entropy(logits, labels).item(),
        "calibrated_nll": F.cross_entropy(logits * beta, labels).item(),
        "optimizer": "convex inverse-temperature derivative bisection",
    }


def select_candidates(candidates, metrics, policies):
    selected = {}
    for arm in sorted({candidate["arm"] for candidate in candidates.values()}):
        pool = [name for name, candidate in candidates.items() if candidate["arm"] == arm]
        selected[arm] = {}
        for policy in policies:

            def order(name, policy=policy):
                score = metrics[name][policy]
                return (-score if policy == "accuracy" else score, candidates[name]["step"])

            selected[arm][policy] = min(pool, key=order)
    return selected
