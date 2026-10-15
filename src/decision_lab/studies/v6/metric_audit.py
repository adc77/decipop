import math

import torch

FLOAT_TOLERANCE = 1e-12
CONFIDENCE_TIE_TOLERANCE = 32 * torch.finfo(torch.float64).eps


def aurc_tie_bounds(logits, labels, temperature):
    confidence, chosen = (logits.double() / temperature).softmax(-1).max(-1)
    order = confidence.argsort(descending=True, stable=True)
    confidence = confidence[order].tolist()
    wrong = (chosen != labels)[order].tolist()
    lower, upper = [], []
    tied_groups = 0
    start = 0
    while start < len(wrong):
        end = start + 1
        while end < len(wrong) and confidence[start] - confidence[end] <= CONFIDENCE_TIE_TOLERANCE:
            end += 1
        errors = sum(wrong[start:end])
        count = end - start
        lower.extend([0] * (count - errors) + [1] * errors)
        upper.extend([1] * errors + [0] * (count - errors))
        tied_groups += int(count > 1)
        start = end
    denominator = torch.arange(1, len(wrong) + 1, dtype=torch.float64)
    bounds = [
        float((torch.tensor(values, dtype=torch.float64).cumsum(0) / denominator).mean())
        for values in (lower, upper)
    ]
    return {"lower": bounds[0], "upper": bounds[1], "near_tie_groups": tied_groups}


def verify_metrics(recomputed, recorded, logits, labels, temperature):
    receipt = {
        "absolute_float_tolerance": FLOAT_TOLERANCE,
        "confidence_tie_tolerance": CONFIDENCE_TIE_TOLERANCE,
        "float_roundoff_differences": [],
        "aurc_near_tie_checks": [],
    }

    def compare(left, right, path):
        if type(left) is not type(right):
            raise ValueError(f"Metric type differs: {path}")
        if isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError(f"Metric keys differ: {path}")
            for key in left:
                compare(left[key], right[key], (*path, key))
        elif isinstance(left, list):
            if len(left) != len(right):
                raise ValueError(f"Metric lengths differ: {path}")
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(a, b, (*path, str(index)))
        elif left == right:
            return
        elif isinstance(left, float) and math.isclose(
            left, right, rel_tol=0, abs_tol=FLOAT_TOLERANCE
        ):
            receipt["float_roundoff_differences"].append(
                {"field": "/".join(path), "recomputed": left, "recorded": right}
            )
        elif path[-1] == "aurc":
            scale = 1.0 if path[0] == "raw" else temperature
            bounds = aurc_tie_bounds(logits, labels, scale)
            if not bounds["lower"] - FLOAT_TOLERANCE <= right <= bounds["upper"] + FLOAT_TOLERANCE:
                raise ValueError(f"Recorded AURC exceeds near-tie order bounds: {path}")
            receipt["aurc_near_tie_checks"].append(
                {"field": "/".join(path), "recomputed": left, "recorded": right, **bounds}
            )
        else:
            raise ValueError(f"Recomputed metric differs: {path}")

    compare(recomputed, recorded, ())
    return receipt
