from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from decision_lab.studies.v3.study import metrics
from decision_lab.studies.v6.metric_audit import verify_metrics


def recorded(logits):
    labels = torch.tensor([0, 0])
    rows = [SimpleNamespace(family_id=str(i)) for i in range(2)]
    spec = {
        "ece_bins": 10,
        "high_confidence_threshold": 0.9,
        "class_labels": ["UNKNOWN", "FALSE", "TRUE"],
    }
    result = metrics(rows, {"logits": logits, "labels": labels}, 1.0, spec)
    return result, labels


def test_accepts_only_the_tied_confidence_aurc_range():
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    expected, labels = recorded(logits)
    changed = deepcopy(expected)
    changed["raw"]["aurc"] = 0.75 if expected["raw"]["aurc"] == 0.25 else 0.25
    receipt = verify_metrics(expected, changed, logits, labels, 1.0)
    assert receipt["aurc_near_tie_checks"][0]["lower"] == 0.25
    assert receipt["aurc_near_tie_checks"][0]["upper"] == 0.75
    changed["raw"]["aurc"] = 0.0
    with pytest.raises(ValueError, match="exceeds near-tie"):
        verify_metrics(expected, changed, logits, labels, 1.0)


def test_rejects_changed_aurc_without_confidence_ties():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    expected, labels = recorded(logits)
    changed = deepcopy(expected)
    changed["raw"]["aurc"] = 0.75
    with pytest.raises(ValueError, match="exceeds near-tie"):
        verify_metrics(expected, changed, logits, labels, 1.0)


@pytest.mark.parametrize("field", ["questions", "accuracy", "nll"])
def test_rejects_count_or_nonroundoff_metric_changes(field):
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    expected, labels = recorded(logits)
    changed = deepcopy(expected)
    changed["raw"][field] += 1 if field == "questions" else 1e-6
    with pytest.raises(ValueError, match="metric differs"):
        verify_metrics(expected, changed, logits, labels, 1.0)
