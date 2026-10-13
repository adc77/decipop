from pathlib import Path

import pytest
import torch

from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.evaluation.metrics import (
    abstention_metrics,
    abstention_thresholds,
    fit_temperature,
    paired_family_interval,
    scores,
)


def test_uniform_predictions_have_known_proper_scores():
    result = scores(torch.zeros(3, 3), [0, 1, 2], 1.0, 10)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["nll"] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert result["brier_sum_over_classes"] == pytest.approx(2 / 3)
    assert result["ece"] == pytest.approx(0)
    assert sum(row["count"] for row in result["ece_bins"]) == 3


def test_temperature_reduces_overconfident_nll_without_changing_decisions():
    config = load_evaluation_config(Path("configs/evaluation.yaml"))
    logits = torch.tensor([[10.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0]])
    labels = [0, 1, 2]
    fit = fit_temperature(logits, labels, config)
    assert fit["temperature"] > 1
    assert fit["calibrated_nll"] < fit["raw_nll"]
    assert torch.equal(logits.argmax(-1), (logits / fit["temperature"]).argmax(-1))


def test_abstention_uses_fixed_calibration_threshold_and_keeps_unknown_as_a_class():
    logits = torch.tensor([[5.0, 0, 0], [0.0, 0, 4], [0.0, 2, 0], [0.0, 0, 0]])
    thresholds = abstention_thresholds(logits, 1.0, [0.5, 1.0])
    result = abstention_metrics(logits, [0, 2, 1, 2], 1.0, thresholds)
    assert result["0.5"]["retained"] == 2
    assert result["0.5"]["risk"] == 0
    assert result["1.0"]["retained"] == 4


def test_paired_bootstrap_samples_families_and_identical_models_have_zero_interval():
    logits = torch.tensor([[5.0, 0, 0], [0.0, 5, 0], [0.0, 0, 5], [5.0, 0, 0]])
    result = paired_family_interval(
        logits, logits, [0, 1, 2, 2], ["a", "a", "a", "b"], 100, 0.95, 17, "nll"
    )
    assert result["families"] == 2
    assert result["interval"] == [0.0, 0.0]
    assert result["left_minus_right"] == 0
