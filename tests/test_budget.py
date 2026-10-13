from pathlib import Path

import pytest

from decision_lab.budget import pilot_preflight
from decision_lab.model_config import load_pilot_config


def test_worker_rate_and_reserved_budget():
    config = load_pilot_config(Path("configs/pilot.yaml"))
    result = pilot_preflight(config, remaining_usd=30.0)
    assert result["worker_hourly_usd"] == pytest.approx(1.243584)
    assert result["timeout_cost_estimate_usd"] < 0.5
    assert not result["billing_verified"]
    with pytest.raises(ValueError, match="Insufficient credits"):
        pilot_preflight(config, remaining_usd=10.0)
    with pytest.raises(ValueError, match="allocation"):
        pilot_preflight(config.model_copy(update={"timeout_seconds": 3600}), remaining_usd=30.0)
