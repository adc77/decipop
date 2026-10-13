from decision_lab.model_config import PilotConfig


def pilot_preflight(config: PilotConfig, remaining_usd: float) -> dict:
    seconds = config.timeout_seconds + config.startup_timeout_seconds + config.scaledown_seconds
    cost = seconds / 3600 * config.hourly_usd
    if cost > config.max_pilot_usd:
        raise ValueError("Pilot timeout permits a cost above its configured allocation")
    if remaining_usd < cost + config.required_remaining_usd:
        raise ValueError("Insufficient credits after reserving required remaining experiments")
    return {
        "worker_hourly_usd": config.hourly_usd,
        "timeout_cost_estimate_usd": cost,
        "required_remaining_usd": config.required_remaining_usd,
        "reported_credits_remaining_usd": remaining_usd,
        "billing_verified": False,
    }
