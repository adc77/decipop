import json
from pathlib import Path

from decision_lab.publication import atomic_json


def allocations(settings, pilot):
    overhead = pilot.startup_timeout_seconds + pilot.scaledown_seconds

    def cost(seconds):
        return (seconds + overhead) / 3600 * pilot.hourly_usd

    prices = {
        "pilot": cost(settings["pilot_timeout_seconds"]),
        "control": cost(settings["training_timeout_seconds"]),
        "treatment": cost(settings["training_timeout_seconds"]),
        "evaluation": cost(settings["evaluation_timeout_seconds"]),
        "verification": cost(settings["verification_timeout_seconds"]),
    }
    maximum = (
        prices["pilot"]
        + prices["evaluation"]
        + prices["verification"]
        + (2 * settings["max_training_calls_per_arm"] + settings["retry_calls_reserved"])
        * prices["control"]
    )
    if maximum > settings["compute_allocation_usd"]:
        raise ValueError("Worst-case invocation allocation exceeds the v2 compute limit")
    return prices, maximum


def reserve(path: Path, stage: str, settings, pilot):
    prices, maximum = allocations(settings, pilot)
    if stage not in prices:
        raise ValueError("Unknown v2 budget stage")
    ledger = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "compute_limit_usd": settings["compute_allocation_usd"],
            "worst_case_allocation_usd": maximum,
            "invocations": [],
        }
    )
    if ledger["compute_limit_usd"] != settings["compute_allocation_usd"]:
        raise ValueError("Existing budget limit differs")
    calls = ledger["invocations"]
    count = sum(row["stage"] == stage for row in calls)
    if stage in {"control", "treatment"}:
        total = sum(row["stage"] in {"control", "treatment"} for row in calls)
        base_limit = settings["max_training_calls_per_arm"]
        if total >= 2 * base_limit + settings["retry_calls_reserved"]:
            raise ValueError("Training invocation allowance exhausted")
        if count >= base_limit:
            failed = sum(row["stage"] == stage and row["status"] == "failed" for row in calls)
            if not failed or count >= base_limit + settings["retry_calls_reserved"]:
                raise ValueError("Arm needs more successful calls than forecast allows")
    elif count:
        raise ValueError(f"Invocation allowance exhausted for {stage}")
    allocated = sum(row["maximum_usd"] for row in calls) + prices[stage]
    retained = sum(
        prices[name]
        for name in ("evaluation", "verification")
        if name != stage and not any(row["stage"] == name for row in calls)
    )
    if allocated + retained > settings["compute_allocation_usd"]:
        raise ValueError("Call would consume required evaluation/verification allocation")
    calls.append({"stage": stage, "maximum_usd": prices[stage], "status": "reserved"})
    atomic_json(path, ledger)
    return len(calls) - 1


def finish(path: Path, index, status, **metadata):
    ledger = json.loads(path.read_text())
    ledger["invocations"][index] |= {"status": status, **metadata}
    atomic_json(path, ledger)
