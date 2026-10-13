import json
import os
from pathlib import Path


def reserve_invocation(
    path: Path, stage: str, maximum_usd: float, training, initial_pilots_usd: float
):
    if path.exists():
        ledger = json.loads(path.read_text())
        if ledger["initial_pilot_allocation_usd"] != initial_pilots_usd:
            raise ValueError("Existing ledger has a different initial pilot allocation")
    else:
        ledger = {
            "starting_verified_credits_usd": 30.0,
            "initial_pilot_allocation_usd": initial_pilots_usd,
            "invocations": [],
            "account_usage_reconciled": False,
        }
    limits = {
        "gate": training.max_gate_calls,
        "token": training.max_training_calls_per_model,
        "head": training.max_training_calls_per_model,
        "scratch": training.max_training_calls_per_model,
        "evaluation": training.max_evaluation_calls,
    }
    if stage not in limits or maximum_usd <= 0:
        raise ValueError("Unknown budget stage or nonpositive invocation estimate")
    amendment_path = path.parent / "operational-invocation-amendment.json"
    if "operational_amendment" in ledger and not amendment_path.exists():
        raise ValueError("Applied operational amendment is missing")
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        protocol = json.loads((path.parent / "protocol.json").read_text())
        if amendment["protocol_identity"] != protocol["identity"] or not amendment["reason"]:
            raise ValueError("Operational amendment does not identify this frozen protocol")
        for amended_stage, additional in amendment["additional_calls"].items():
            if amended_stage not in limits or type(additional) is not int or additional <= 0:
                raise ValueError("Invalid operational call amendment")
            limits[amended_stage] += additional
        if "operational_amendment" in ledger and ledger["operational_amendment"] != amendment:
            raise ValueError("Previously applied operational amendment changed")
        ledger["operational_amendment"] = amendment
    calls = sum(item["stage"] == stage for item in ledger["invocations"])
    if calls >= limits[stage]:
        raise ValueError(f"Invocation allocation exhausted for {stage}")
    allocated = initial_pilots_usd + sum(
        item["maximum_estimated_usd"] for item in ledger["invocations"]
    )
    if allocated + maximum_usd > training.project_compute_limit_usd:
        raise ValueError("Invocation would exceed the project's compute allocation")
    if (
        ledger["starting_verified_credits_usd"] - allocated - maximum_usd
        < training.credits_reserved_usd
    ):
        raise ValueError("Invocation would consume reserved credits")
    ledger["invocations"].append(
        {"stage": stage, "maximum_estimated_usd": maximum_usd, "status": "reserved"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    os.replace(temporary, path)
    return len(ledger["invocations"]) - 1


def finish_invocation(path: Path, index: int, outcome: dict):
    ledger = json.loads(path.read_text())
    ledger["invocations"][index] |= outcome
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    os.replace(temporary, path)
