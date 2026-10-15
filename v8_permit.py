import json
from pathlib import Path

from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal
from v8_lab.protocol import ROOT, reservation, verify_study


def permit():
    path = ROOT / "auth-recovery-amendment.json"
    record = check_seal(json.loads(path.read_text()))
    if record["study_identity"] != verify_study()["identity"]:
        raise ValueError("Operational amendment belongs to another study")
    if file_hash(ROOT / "jwt-failure-inspection.json") != record["inspection_sha256"]:
        raise ValueError("Inspected recovery evidence changed")
    if file_hash(Path("docs/V8_AUTH_RECOVERY_AMENDMENT.md")) != record["document_sha256"]:
        raise ValueError("Operational amendment document changed")
    return record


def reserve(ledger, repetition, stage, name):
    path = ROOT / "auth-recovery-amendment.json"
    if not path.exists():
        return reservation(ledger, repetition, stage, name)
    record = permit()
    if stage == "auth-recovery":
        if repetition != "r2" or name != "jwt-recovery" or any(r["stage"] == stage for r in ledger):
            raise ValueError("Only the single r2 authentication recovery is permitted")
        ceiling = record["additional_call_ceiling_usd"]
    else:
        original = [r for r in ledger if r["stage"] != "auth-recovery"]
        ceiling = reservation(original, repetition, stage, name)
    if sum(r["allocation_usd"] for r in ledger) + ceiling > record["compute_allocation_usd"]:
        raise ValueError("Amended joint allocation exceeded, including failed reservations")
    own = sum(r["allocation_usd"] for r in ledger if r["repetition"] == repetition)
    if own + ceiling > record["per_repetition_allocation_usd"][repetition]:
        raise ValueError("Amended repetition allocation exceeded")
    return ceiling
