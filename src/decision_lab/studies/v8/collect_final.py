import json
from datetime import UTC, datetime

import modal

from decision_lab.publication import atomic_json
from decision_lab.studies.v8.protocol import ROOT


def main():
    job = json.loads((ROOT / "final-evaluation-job.json").read_text())
    call = modal.FunctionCall.from_id(job["call_id"])
    try:
        result = call.get(timeout=10)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": job["call_id"]}), flush=True)
        return
    root = ROOT / "r2"
    for name, value in (
        ("selection-freeze.json", result["selection"]),
        ("evaluation.json", result["evaluation"]),
        ("artifact-verification.json", result["verification"]),
    ):
        path = root / name
        if path.exists():
            raise FileExistsError("Inspect existing collected output before re-collection")
        atomic_json(path, value)
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    record = next(r for r in ledger if r.get("function_call_id") == job["call_id"])
    if record["status"] != "submitted":
        raise ValueError("Function-call ledger state differs")
    record.update(status="complete", collected_at_utc=datetime.now(UTC).isoformat())
    atomic_json(ROOT / "allocation-ledger.json", ledger)
    atomic_json(
        ROOT / "final-evaluation-collected.json",
        dict(
            call_id=job["call_id"],
            selection_identity=result["selection"]["identity"],
            verified=True,
            collected_at_utc=record["collected_at_utc"],
            new_gpu_calls=0,
        ),
    )
    print(json.dumps({"status": "complete", "call_id": job["call_id"]}), flush=True)


if __name__ == "__main__":
    main()
