import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from decision_lab.studies.v8.protocol import ROOT, settings, verify, verify_study

spec = settings("r2")
app = modal.App("decision-model-lab-v8-auth-recovery")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud", "analysis"], extra_options="--no-dev")
    .add_local_python_source(
        "decision_lab",
        "decision_lab.studies.v4",
        "decision_lab.studies.v5",
        "decision_lab.studies.v6",
        "decision_lab.studies.v7",
        "decision_lab.studies.v8",
    )
    .add_local_dir("configs", "/root/configs")
    .add_local_file("LICENSE", "/root/LICENSE")
    .add_local_file("NOTICE", "/root/NOTICE")
    .add_local_file("uv.lock", "/root/uv.lock")
    .add_local_file("docs/V8_PROTOCOL.md", "/root/docs/V8_PROTOCOL.md")
)
volume = modal.Volume.from_name(spec["volume"], create_if_missing=False)


@app.function(
    image=image,
    gpu=spec["gpu"],
    cpu=(4, 4),
    memory=(32768, 32768),
    startup_timeout=180,
    scaledown_window=2,
    max_containers=1,
    retries=0,
    timeout=1050,
    volumes={"/runs": volume},
    env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
)
def continue_exactly():
    from decision_lab.studies.v8.study import train_chunk
    from decision_lab.training.checkpoint import verify_checkpoint

    root = Path("/runs/study-v8/r2")
    protocol = verify(root)
    state = json.loads((root / "state-treatment.json").read_text())
    if state["step"] != 756 or state["complete"]:
        raise ValueError("Authentication recovery must start at the committed update 756")
    checkpoint = root / state["resume"]
    verify_checkpoint(checkpoint, protocol["identity"])
    prior = json.loads((checkpoint / "progress.json").read_text())["updates"]
    result = train_chunk(root, "treatment", volume.commit)
    final = json.loads((root / "trajectory-treatment.json").read_text())
    if not result["complete"] or result["step"] != 1134 or final["updates"][:756] != prior:
        raise ValueError("Exact remaining exposure or trajectory prefix differs")
    receipt = dict(
        start_step=756,
        end_step=result["step"],
        presentations=result["presentations"],
        verified_checkpoint_hashes=True,
        optimizer_scheduler_rng_restored=True,
        prior_trajectory_prefix_unchanged=True,
        protocol_identity=protocol["identity"],
        originally_failed_call="treatment-3",
        no_completed_updates_repeated=True,
    )
    atomic_json(root / "jwt-recovery.json", receipt)
    volume.commit()
    return result | {"recovery": receipt, "protocol_identity": protocol["identity"]}


@app.local_entrypoint()
def main():
    from decision_lab.studies.v8.permit import permit, reserve

    verify_study()
    amendment = permit()
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    ceiling = reserve(ledger, "r2", "auth-recovery", "jwt-recovery")
    output = subprocess.check_output(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "billing",
            "summary",
            "--profile",
            "amandogra2016",
            "--json",
        ],
        text=True,
    )
    summary = json.loads(output)
    required = amendment["maximum_compute_usd"] - sum(r["allocation_usd"] for r in ledger)
    required += amendment["storage_reserve_usd"]
    if float(summary["billed_cost"]) != 0 or 30 - float(summary["metered_cost"]) < required:
        raise ValueError("Original budget no longer supports exact recovery")
    volume.hydrate()
    ownership = json.loads((ROOT / "modal-ownership.json").read_text())
    if ownership["volume_id"] != volume.object_id:
        raise ValueError("Owned Volume identity changed")
    ownership["app_ids"].append(app.app_id)
    atomic_json(ROOT / "modal-ownership.json", ownership)
    record = dict(
        repetition="r2",
        name="jwt-recovery",
        stage="auth-recovery",
        allocation_usd=ceiling,
        app_id=app.app_id,
        status="submitted",
        protocol_identity=verify(ROOT / "r2")["identity"],
        amendment_identity=amendment["identity"],
        submitted_at_utc=datetime.now(UTC).isoformat(),
    )
    ledger.append(record)
    atomic_json(ROOT / "allocation-ledger.json", ledger)
    start = time.monotonic()
    try:
        result = continue_exactly.remote()
    except BaseException:
        record.update(status="failed", client_seconds=time.monotonic() - start)
        atomic_json(ROOT / "allocation-ledger.json", ledger)
        raise
    record.update(status="complete", client_seconds=time.monotonic() - start)
    atomic_json(ROOT / "allocation-ledger.json", ledger)
    atomic_json(ROOT / "r2/training-call-treatment-3.json", result)
    atomic_json(ROOT / "r2/jwt-recovery.json", result["recovery"])
    print(json.dumps(result), flush=True)
