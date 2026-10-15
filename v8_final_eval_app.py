import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from v8_lab.protocol import ROOT, settings, verify, verify_study

spec = settings("r2")
app = modal.App("decision-model-lab-v8-final-evaluation")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud", "analysis"], extra_options="--no-dev")
    .add_local_python_source(
        "decision_lab", "v4_lab", "v5_lab", "v6_lab", "v7_lab", "v8_lab", "v8_selection_completion"
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
    timeout=900,
    volumes={"/runs": volume},
    env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
)
def finish_and_evaluate(cache):
    from v8_lab.study import evaluate
    from v8_selection_completion import finish_selection

    root = Path("/runs/study-v8/r2")
    frozen = finish_selection(root, cache, volume.commit)
    result = evaluate(root, volume.commit, Path("/root/configs"))
    return result | {"selection": frozen}


@app.local_entrypoint()
def main():
    from decision_lab.training.checkpoint import file_hash
    from decision_lab.v3.protocol import check_seal
    from v8_permit import permit, reserve

    verify_study()
    operational = permit()
    continuation = check_seal(json.loads((ROOT / "selection-continuation-permit.json").read_text()))
    cache = json.loads((ROOT / "selection-cache-inventory.json").read_text())
    if continuation["cache_sha256"] != file_hash(ROOT / "selection-cache-inventory.json"):
        raise ValueError("Validated cache inventory changed")
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    ceiling = reserve(ledger, "r2", "test", "test")
    summary = json.loads(
        subprocess.check_output(
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
    )
    required = operational["maximum_compute_usd"] - sum(r["allocation_usd"] for r in ledger)
    if float(summary["billed_cost"]) != 0 or 30 - float(summary["metered_cost"]) < required + 0.5:
        raise ValueError("Original budget no longer supports the planned test")
    volume.hydrate()
    ownership = json.loads((ROOT / "modal-ownership.json").read_text())
    if volume.object_id != ownership["volume_id"]:
        raise ValueError("Owned Volume identity differs")
    ownership["app_ids"].append(app.app_id)
    atomic_json(ROOT / "modal-ownership.json", ownership)
    record = dict(
        repetition="r2",
        name="test",
        stage="test",
        allocation_usd=ceiling,
        app_id=app.app_id,
        protocol_identity=verify(ROOT / "r2")["identity"],
        status="submitted",
        continuation_identity=continuation["identity"],
        submitted_at_utc=datetime.now(UTC).isoformat(),
    )
    ledger.append(record)
    atomic_json(ROOT / "allocation-ledger.json", ledger)
    started = time.monotonic()
    try:
        call = finish_and_evaluate.spawn(cache)
    except BaseException:
        record.update(status="failed", client_seconds=time.monotonic() - started)
        atomic_json(ROOT / "allocation-ledger.json", ledger)
        raise
    record["function_call_id"] = call.object_id
    atomic_json(ROOT / "allocation-ledger.json", ledger)
    atomic_json(
        ROOT / "final-evaluation-job.json",
        dict(
            call_id=call.object_id,
            app_id=app.app_id,
            continuation_identity=continuation["identity"],
            submitted_at_utc=record["submitted_at_utc"],
        ),
    )
    print(
        json.dumps({"spawned_call_id": call.object_id, "gpu_calls_added_beyond_reservation": 0}),
        flush=True,
    )
