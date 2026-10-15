import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from decision_lab.studies.v8.permit import permit, reserve
from decision_lab.studies.v8.protocol import ROOT, settings, verify, verify_study

spec = settings("r1")
app = modal.App(spec["app_name"])
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
        "decision_lab.studies.v8.permit",
    )
    .add_local_dir("configs", "/root/configs")
    .add_local_file("LICENSE", "/root/LICENSE")
    .add_local_file("NOTICE", "/root/NOTICE")
    .add_local_file("uv.lock", "/root/uv.lock")
    .add_local_file("docs/V8_PROTOCOL.md", "/root/docs/V8_PROTOCOL.md")
)
volume = modal.Volume.from_name(spec["volume"], create_if_missing=True)
resources = {
    "image": image,
    "gpu": spec["gpu"],
    "cpu": (spec["cpu_cores"], spec["cpu_cores"]),
    "memory": (spec["memory_mib"], spec["memory_mib"]),
    "startup_timeout": spec["startup_timeout_seconds"],
    "scaledown_window": spec["scaledown_seconds"],
    "max_containers": 1,
    "retries": 0,
    "volumes": {"/runs": volume},
    "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
}


@app.function(**resources, timeout=spec["pilot_timeout_seconds"])
def gpu_pilot(repetition):
    from decision_lab.studies.v8.study import pilot

    return pilot(Path("/runs/study-v8") / repetition, volume.commit)


@app.function(**resources, timeout=spec["training_timeout_seconds"])
def gpu_training(repetition, arm):
    from decision_lab.studies.v8.study import train_chunk

    return train_chunk(Path("/runs/study-v8") / repetition, arm, volume.commit)


@app.function(**resources, timeout=spec["selection_timeout_seconds"])
def gpu_selection(repetition):
    from decision_lab.studies.v8.study import select

    return select(Path("/runs/study-v8") / repetition, volume.commit)


@app.function(**resources, timeout=spec["retry_timeout_seconds"])
def gpu_recovery(repetition, arm, failed_call):
    from decision_lab.studies.v8.study import train_chunk
    from decision_lab.training.checkpoint import verify_checkpoint

    root = Path("/runs/study-v8") / repetition
    protocol = verify(root)
    state = json.loads((root / f"state-{arm}.json").read_text())
    checkpoint = root / state["resume"]
    verify_checkpoint(checkpoint, protocol["identity"])
    if state["complete"] or state["step"] <= 0:
        raise ValueError("Recovery requires an incomplete saved training state")
    result = train_chunk(root, arm, volume.commit)
    receipt = {
        "failed_call": failed_call,
        "start_step": state["step"],
        "resumed_through_step": result["step"],
        "checkpoint_hashes_verified": True,
        "optimizer_and_rng_restored": True,
        "protocol_identity": protocol["identity"],
    }
    atomic_json(root / "checkpoint-recovery.json", receipt)
    volume.commit()
    return result | {"recovered_from_call": failed_call, "protocol_identity": protocol["identity"]}


@app.function(**resources, timeout=spec["test_timeout_seconds"])
def gpu_test(repetition):
    from decision_lab.studies.v8.study import evaluate

    return evaluate(Path("/runs/study-v8") / repetition, volume.commit, Path("/root/configs"))


def invoke(repetition, stage, name, function, *args):
    protocol = verify(ROOT / repetition)
    path = ROOT / "allocation-ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    ceiling = reserve(ledger, repetition, stage, name)
    record = {
        "repetition": repetition,
        "name": name,
        "stage": stage,
        "allocation_usd": ceiling,
        "app_id": app.app_id,
        "protocol_identity": protocol["identity"],
        "status": "submitted",
        "submitted_at_utc": datetime.now(UTC).isoformat(),
    }
    ledger.append(record)
    atomic_json(path, ledger)
    started = time.monotonic()
    try:
        result = function.remote(repetition, *args)
    except BaseException:
        record |= {"status": "failed", "client_seconds": time.monotonic() - started}
        atomic_json(path, ledger)
        raise
    record |= {"status": "complete", "client_seconds": time.monotonic() - started}
    atomic_json(path, ledger)
    return result


@app.local_entrypoint()
def main(repetition: str, action: str, arm: str = "", failed_call: str = ""):
    if repetition not in ("r1", "r2") or action not in (
        "pilot",
        "pilot-recovery",
        "position-revalidation",
        "train",
        "selection",
        "test",
        "recovery",
    ):
        raise ValueError("Invalid repetition/action")
    verify_study()
    root = ROOT / repetition
    summary = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "billing",
            "summary",
            "--profile",
            spec["profile"],
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    current = json.loads(summary.stdout)
    master = verify_study()
    ledger_path = ROOT / "allocation-ledger.json"
    used = (
        sum(r["allocation_usd"] for r in json.loads(ledger_path.read_text()))
        if ledger_path.exists()
        else 0
    )
    if (ROOT / "auth-recovery-amendment.json").exists():
        operational = permit()
        required = operational["maximum_compute_usd"] - used + operational["storage_reserve_usd"]
    else:
        required = master["maximum_compute_usd"] - used + master["joint_storage_reserve_usd"]
    if float(current["billed_cost"]) != 0 or 30 - float(current["metered_cost"]) < required:
        raise ValueError("Billing no longer supports the original credit-only scope")
    atomic_json(root / f"billing-before-{action}.json", current)
    volume.hydrate()
    ownership_path = ROOT / "modal-ownership.json"
    ownership = (
        json.loads(ownership_path.read_text())
        if ownership_path.exists()
        else {
            "volume": spec["volume"],
            "volume_id": volume.object_id,
            "app_ids": [],
            "original_resource_ids": json.loads((ROOT / "position-correction.json").read_text())[
                "original_resource_ids"
            ],
        }
    )
    if ownership["volume_id"] != volume.object_id:
        raise ValueError("Owned Volume changed")
    ownership["app_ids"].append(app.app_id)
    atomic_json(ownership_path, ownership)
    if action == "pilot":
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(root / "parent", f"/study-v8/{repetition}/parent")
            upload.put_directory(root / "data", f"/study-v8/{repetition}/data")
            upload.put_file(root / "protocol.json", f"/study-v8/{repetition}/protocol.json")
        result = invoke(repetition, "pilot", "pilot", gpu_pilot)
        atomic_json(root / "pilot.json", result)
    elif action == "position-revalidation":
        legacy = json.loads((ROOT / "allocation-ledger.json").read_text())
        matching = [r for r in legacy if r["repetition"] == repetition and r["stage"] == "pilot"]
        if (
            len(matching) != 1
            or matching[0]["status"] != "complete"
            or matching[0]["protocol_identity"] == verify(root)["identity"]
        ):
            raise ValueError("Revalidation requires the preserved original completed pilot")
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(root / "parent", f"/study-v8/{repetition}/parent")
            upload.put_directory(root / "data", f"/study-v8/{repetition}/data")
            upload.put_file(root / "protocol.json", f"/study-v8/{repetition}/protocol.json")
        result = invoke(repetition, "retry", "position-revalidation", gpu_pilot)
        atomic_json(root / "pilot.json", result)
    elif action == "pilot-recovery":
        ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
        failures = [r for r in ledger if r["repetition"] == repetition and r["status"] == "failed"]
        if len(failures) != 1 or failures[0]["name"] != "pilot" or (root / "pilot.json").exists():
            raise ValueError("Pilot recovery must match the one recorded pre-scoring failure")
        result = invoke(repetition, "retry", "pilot-recovery", gpu_pilot)
        atomic_json(root / "pilot.json", result)
    elif action == "train":
        pilot = json.loads((root / "pilot.json").read_text())
        projected = (
            max(v["seconds_per_update"] for v in pilot["timings"].values())
            * spec["chunk_steps"]
            * 1.15
            + 100
        )
        if (
            pilot["protocol_identity"] != verify(root)["identity"]
            or projected >= spec["training_timeout_seconds"] - 30
        ):
            raise ValueError("Pilot does not support the frozen scope")
        for arm in ("control", "treatment"):
            for chunk in range(1, 4):
                path = root / f"training-call-{arm}-{chunk}.json"
                if path.exists():
                    result = json.loads(path.read_text())
                    if result["protocol_identity"] != verify(root)["identity"]:
                        raise ValueError("Previous receipt belongs to another protocol")
                else:
                    result = invoke(repetition, "training", f"{arm}-{chunk}", gpu_training, arm)
                    result["protocol_identity"] = verify(root)["identity"]
                    atomic_json(path, result)
                print(json.dumps(result), flush=True)
                if result["complete"]:
                    break
            if not result["complete"]:
                raise RuntimeError("Frozen exposure needs recorded checkpoint recovery")
    elif action == "selection":
        result = invoke(repetition, "selection", "selection", gpu_selection)
        atomic_json(root / "selection-freeze.json", result)
    elif action == "recovery":
        ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
        failed = [r for r in ledger if r["repetition"] == repetition and r["status"] == "failed"]
        if (
            len(failed) != 1
            or failed[0]["name"] != failed_call
            or failed[0]["stage"] != "training"
            or arm not in ("control", "treatment")
            or not failed_call.startswith(arm + "-")
        ):
            raise ValueError("Recovery must identify the one failed training call")
        state = json.loads(b"".join(volume.read_file(f"/study-v8/{repetition}/state-{arm}.json")))
        if (
            state["protocol_identity"] != verify(root)["identity"]
            or state["complete"]
            or state["step"] <= 0
        ):
            raise ValueError("Inspect a valid saved checkpoint before requesting recovery")
        atomic_json(root / "recovery-inspection.json", state)
        result = invoke(
            repetition, "retry", f"recovery-{failed_call}", gpu_recovery, arm, failed_call
        )
        atomic_json(root / f"training-call-{failed_call}.json", result)
    else:
        result = invoke(repetition, "test", "test", gpu_test)
        atomic_json(root / "evaluation.json", result["evaluation"])
        atomic_json(root / "artifact-verification.json", result["verification"])
    print(json.dumps({"repetition": repetition, "action": action, "completed": True}), flush=True)


if __name__ == "__main__":
    raise RuntimeError("Use modal run -m decision_lab.studies.v8.app")
