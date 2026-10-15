import json
import time
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from v6_lab.protocol import budget, settings, verify

config_directory = Path(__file__).parent / "configs" if modal.is_local() else Path("/root/configs")
spec = settings(config_directory)
app = modal.App(spec["app_name"])
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud", "analysis"], extra_options="--no-dev")
    .add_local_python_source("decision_lab", "v4_lab", "v5_lab", "v6_lab", "v6_recovery")
    .add_local_dir("configs", "/root/configs")
    .add_local_file("LICENSE", "/root/LICENSE")
    .add_local_file("NOTICE", "/root/NOTICE")
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
def gpu_pilot():
    from v6_lab.study import pilot

    return pilot(Path("/runs/study-v6"), volume.commit)


@app.function(**resources, timeout=spec["training_timeout_seconds"])
def gpu_training(arm):
    from v6_lab.study import train_chunk

    return train_chunk(Path("/runs/study-v6"), arm, volume.commit)


@app.function(**resources, timeout=spec["retry_timeout_seconds"])
def gpu_recovery():
    from v6_lab.study import train_chunk
    from v6_recovery import prepare_resume

    root = Path("/runs/study-v6")
    protocol = verify(root)
    recovery = prepare_resume(
        root, "control", protocol["parent_protocol_identity"], protocol["identity"], 250
    )
    volume.commit()
    result = train_chunk(root, "control", volume.commit)
    recovery |= {
        "optimizer_and_rng_restoration_pending": False,
        "resumed_through_step": result["step"],
        "original_call": "control-1",
    }
    atomic_json(root / "checkpoint-recovery.json", recovery)
    volume.commit()
    return result | {"protocol_identity": protocol["identity"], "recovered_from_call": "control-1"}


@app.function(**resources, timeout=spec["selection_timeout_seconds"])
def gpu_selection():
    from v6_lab.study import select

    return select(Path("/runs/study-v6"), volume.commit)


@app.function(**resources, timeout=spec["test_timeout_seconds"])
def gpu_test():
    from v6_lab.study import evaluate

    return evaluate(Path("/runs/study-v6"), volume.commit, config_directory)


def invoke(root, stage, name, function, *args):
    protocol = verify(root)
    path = root / "allocation-ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    if any(r["name"] == name for r in ledger):
        raise ValueError("Call name already submitted")
    ceiling = budget(spec)["call_ceiling_usd"][stage]
    if sum(r["allocation_usd"] for r in ledger) + ceiling > spec["compute_allocation_usd"]:
        raise ValueError("Submission would exceed allocation")
    record = {
        "name": name,
        "stage": stage,
        "allocation_usd": ceiling,
        "app_id": app.app_id,
        "status": "submitted",
        "protocol_identity": protocol["identity"],
    }
    ledger.append(record)
    atomic_json(path, ledger)
    started = time.perf_counter()
    try:
        result = function.remote(*args)
    except BaseException:
        record |= {"status": "failed", "client_seconds": time.perf_counter() - started}
        atomic_json(path, ledger)
        raise
    record |= {"status": "complete", "client_seconds": time.perf_counter() - started}
    atomic_json(path, ledger)
    return result


@app.local_entrypoint()
def main(action: str):
    if action not in ("pilot", "train", "selection", "test", "recovery"):
        raise ValueError("Invalid action")
    root = Path("runs/study-v6")
    verify(root)
    if action == "train":
        pilot = json.loads((root / "pilot.json").read_text())
        projections = {
            arm: record["seconds_per_update"] * spec["chunk_steps"] * 1.15 + 100
            for arm, record in pilot["timings"].items()
        }
        if max(projections.values()) >= spec["training_timeout_seconds"] - 30:
            raise ValueError("Pilot does not support frozen training scope")
        atomic_json(
            root / "timing-preflight.json",
            {
                "pilot_identity": pilot["protocol_identity"],
                "projected_chunk_seconds": projections,
                "margin": 1.15,
                "overhead_seconds": 100,
            },
        )
    volume.hydrate()
    ownership_path = root / "modal-ownership.json"
    ownership = (
        json.loads(ownership_path.read_text())
        if ownership_path.exists()
        else {"volume": spec["volume"], "volume_id": volume.object_id, "app_ids": []}
    )
    if ownership["volume_id"] != volume.object_id:
        raise ValueError("Owned Volume identity differs")
    ownership["app_ids"].append(app.app_id)
    atomic_json(ownership_path, ownership)
    if (root / "amended-protocol.json").exists():
        with volume.batch_upload(force=True) as upload:
            upload.put_file(root / "amended-protocol.json", "/study-v6/amended-protocol.json")
    if action == "pilot":
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(root / "data", "/study-v6/data")
            upload.put_directory(root / "parent", "/study-v6/parent")
            upload.put_file(root / "protocol.json", "/study-v6/protocol.json")
        atomic_json(root / "pilot.json", invoke(root, "pilot", "pilot", gpu_pilot))
    elif action == "recovery":
        ledger = json.loads((root / "allocation-ledger.json").read_text())
        failures = [r for r in ledger if r["status"] == "failed"]
        if len(failures) != 1 or failures[0]["name"] != "control-1":
            raise ValueError("Recovery is restricted to the recorded packaging failure")
        result = invoke(root, "retry", "recovery-control", gpu_recovery)
        atomic_json(root / "training-call-control-1.json", result)
        print(json.dumps(result), flush=True)
    elif action == "train":
        for arm in ("control", "treatment"):
            for chunk in range(spec["max_training_calls_per_arm"]):
                receipt_path = root / f"training-call-{arm}-{chunk + 1}.json"
                if receipt_path.exists():
                    result = json.loads(receipt_path.read_text())
                    if result["protocol_identity"] != verify(root)["identity"]:
                        raise ValueError("Prior training call belongs to another protocol")
                    if result["complete"]:
                        break
                    continue
                result = invoke(root, "training", f"{arm}-{chunk + 1}", gpu_training, arm)
                result["protocol_identity"] = verify(root)["identity"]
                atomic_json(root / f"training-call-{arm}-{chunk + 1}.json", result)
                print(json.dumps(result), flush=True)
                if result["complete"]:
                    break
            if not result["complete"]:
                raise RuntimeError(
                    "Arm needs recovery; preserve ledger and review before submission"
                )
    elif action == "selection":
        result = invoke(root, "selection", "selection", gpu_selection)
        atomic_json(root / "selection-freeze.json", result)
        print(json.dumps({"choices": result["choices"]}), flush=True)
    else:
        result = invoke(root, "test", "test", gpu_test)
        for name, record in (
            ("evaluation", result["evaluation"]),
            ("artifact-verification", result["verification"]),
        ):
            atomic_json(root / f"{name}.json", record)
