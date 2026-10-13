import json
import time
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from v4_lab.protocol import budget, settings, verify

config_directory = Path(__file__).parent / "configs" if modal.is_local() else Path("/root/configs")
spec = settings(config_directory)
app = modal.App(spec["app_name"])
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud"], extra_options="--no-dev")
    .add_local_python_source("decision_lab", "v4_lab")
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


@app.function(**resources, timeout=spec["diagnostic_timeout_seconds"])
def gpu_diagnostic():
    from v4_lab.study import diagnose

    return diagnose(Path("/runs/study-v4"), volume.commit)


@app.function(**resources, timeout=spec["pilot_timeout_seconds"])
def gpu_pilot():
    from v4_lab.study import pilot

    return pilot(Path("/runs/study-v4"), volume.commit)


@app.function(**resources, timeout=spec["training_timeout_seconds"])
def gpu_training(arm):
    from v4_lab.study import train_chunk

    return train_chunk(Path("/runs/study-v4"), arm, volume.commit)


@app.function(**resources, timeout=spec["selection_timeout_seconds"])
def gpu_selection():
    from v4_lab.study import select

    return select(Path("/runs/study-v4"), volume.commit)


@app.function(**resources, timeout=spec["test_timeout_seconds"])
def gpu_test():
    from v4_lab.study import evaluate

    return evaluate(Path("/runs/study-v4"), volume.commit, config_directory)


def invoke(root, stage, name, function, *args):
    phase = "diagnostic" if stage in ("diagnostic", "pilot") else "training"
    protocol = verify(root, phase)
    path = root / "allocation-ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    if any(r["name"] == name for r in ledger):
        raise ValueError("Call name was already submitted")
    allocation = budget(protocol["settings"])["call_ceiling_usd"][stage]
    if sum(r["allocation_usd"] for r in ledger) + allocation > spec["compute_allocation_usd"]:
        raise ValueError("Submission would exceed the compute allocation")
    record = {
        "name": name,
        "stage": stage,
        "allocation_usd": allocation,
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
    if action not in ("diagnostic", "pilot", "train", "selection", "test"):
        raise ValueError("Invalid V4 action")
    root = Path("runs/study-v4")
    volume.hydrate()
    ownership_path = root / "modal-ownership.json"
    ownership = (
        json.loads(ownership_path.read_text())
        if ownership_path.exists()
        else {"volume": spec["volume"], "volume_id": volume.object_id, "app_ids": []}
    )
    if ownership["volume_id"] != volume.object_id:
        raise ValueError("Owned Volume identity changed")
    ownership["app_ids"].append(app.app_id)
    atomic_json(ownership_path, ownership)
    if action in ("diagnostic", "train"):
        phase = "diagnostic" if action == "diagnostic" else "training"
        verify(root, phase)
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(root / phase, f"/study-v4/{phase}")
            if action == "diagnostic":
                upload.put_directory(root / "parent", "/study-v4/parent")
            upload.put_file(root / f"{phase}-protocol.json", f"/study-v4/{phase}-protocol.json")
    if action == "diagnostic":
        atomic_json(
            root / "diagnostics.json", invoke(root, "diagnostic", "diagnostic", gpu_diagnostic)
        )
    elif action == "pilot":
        atomic_json(root / "pilot.json", invoke(root, "pilot", "pilot", gpu_pilot))
    elif action == "train":
        for arm in ("control", "treatment"):
            for chunk in range(spec["max_training_calls_per_arm"]):
                result = invoke(root, "training", f"{arm}-{chunk + 1}", gpu_training, arm)
                atomic_json(root / f"training-call-{arm}-{chunk + 1}.json", result)
                print(json.dumps(result), flush=True)
                if result["complete"]:
                    break
            if not result["complete"]:
                raise RuntimeError(
                    "Arm needs the reserved recovery call; stop to record the reason"
                )
    elif action == "selection":
        result = invoke(root, "selection", "selection", gpu_selection)
        atomic_json(root / "selection-freeze.json", result)
        print(json.dumps({"choices": result["choices"]}), flush=True)
    elif action == "test":
        result = invoke(root, "test", "test", gpu_test)
        atomic_json(root / "evaluation.json", result["evaluation"])
        atomic_json(root / "artifact-verification.json", result["verification"])
