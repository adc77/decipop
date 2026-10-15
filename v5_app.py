import json
import time
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from v5_lab.protocol import budget, settings, verify

config_directory = Path(__file__).parent / "configs" if modal.is_local() else Path("/root/configs")
spec = settings(config_directory)
app = modal.App(spec["app_name"])
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud", "analysis"], extra_options="--no-dev")
    .add_local_python_source("decision_lab", "v5_lab")
    .add_local_dir("configs", "/root/configs")
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
    from v5_lab.study import run

    return run(Path("/runs/study-v5"), "pilot", volume.commit, config_directory)


@app.function(**resources, timeout=spec["diagnostic_timeout_seconds"])
def gpu_diagnostic():
    from v5_lab.study import run

    return run(Path("/runs/study-v5"), "diagnostic", volume.commit, config_directory)


@app.local_entrypoint()
def main(action: str):
    if action not in ("pilot", "diagnostic"):
        raise ValueError("Invalid action")
    root = Path("runs/study-v5")
    protocol = verify(root)
    if action == "diagnostic":
        pilot = json.loads((root / "pilot-execution.json").read_text())
        questions = protocol["audit"]["partitions"]["diagnostic"]["questions"]
        projected = (
            sum(
                questions / m["rows_per_second"] + m["load_seconds"]
                for m in pilot["models"].values()
            )
            * 1.5
            + 90
        )
        if projected >= spec["diagnostic_timeout_seconds"] - 30:
            raise ValueError("Pilot does not support frozen inference scope")
        atomic_json(
            root / "timing-preflight.json",
            {"projected_seconds": projected, "pilot_identity": pilot["protocol_identity"]},
        )
    ledger_path = root / "allocation-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if any(r["stage"] == action for r in ledger):
        raise ValueError("Phase was already submitted; record any recovery explicitly")
    allocation = budget(spec)["call_ceiling_usd"][action]
    if sum(r["allocation_usd"] for r in ledger) + allocation > spec["compute_allocation_usd"]:
        raise ValueError("Compute allocation exceeded")
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
    if action == "pilot":
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(root / "data", "/study-v5/data")
            upload.put_directory(root / "models", "/study-v5/models")
            upload.put_file(root / "protocol.json", "/study-v5/protocol.json")
    record = {
        "stage": action,
        "allocation_usd": allocation,
        "app_id": app.app_id,
        "status": "submitted",
        "protocol_identity": protocol["identity"],
    }
    ledger.append(record)
    atomic_json(ledger_path, ledger)
    start = time.perf_counter()
    try:
        result = (gpu_pilot if action == "pilot" else gpu_diagnostic).remote()
    except BaseException:
        record.update(status="failed", client_seconds=time.perf_counter() - start)
        atomic_json(ledger_path, ledger)
        raise
    record.update(status="complete", client_seconds=time.perf_counter() - start)
    atomic_json(ledger_path, ledger)
    atomic_json(root / f"{action}-execution.json", result)
