import json
import time
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from decision_lab.v3.protocol import settings, verify

config_directory = Path(__file__).parent / "configs" if modal.is_local() else Path("/root/configs")
spec = settings(config_directory)
app = modal.App(spec["app_name"])
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud"], extra_options="--no-dev")
    .add_local_python_source("decision_lab")
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


@app.function(**resources, timeout=spec["selection_timeout_seconds"])
def gpu_selection():
    from decision_lab.v3.study import selection

    return selection(Path("/runs/study-v3"), config_directory, volume.commit)


@app.function(**resources, timeout=spec["test_timeout_seconds"])
def gpu_test():
    from decision_lab.v3.study import evaluate

    return evaluate(Path("/runs/study-v3"), config_directory, volume.commit)


def invoke(root, stage, function):
    protocol = verify(root)
    path = root / "allocation-ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    if any(record["stage"] == stage for record in ledger):
        raise ValueError("This stage was already submitted; recovery requires a reviewed ledger")
    allocation = protocol["budget"]["call_ceiling_usd"][stage]
    if (
        sum(record["allocation_usd"] for record in ledger) + allocation
        > spec["compute_allocation_usd"]
    ):
        raise ValueError("Submission exceeds the compute allocation")
    record = {
        "stage": stage,
        "app_id": app.app_id,
        "allocation_usd": allocation,
        "status": "submitted",
    }
    ledger.append(record)
    atomic_json(path, ledger)
    started = time.perf_counter()
    try:
        result = function.remote()
    except BaseException:
        record |= {"status": "failed", "client_seconds": time.perf_counter() - started}
        atomic_json(path, ledger)
        raise
    record |= {"status": "complete", "client_seconds": time.perf_counter() - started}
    atomic_json(path, ledger)
    return result


@app.local_entrypoint()
def main(action: str = "all"):
    if action not in {"selection", "test", "all"}:
        raise ValueError("Action must be selection, test, or all")
    root = Path("runs/study-v3")
    protocol = verify(root)
    if action in {"selection", "all"}:
        volume.hydrate()
        atomic_json(
            root / "modal-ownership.json",
            {"app_id": app.app_id, "volume": spec["volume"], "volume_id": volume.object_id},
        )
        with volume.batch_upload(force=False) as upload:
            for directory in ("data", "candidates"):
                upload.put_directory(root / directory, f"/study-v3/{directory}")
            upload.put_file(root / "protocol.json", "/study-v3/protocol.json")
        selected = invoke(root, "selection", gpu_selection)
        atomic_json(root / "selection-freeze.json", selected)
        print(
            json.dumps(
                {"choices": selected["choices"], "selection_identity": selected["identity"]}
            ),
            flush=True,
        )
    if action in {"test", "all"}:
        result = invoke(root, "test", gpu_test)
        atomic_json(root / "evaluation.json", result["evaluation"])
        atomic_json(root / "artifact-verification.json", result["verification"])
    if protocol["identity"] != verify(root)["identity"]:
        raise ValueError("Protocol changed during execution")
