import json
import re
import time
from pathlib import Path

import modal

from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.model_config import load_pilot_config
from decision_lab.training.config import load_training_config
from decision_lab.training.ledger import finish_invocation, reserve_invocation
from decision_lab.training.protocol import verify_protocol

config_directory = Path.cwd() / "configs" if modal.is_local() else Path("/root/configs")
pilot = load_pilot_config(config_directory / "pilot.yaml")
training = load_training_config(config_directory / "training.yaml")
evaluation = load_evaluation_config(config_directory / "evaluation.yaml")
app = modal.App("decision-model-lab-evaluation")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud"], extra_options="--no-dev")
    .add_local_python_source("decision_lab")
    .add_local_dir("configs", "/root/configs")
    .add_local_file("LICENSE", "/root/LICENSE")
    .add_local_file("NOTICE", "/root/NOTICE")
)
volume = modal.Volume.from_name(pilot.volume_name)
resources = {
    "image": image,
    "gpu": pilot.gpu,
    "cpu": (pilot.cpu_cores, pilot.cpu_cores),
    "memory": (pilot.memory_mib, pilot.memory_mib),
    "startup_timeout": pilot.startup_timeout_seconds,
    "scaledown_window": pilot.scaledown_seconds,
    "max_containers": 1,
    "retries": 0,
    "volumes": {"/runs": volume},
    "env": {"HF_HOME": "/runs/hf-cache", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
}


@app.function(**resources, timeout=evaluation.evaluation_timeout_seconds)
def calibrate(study, sent_at):
    ready_at = time.time()
    started = time.perf_counter()
    from decision_lab.evaluation.study import calibrate_all

    result = calibrate_all(Path("/runs") / study)
    volume.commit()
    return {
        "result": result,
        "timing": {
            "dispatch_to_worker_body_seconds": ready_at - sent_at,
            "worker_body_including_volume_commit_seconds": time.perf_counter() - started,
        },
    }


@app.function(**resources, timeout=evaluation.evaluation_timeout_seconds)
def evaluate(study, sent_at):
    ready_at = time.time()
    started = time.perf_counter()
    from decision_lab.evaluation.study import evaluate_all

    result = evaluate_all(Path("/runs") / study)
    volume.commit()
    return {
        "result": result,
        "timing": {
            "dispatch_to_worker_body_seconds": ready_at - sent_at,
            "worker_body_including_volume_commit_seconds": time.perf_counter() - started,
        },
    }


@app.function(**resources, timeout=evaluation.evaluation_timeout_seconds)
def verify_artifacts(study, sent_at):
    ready_at = time.time()
    started = time.perf_counter()
    import torch

    from decision_lab.release import verify_release

    torch.set_num_threads(pilot.cpu_threads)
    result = {
        name: verify_release(Path("/runs") / study / "releases" / name, "cuda")
        for name in ("token", "head", "scratch")
    }
    volume.commit()
    return {
        "result": result,
        "timing": {
            "dispatch_to_worker_body_seconds": ready_at - sent_at,
            "worker_body_including_volume_commit_seconds": time.perf_counter() - started,
        },
    }


@app.local_entrypoint()
def main(action: str = "all", study: str = "study-v1"):
    if re.fullmatch(r"study-[a-z0-9-]+", study) is None:
        raise ValueError("Study name must be a safe study-prefixed identifier")
    local = Path("runs") / study
    protocol = json.loads((local / "protocol.json").read_text())
    verify_protocol(protocol)
    if action not in {"all", "calibrate", "evaluate", "verify"}:
        raise ValueError("Invalid evaluation action")
    selected = {}
    for name in ("token", "head", "scratch"):
        state = json.loads((local / f"{name}-state.json").read_text())
        if not state["complete"] or state["protocol_identity"] != protocol["identity"]:
            raise ValueError("All three models must complete the same frozen training protocol")
        selected[name] = state
    selection = local / "selection.json"
    selection.write_text(json.dumps(selected, indent=2) + "\n")
    with volume.batch_upload() as batch:
        batch.put_file(selection, f"/{study}/selection.json")
        batch.put_file(local / "baselines/freeze.json", f"/{study}/baseline-freeze.json")
        amendment = local / "operational-invocation-amendment.json"
        if amendment.exists():
            batch.put_file(amendment, f"/{study}/operational-invocation-amendment.json")
    operations = [
        ("calibrate", calibrate, "calibration-freeze.json"),
        ("evaluate", evaluate, "evaluation.json"),
        ("verify", verify_artifacts, "artifact-verification.json"),
    ]
    for phase, function, filename in operations:
        if action not in {"all", phase}:
            continue
        output = local / filename
        if output.exists():
            print(json.dumps({"already_complete": phase, "report": str(output)}), flush=True)
            continue
        timeout = protocol["evaluation"]["evaluation_timeout_seconds"]
        allocation = (
            (timeout + pilot.startup_timeout_seconds + pilot.scaledown_seconds)
            / 3600
            * pilot.hourly_usd
        )
        index = reserve_invocation(local / "ledger.json", "evaluation", allocation, training, 1.0)
        try:
            started = time.perf_counter()
            response = function.remote(study, time.time())
            result = response["result"]
            timing = response["timing"] | {"client_wall_seconds": time.perf_counter() - started}
        except BaseException:
            finish_invocation(local / "ledger.json", index, {"status": "failed", "phase": phase})
            raise
        output.write_text(json.dumps(result, indent=2) + "\n")
        (local / f"{phase}-timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        finish_invocation(local / "ledger.json", index, {"status": "complete", "phase": phase})
        print(json.dumps({"completed": phase, "report": str(output)}), flush=True)
    if (local / "evaluation.json").exists():
        result = json.loads((local / "evaluation.json").read_text())
        for name in result["models"]:
            for split in [
                "worked_development",
                "test_id",
                "test_depth",
                "test_wording",
                "test_structure",
                *["robust_" + item for item in result["robustness_manifest"]],
            ]:
                output = local / "predictions" / name / f"{split}.jsonl"
                if output.exists():
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as stream:
                    for chunk in volume.read_file(f"/{study}/predictions/{name}/{split}.jsonl"):
                        stream.write(chunk)
