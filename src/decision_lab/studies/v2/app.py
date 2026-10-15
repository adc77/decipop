import json
import time
import uuid
from pathlib import Path

import modal

from decision_lab.publication import atomic_json
from decision_lab.studies.v2.budget import finish, reserve
from decision_lab.studies.v2.protocol import freeze, settings, verify_comparison
from decision_lab.training.protocol import verify_protocol

config_directory = Path.cwd() / "configs" if modal.is_local() else Path("/root/configs")
spec, pilot, scratch, training, evaluation = settings(config_directory)
app = modal.App("decision-model-lab-v2")
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


def verify_runtime(root):
    comparison = verify_comparison(root)
    if comparison["settings"] != spec:
        raise ValueError("Runtime v2 settings differ from the frozen comparison")
    for arm in ("control", "treatment"):
        protocol = json.loads((root / arm / "protocol.json").read_text())
        for key, value in (
            ("pilot", pilot),
            ("training", training),
            ("scratch", scratch),
            ("evaluation", evaluation),
        ):
            if protocol[key] != value.model_dump():
                raise ValueError(f"Runtime {key} settings differ from the frozen arm")
    return comparison


@app.function(**resources, timeout=spec["pilot_timeout_seconds"])
def gpu_pilot():
    from decision_lab.studies.v2.study import run_pilot

    result = run_pilot(Path("/runs/study-v2"))
    volume.commit()
    return result


@app.function(**resources, timeout=spec["training_timeout_seconds"])
def gpu_train(arm, attempt, parent):
    from decision_lab.studies.v2.recovery import preserve_orphan_selections
    from decision_lab.training.runner import prune_recovery, run_training_chunk

    root = Path("/runs/study-v2")
    if (root / "test-evaluation-started.json").exists():
        raise ValueError("Training is closed after final evaluation begins")
    verify_runtime(root)
    protocol = json.loads((root / arm / "protocol.json").read_text())
    verify_protocol(protocol)
    preserved = preserve_orphan_selections(root / arm, parent, protocol["identity"] + ":head")
    if preserved:
        atomic_json(root / arm / f"{attempt}-orphan-preservation.json", preserved)
        volume.commit()
    result = run_training_chunk(
        root / arm, "head", pilot, scratch, training, protocol, attempt, parent
    )
    atomic_json(root / arm / "head-state.json", result)
    volume.commit()
    prune_recovery(root / arm / "head")
    volume.commit()
    return result


@app.function(**resources, timeout=spec["evaluation_timeout_seconds"])
def gpu_evaluate():
    from decision_lab.studies.v2.study import evaluate

    verify_runtime(Path("/runs/study-v2"))
    result = evaluate(Path("/runs/study-v2"))
    volume.commit()
    return result


@app.function(**resources, timeout=spec["verification_timeout_seconds"])
def gpu_verify():
    from decision_lab.studies.v2.study import verify_exports

    verify_runtime(Path("/runs/study-v2"))
    result = verify_exports(Path("/runs/study-v2"))
    volume.commit()
    return result


def invoke(root, stage, function, *args):
    index = reserve(root / "ledger.json", stage, spec, pilot)
    if stage in {"control", "treatment"}:
        finish(root / "ledger.json", index, "reserved", attempt=args[1])
    started = time.perf_counter()
    try:
        result = function.remote(*args)
    except BaseException:
        finish(root / "ledger.json", index, "failed")
        raise
    finish(root / "ledger.json", index, "complete", client_seconds=time.perf_counter() - started)
    return result


@app.local_entrypoint()
def main(action: str):
    if action not in {"pilot", "freeze", "train", "evaluate", "verify", "all"}:
        raise ValueError("Unknown v2 action")
    root = Path("runs/study-v2")
    root.mkdir(parents=True, exist_ok=True)
    if action == "pilot":
        billing = json.loads((root / "billing-before.json").read_text())
        if float(billing["billed_cost"]) != 0:
            raise ValueError("Workspace has billed charges; credit-only preflight failed")
        remaining = 30 + float(billing["adjustments"]["credits"])
        if remaining < spec["compute_allocation_usd"] + spec["storage_reserve_usd"]:
            raise ValueError("Insufficient verified credits for v2 and storage reserve")
        if (root / "pilot.json").exists():
            raise FileExistsError("Pilot already completed")
        with volume.batch_upload() as upload:
            for arm in ("control", "treatment"):
                upload.put_directory(Path("data/v2") / arm, f"/study-v2/{arm}/data")
                upload.put_directory(Path("artifacts/v2-audit") / arm, f"/study-v2/{arm}/audit")
        atomic_json(root / "pilot.json", invoke(root, "pilot", gpu_pilot))
        return
    if action in {"freeze", "train", "all"}:
        if not (root / "comparison.json").exists():
            comparison = freeze(
                root,
                Path("data/v2"),
                Path("artifacts/v2-audit"),
                json.loads((root / "pilot.json").read_text()),
            )
            with volume.batch_upload() as upload:
                upload.put_file(root / "comparison.json", "/study-v2/comparison.json")
                for arm in ("control", "treatment"):
                    upload.put_file(root / arm / "protocol.json", f"/study-v2/{arm}/protocol.json")
                    upload.put_directory(root / arm / "test-inputs", f"/study-v2/{arm}/test-inputs")
            print(json.dumps({"comparison_frozen": comparison["identity"]}), flush=True)
    verify_comparison(root)
    if action in {"train", "all"}:
        for arm in ("control", "treatment"):
            path = root / arm / "head-state.json"
            state = json.loads(path.read_text()) if path.exists() else None
            entries = list(volume.iterdir(f"/study-v2/{arm}"))
            if any(entry.path.endswith("head-state.json") for entry in entries):
                remote = json.loads(b"".join(volume.read_file(f"/study-v2/{arm}/head-state.json")))
                if (
                    remote["protocol_identity"]
                    != json.loads((root / arm / "protocol.json").read_text())["identity"]
                ):
                    raise ValueError("Remote durable state belongs to another protocol")
                if state is None or remote["step"] > state["step"]:
                    state = remote
                    atomic_json(path, state)
                    print(
                        json.dumps({"recovered_durable_arm_state": arm, "step": state["step"]}),
                        flush=True,
                    )
            while state is None or not state["complete"]:
                attempt = "attempt-" + uuid.uuid4().hex
                state = invoke(
                    root, arm, gpu_train, arm, attempt, state["checkpoint"] if state else None
                )
                atomic_json(path, state)
                atomic_json(root / arm / f"{attempt}.json", state)
                print(
                    json.dumps({"arm": arm, "step": state["step"], "complete": state["complete"]}),
                    flush=True,
                )
    if action in {"evaluate", "all"} and not (root / "evaluation.json").exists():
        result = invoke(root, "evaluation", gpu_evaluate)
        atomic_json(root / "calibration-freeze.json", result["calibration"])
        atomic_json(root / "evaluation.json", result["evaluation"])
        for arm in ("control", "treatment"):
            for split in (
                "test_original",
                "test_counterfactual",
                "test_depth",
                "test_wording",
                "test_single_branch",
            ):
                path = root / "predictions" / arm / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    b"".join(volume.read_file(f"/study-v2/predictions/{arm}/{split}.jsonl"))
                )
    if action in {"verify", "all"} and not (root / "artifact-verification.json").exists():
        atomic_json(root / "artifact-verification.json", invoke(root, "verification", gpu_verify))
