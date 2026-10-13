import json
import math
import re
import uuid
from pathlib import Path

import modal

from decision_lab.data.validate import validate_dataset
from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.model_config import load_pilot_config, load_scratch_config
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.config import load_training_config
from decision_lab.training.ledger import finish_invocation, reserve_invocation
from decision_lab.training.protocol import make_protocol, verify_protocol

config_directory = Path(__file__).parent / "configs" if modal.is_local() else Path("/root/configs")
pilot = load_pilot_config(config_directory / "pilot.yaml")
scratch = load_scratch_config(config_directory / "scratch.yaml")
training = load_training_config(config_directory / "training.yaml")
evaluation = load_evaluation_config(config_directory / "evaluation.yaml")
app = modal.App("decision-model-lab-study")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud"], extra_options="--no-dev")
    .add_local_python_source("decision_lab")
    .add_local_dir("configs", "/root/configs")
)
volume = modal.Volume.from_name(pilot.volume_name, create_if_missing=True)
resources = {
    "image": image,
    "gpu": pilot.gpu,
    "cpu": (pilot.cpu_cores, pilot.cpu_cores),
    "memory": (pilot.memory_mib, pilot.memory_mib),
    "timeout": training.timeout_seconds,
    "startup_timeout": pilot.startup_timeout_seconds,
    "scaledown_window": pilot.scaledown_seconds,
    "max_containers": 1,
    "retries": 0,
    "volumes": {"/runs": volume},
    "env": {"HF_HOME": "/runs/hf-cache", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
}


@app.function(**resources)
def training_gate(study: str, attempt: str):
    from decision_lab.training.gate import run_gpu_gate

    root = Path("/runs") / study
    result = run_gpu_gate(root, pilot, scratch, training, root / "gates" / attempt)
    volume.commit()
    return result


@app.function(**resources)
def training_chunk(study: str, architecture: str, attempt: str, parent: str | None):
    from decision_lab.training.runner import prune_recovery, run_training_chunk

    root = Path("/runs") / study
    if (root / "test-evaluation-started.json").exists():
        raise RuntimeError("Training is closed because final test evaluation has started")
    protocol = json.loads((root / "protocol.json").read_text())
    verify_protocol(protocol)
    result = run_training_chunk(
        root, architecture, pilot, scratch, training, protocol, attempt, parent
    )
    volume.commit()
    prune_recovery(root / architecture)
    volume.commit()
    return result


def reserve(local: Path, stage: str):
    cost = (
        (training.timeout_seconds + pilot.startup_timeout_seconds + pilot.scaledown_seconds)
        / 3600
        * pilot.hourly_usd
    )
    return reserve_invocation(local / "ledger.json", stage, cost, training, 1.0)


def upload_inputs(local: Path, dataset: Path, audit: Path, study: str):
    validate_dataset(dataset)
    token_audit = json.loads((audit / "token-audit.json").read_text())
    if (
        not token_audit["within_limit"]
        or token_audit["model_revision"] != pilot.revision
        or token_audit["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json")
    ):
        raise ValueError("Input token audit does not match this dataset and checkpoint")
    inventory = {
        str(Path(prefix) / path.relative_to(directory)): file_hash(path)
        for prefix, directory in (("data", dataset), ("audit", audit))
        for path in directory.rglob("*")
        if path.is_file()
    }
    marker = local / "uploaded.json"
    if marker.exists():
        if json.loads(marker.read_text()) != inventory:
            raise ValueError("Uploaded inputs changed; use a new study directory")
        return
    with volume.batch_upload() as batch:
        batch.put_directory(dataset, f"/{study}/data")
        batch.put_directory(audit, f"/{study}/audit")
    marker.write_text(json.dumps(inventory, indent=2) + "\n")


def freeze(local: Path, study: str, dataset: Path, audit: Path):
    gates = sorted(local.glob("gate-*.json"), key=lambda path: path.stat().st_mtime_ns)
    if not gates:
        raise ValueError("A successful training gate is required before freeze")
    gate = json.loads(gates[-1].read_text())
    if not gate["scratch_fit_passed"] or gate["training_config"] != training.model_dump():
        raise ValueError("Gate failed or belongs to different training settings")
    estimated_calls = {}
    manifest = json.loads((dataset / "manifest.json").read_text())
    total_steps = (
        math.ceil(manifest["splits"]["train"]["questions"] / training.effective_batch)
        * training.epochs
    )
    for architecture, report in gate["models"].items():
        usable = training.chunk_training_seconds / report["mean_optimizer_step_seconds"]
        steps = min(training.chunk_steps, usable)
        estimated_calls[architecture] = math.ceil(total_steps / steps)
        if estimated_calls[architecture] > training.max_training_calls_per_model:
            raise ValueError(f"Forecast exceeds call allocation for {architecture}")
    from decision_lab.evaluation.manifest import prepare_test_inputs

    test_inputs = local / "test-inputs"
    if not test_inputs.exists():
        prepare_test_inputs(dataset, audit, test_inputs, pilot, evaluation)
        with volume.batch_upload() as batch:
            batch.put_directory(test_inputs, f"/{study}/test-inputs")
    protocol = make_protocol(
        dataset,
        audit,
        pilot,
        scratch,
        training,
        evaluation.model_dump(),
        Path("uv.lock"),
        test_inputs,
    )
    path = local / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Frozen protocol differs from local inputs or training code")
        return protocol
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    with volume.batch_upload() as batch:
        batch.put_file(path, f"/{study}/protocol.json")
    (local / "forecast.json").write_text(
        json.dumps({"training_calls": estimated_calls}, indent=2) + "\n"
    )
    print(
        json.dumps({"protocol_frozen": protocol["identity"], "calls": estimated_calls}), flush=True
    )
    return protocol


@app.local_entrypoint()
def main(
    action: str,
    dataset: str = "data/pilot-v3",
    audit: str = "artifacts/token-audit-v2",
    study: str = "study-v1",
    architecture: str = "all",
):
    if re.fullmatch(r"study-[a-z0-9-]+", study) is None:
        raise ValueError("Study name must be a safe study-prefixed identifier")
    if action not in {"gate", "freeze", "train"}:
        raise ValueError("Action must be gate, freeze, or train")
    if architecture not in {"all", "token", "head", "scratch"}:
        raise ValueError("Invalid architecture")
    local = Path("runs") / study
    local.mkdir(parents=True, exist_ok=True)
    upload_inputs(local, Path(dataset), Path(audit), study)
    if action == "gate":
        attempt = "gate-" + uuid.uuid4().hex
        index = reserve(local, "gate")
        try:
            result = training_gate.remote(study, attempt)
        except BaseException:
            finish_invocation(
                local / "ledger.json", index, {"status": "failed", "attempt": attempt}
            )
            raise
        (local / f"{attempt}.json").write_text(json.dumps(result, indent=2) + "\n")
        finish_invocation(local / "ledger.json", index, {"status": "complete", "attempt": attempt})
        print(
            json.dumps(
                {
                    "gate_report": str(local / f"{attempt}.json"),
                    "scratch_fit": result["scratch_fit_passed"],
                }
            ),
            flush=True,
        )
        return
    protocol = freeze(local, study, Path(dataset), Path(audit))
    verify_protocol(protocol)
    if action == "freeze":
        return
    for name in ("token", "head", "scratch") if architecture == "all" else (architecture,):
        state = local / f"{name}-state.json"
        prior = json.loads(state.read_text()) if state.exists() else None
        if prior and prior["protocol_identity"] != protocol["identity"]:
            raise ValueError("Training state belongs to another protocol")
        while prior is None or not prior["complete"]:
            attempt = "attempt-" + uuid.uuid4().hex
            index = reserve(local, name)
            parent = prior["checkpoint"] if prior else None
            try:
                result = training_chunk.remote(study, name, attempt, parent)
            except BaseException:
                finish_invocation(
                    local / "ledger.json", index, {"status": "failed", "attempt": attempt}
                )
                raise
            (local / f"{name}-{attempt}.json").write_text(json.dumps(result, indent=2) + "\n")
            state.write_text(json.dumps(result, indent=2) + "\n")
            finish_invocation(
                local / "ledger.json", index, {"status": "complete", "attempt": attempt}
            )
            prior = result
            print(
                json.dumps(
                    {
                        "architecture": name,
                        "step": result["step"],
                        "complete": result["complete"],
                        "report": str(state),
                    }
                ),
                flush=True,
            )
