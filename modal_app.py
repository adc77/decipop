import json
import random
import uuid
from pathlib import Path

import modal

from decision_lab.budget import pilot_preflight
from decision_lab.data.validate import read_examples
from decision_lab.model_config import PilotConfig, load_pilot_config

config_path = (
    Path(__file__).parent / "configs/pilot.yaml"
    if modal.is_local()
    else Path("/root/configs/pilot.yaml")
)
config = load_pilot_config(config_path)
app = modal.App(config.app_name)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["train", "cloud"], extra_options="--no-dev")
    .add_local_python_source("decision_lab")
    .add_local_dir("configs", "/root/configs")
)
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True)


@app.function(
    image=image,
    gpu=config.gpu,
    cpu=config.cpu_cores,
    memory=config.memory_mib,
    timeout=config.timeout_seconds,
    startup_timeout=config.startup_timeout_seconds,
    scaledown_window=config.scaledown_seconds,
    max_containers=1,
    retries=0,
    volumes={"/runs": volume},
    env={"HF_HOME": "/runs/hf-cache"},
)
def gpu_pilot(settings: dict, examples: list[dict], run_id: str):
    from decision_lab.training.pilot import run_gpu_pilot

    result = run_gpu_pilot(PilotConfig.model_validate(settings), examples, Path("/runs") / run_id)
    volume.commit()
    return result


@app.local_entrypoint()
def main(dataset: str, token_audit: str, remaining_usd: float, output: str):
    budget = pilot_preflight(config, remaining_usd)
    directory = Path(dataset)
    audit = json.loads(Path(token_audit).read_text())
    if not audit["within_limit"] or audit["model_revision"] != config.revision:
        raise ValueError("Pilot requires a passing token audit for the pinned model")
    import hashlib

    manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    if audit["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("The token audit belongs to a different dataset")
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite local pilot report: {path}")
    rows = read_examples(directory / "train.jsonl")
    selected = random.Random(config.seed).sample(rows, config.examples)
    run_id = f"pilot-{uuid.uuid4().hex}"
    result = gpu_pilot.remote(
        config.model_dump(),
        [{"prompt": row.prompt, "label": row.label.value} for row in selected],
        run_id,
    )
    result["run_id"] = run_id
    result["budget_preflight"] = budget
    result["dataset_manifest_sha256"] = manifest_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_id": run_id, "report": str(path), "status": result["status"]}))
