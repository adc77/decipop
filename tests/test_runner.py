import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import generate_dataset
from decision_lab.model_config import load_pilot_config, load_scratch_config
from decision_lab.tokenization import train_scratch_tokenizer
from decision_lab.training.config import load_training_config
from decision_lab.training.protocol import make_protocol
from decision_lab.training.runner import prune_recovery, run_training_chunk


def test_chunked_runner_matches_uninterrupted_training_and_prunes_recovery(tmp_path):
    pilot = load_pilot_config(Path("configs/pilot.yaml"))
    scratch = load_scratch_config(Path("configs/scratch.yaml")).model_copy(
        update={
            "vocab_size": 512,
            "hidden_size": 16,
            "heads": 2,
            "layers": 1,
            "intermediate_size": 32,
        }
    )
    training = load_training_config(Path("configs/training.yaml")).model_copy(
        update={
            "epochs": 1,
            "effective_batch": 8,
            "scratch_microbatch": 4,
            "evaluation_batch": 4,
            "evaluation_interval": 1,
            "chunk_steps": 1,
        }
    )
    root = tmp_path / "chunked"
    root.mkdir()
    generate_dataset(load_dataset_config(Path("configs/smoke.yaml")), root / "data")
    (root / "audit").mkdir()
    train_scratch_tokenizer(root / "data", root / "audit/scratch-tokenizer", scratch)
    (root / "test-inputs").mkdir()
    (root / "test-inputs/manifest.json").write_text("{}")
    uninterrupted = tmp_path / "uninterrupted"
    shutil.copytree(root, uninterrupted)
    protocol = make_protocol(
        root / "data",
        root / "audit",
        pilot,
        scratch,
        training,
        {},
        Path("uv.lock"),
        root / "test-inputs",
    )
    prior = None
    count = 0
    while prior is None or not prior["complete"]:
        prior = run_training_chunk(
            root,
            "scratch",
            pilot,
            scratch,
            training,
            protocol,
            f"attempt-{count}",
            prior["checkpoint"] if prior else None,
            "cpu",
        )
        prune_recovery(root / "scratch")
        count += 1
    expected_training = training.model_copy(update={"chunk_steps": 100})
    full = run_training_chunk(
        uninterrupted,
        "scratch",
        pilot,
        scratch,
        expected_training,
        protocol,
        "attempt-0",
        None,
        "cpu",
    )
    assert prior["examples_seen"] == full["examples_seen"]
    assert prior["step"] == full["step"]
    actual = load_file(root / "scratch/final/model.safetensors")
    expected = load_file(uninterrupted / "scratch/final/model.safetensors")
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert len(list((root / "scratch").glob("attempt-*/checkpoint-*"))) == 2
    assert (Path(prior["selected"]) / "model.safetensors").exists()
    assert (
        len(json.loads((Path(prior["checkpoint"]) / "progress.json").read_text())["history"])
        == prior["step"]
    )
