import json
import random
from pathlib import Path

import pytest
import torch

from decision_lab.model_config import ScratchConfig
from decision_lab.models.scratch_transformer import ScratchDecisionModel
from decision_lab.training.checkpoint import load_checkpoint, save_checkpoint
from decision_lab.training.core import epoch_order, make_scheduler, update


def setup_model(dropout=0.0):
    model = ScratchDecisionModel(
        ScratchConfig(
            vocab_size=32,
            hidden_size=16,
            heads=2,
            layers=1,
            intermediate_size=32,
            max_length=16,
            dropout=dropout,
            classes=3,
            pad_id=0,
            initializer_std=0.02,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = make_scheduler(optimizer, 8, 0.25)
    return model, optimizer, scheduler


def step(objects, microbatch):
    return update(
        *objects,
        [[2, 4, 7], [2, 5, 7], [2, 6, 7]],
        [0, 1, 2],
        microbatch,
        0,
        1.0,
        "cpu",
    )


def test_accumulation_weights_uneven_microbatches_correctly():
    torch.manual_seed(17)
    full = setup_model()
    torch.manual_seed(17)
    accumulated = setup_model()
    step(full, 3)
    step(accumulated, 2)
    for actual, expected in zip(accumulated[0].parameters(), full[0].parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)


def test_resume_preserves_next_update_schedule_and_random_state(tmp_path):
    torch.manual_seed(17)
    random.seed(17)
    original = setup_model(dropout=0.2)
    step(original, 2)
    step(original, 2)
    progress = {"epoch": 1, "offset": 32, "step": 2, "order": epoch_order(50, 17, 1)}
    save_checkpoint(tmp_path / "checkpoint", *original, progress, "protocol-17")
    expected = step(original, 2)
    expected_random = random.random()
    expected_torch = torch.rand(4)
    restored = setup_model(dropout=0.2)
    assert load_checkpoint(tmp_path / "checkpoint", *restored, "protocol-17") == progress
    actual = step(restored, 2)
    assert actual == expected
    assert random.random() == expected_random
    torch.testing.assert_close(torch.rand(4), expected_torch, atol=0, rtol=0)
    assert restored[2].state_dict() == original[2].state_dict()
    for actual_parameter, expected_parameter in zip(
        restored[0].parameters(), original[0].parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter, atol=0, rtol=0)


def test_checkpoint_rejects_tampering_identity_mismatch_and_overwrite(tmp_path):
    objects = setup_model()
    path = tmp_path / "checkpoint"
    save_checkpoint(path, *objects, {"step": 0}, "protocol-17")
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, *objects, "protocol-29")
    with pytest.raises(FileExistsError):
        save_checkpoint(path, *objects, {"step": 1}, "protocol-17")
    (path / "progress.json").write_text(json.dumps({"step": 99}))
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, *objects, "protocol-17")


def test_incomplete_checkpoint_is_not_loadable_and_failed_write_is_removed(tmp_path, monkeypatch):
    from decision_lab.training import checkpoint

    path = tmp_path / "checkpoint"

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(checkpoint, "save_model", fail)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(path, *setup_model(), {"step": 0}, "protocol-17")
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        load_checkpoint(path, *setup_model(), "protocol-17")


def test_epoch_order_is_reproducible_without_consuming_training_rng():
    torch.manual_seed(17)
    before = torch.get_rng_state()
    order = epoch_order(20, 17, 1)
    assert order == epoch_order(20, 17, 1)
    assert order != epoch_order(20, 17, 2)
    assert sorted(order) == list(range(20))
    torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)


def test_training_config_rejects_incompatible_batch_and_timeout():
    from decision_lab.training.config import TrainingConfig, load_training_config

    settings = load_training_config(Path("configs/training.yaml")).model_dump()
    with pytest.raises(ValueError, match="Microbatch"):
        TrainingConfig.model_validate(settings | {"pretrained_microbatch": 33})
    with pytest.raises(ValueError, match="Chunk requires"):
        TrainingConfig.model_validate(settings | {"timeout_seconds": 650})
