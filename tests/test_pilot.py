import runpy
from pathlib import Path

import pytest
import torch

from decision_lab.model_config import load_pilot_config
from decision_lab.training.pilot import run_gpu_pilot


def test_pilot_fails_on_cpu_before_downloading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        run_gpu_pilot(load_pilot_config(Path("configs/pilot.yaml")), [], tmp_path / "pilot")
    assert not (tmp_path / "pilot").exists()


def test_modal_app_definition_does_not_launch_remote_work():
    definition = runpy.run_path("src/decision_lab/studies/v1/pilot_app.py")
    assert definition["app"].name == definition["config"].app_name
