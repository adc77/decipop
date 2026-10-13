from pathlib import Path

import yaml
from pydantic import Field, model_validator

from decision_lab.config import StrictModel


class TrainingConfig(StrictModel):
    seed: int
    deterministic: bool
    epochs: int = Field(gt=0)
    effective_batch: int = Field(gt=0)
    pretrained_microbatch: int = Field(gt=0)
    scratch_microbatch: int = Field(gt=0)
    evaluation_batch: int = Field(gt=0)
    backbone_lr: float = Field(gt=0)
    head_lr: float = Field(gt=0)
    scratch_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip: float = Field(gt=0)
    warmup_fraction: float = Field(gt=0, lt=1)
    evaluation_interval: int = Field(gt=0)
    chunk_steps: int = Field(gt=0)
    chunk_training_seconds: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    gate_steps: int = Field(gt=0)
    scratch_gate_steps: int = Field(gt=0)
    gate_examples: int = Field(gt=0)
    gate_warmup_steps: int = Field(ge=0)
    gate_accuracy: float = Field(gt=0, le=1)
    max_training_calls_per_model: int = Field(gt=0)
    max_gate_calls: int = Field(gt=0)
    max_evaluation_calls: int = Field(gt=0)
    project_compute_limit_usd: float = Field(gt=0)
    credits_reserved_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_batches(self):
        if max(self.pretrained_microbatch, self.scratch_microbatch) > self.effective_batch:
            raise ValueError("Microbatch exceeds the effective batch")
        if self.gate_warmup_steps >= self.gate_steps:
            raise ValueError("Gate warmup must precede measured steps")
        if self.chunk_training_seconds >= self.timeout_seconds:
            raise ValueError("Chunk requires time for evaluation and checkpointing")
        return self


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text()))
