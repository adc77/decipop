from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from decision_lab.config import StrictModel


class PilotConfig(StrictModel):
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int
    max_length: int = Field(gt=0)
    class_tokens: list[str]
    class_labels: list[str]
    answer_prefix: str
    microbatch: int = Field(gt=0)
    steps: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    examples: int = Field(gt=0)
    backbone_lr: float = Field(gt=0)
    head_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip: float = Field(gt=0)
    cpu_threads: int = Field(gt=0)
    gpu: Literal["L4"]
    cpu_cores: int = Field(gt=0)
    memory_mib: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    startup_timeout_seconds: int = Field(gt=0)
    scaledown_seconds: int = Field(ge=0)
    gpu_hourly_usd: float = Field(gt=0)
    cpu_core_hourly_usd: float = Field(gt=0)
    memory_gib_hourly_usd: float = Field(gt=0)
    max_pilot_usd: float = Field(gt=0)
    required_remaining_usd: float = Field(gt=0)
    volume_name: str
    app_name: str

    @model_validator(mode="after")
    def validate_classes(self):
        if self.class_labels != ["TRUE", "FALSE", "UNKNOWN"]:
            raise ValueError("The shared class order must be TRUE, FALSE, UNKNOWN")
        if len(set(self.class_tokens)) != len(self.class_labels):
            raise ValueError("Each class needs a distinct answer token")
        if self.warmup_steps >= self.steps:
            raise ValueError("Warmup steps must be less than total steps")
        return self

    @property
    def hourly_usd(self) -> float:
        return (
            self.gpu_hourly_usd
            + self.cpu_cores * self.cpu_core_hourly_usd
            + self.memory_mib / 1024 * self.memory_gib_hourly_usd
        )


class ScratchConfig(StrictModel):
    vocab_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    heads: int = Field(gt=0)
    layers: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    classes: Literal[3]
    pad_id: int = Field(ge=0)
    initializer_std: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.hidden_size % self.heads:
            raise ValueError("Hidden dimension must be divisible by attention heads")
        if self.pad_id >= self.vocab_size:
            raise ValueError("Padding ID must be within the vocabulary")
        return self


def load_pilot_config(path: Path) -> PilotConfig:
    return PilotConfig.model_validate(yaml.safe_load(path.read_text()))


def load_scratch_config(path: Path) -> ScratchConfig:
    return ScratchConfig.model_validate(yaml.safe_load(path.read_text()))
