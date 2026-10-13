from pathlib import Path

import yaml
from pydantic import Field, model_validator

from decision_lab.config import StrictModel


class EvaluationConfig(StrictModel):
    seed: int
    calibration_max_iterations: int = Field(gt=0)
    calibration_learning_rate: float = Field(gt=0)
    calibration_min_temperature: float = Field(gt=0, lt=1)
    calibration_max_temperature: float = Field(gt=1)
    ece_bins: int = Field(gt=0)
    high_confidence_threshold: float = Field(gt=0, le=1)
    bootstrap_repetitions: int = Field(gt=0)
    bootstrap_confidence: float = Field(gt=0, lt=1)
    robustness_families: int = Field(gt=0)
    performance_requests: int = Field(gt=0)
    performance_warmup: int = Field(ge=0)
    performance_batch: int = Field(gt=0)
    coverage_targets: list[float]
    evaluation_timeout_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_targets(self):
        if not self.coverage_targets or any(not 0 < value <= 1 for value in self.coverage_targets):
            raise ValueError("Coverage targets must be between zero and one")
        if self.performance_requests < self.performance_batch:
            raise ValueError("Performance requests must cover a full batch")
        return self


def load_evaluation_config(path: Path) -> EvaluationConfig:
    return EvaluationConfig.model_validate(yaml.safe_load(path.read_text()))
