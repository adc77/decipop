import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SplitConfig(StrictModel):
    families: int = Field(gt=0)
    depths: list[int]
    motifs: list[Literal["chain", "conjunction", "relational", "diamond"]]
    heldout_wording: bool

    @model_validator(mode="after")
    def validate_design(self):
        if not self.depths or min(self.depths) < 0 or max(self.depths) > 5:
            raise ValueError("Reasoning depths must be nonempty and between zero and five")
        if not self.motifs:
            raise ValueError("At least one motif is required")
        if "diamond" in self.motifs and self.depths != [3]:
            raise ValueError("The reserved diamond motif requires depth three")
        return self


class DatasetConfig(StrictModel):
    schema_version: Literal[1]
    seed: int
    max_attempts_per_family: int = Field(gt=0)
    entities_min: int = Field(ge=3)
    entities_max: int
    extra_facts_min: int = Field(ge=0)
    extra_facts_max: int
    max_text_characters: int = Field(gt=0)
    train_templates: list[Literal["canonical", "conditional"]]
    heldout_template: Literal["paraphrase"]
    train_entities: list[str]
    heldout_entities: list[str]
    predicates: list[list[str]]
    relations: list[str]
    splits: dict[str, SplitConfig]

    @model_validator(mode="after")
    def validate_design(self):
        if self.entities_max < self.entities_min:
            raise ValueError("entities_max must be at least entities_min")
        if self.extra_facts_max < self.extra_facts_min:
            raise ValueError("extra_facts_max must be at least extra_facts_min")
        if not self.train_templates or not self.relations or not self.splits:
            raise ValueError("Templates, relations, and splits must be nonempty")
        if set(self.train_entities) & set(self.heldout_entities):
            raise ValueError("Held-out entity names overlap with training entity names")
        for names in (self.train_entities, self.heldout_entities):
            if len(set(names)) != len(names) or len(names) < self.entities_max:
                raise ValueError("Entity pools must be unique and cover entities_max")
        if not self.predicates or any(len(set(pool)) < 8 for pool in self.predicates):
            raise ValueError("Each predicate vocabulary needs eight unique predicates")
        words = [*self.relations, *(word for pool in self.predicates for word in pool)]
        if any(re.fullmatch(r"[a-z][a-z_]*", word) is None for word in words):
            raise ValueError("Predicate names must be lowercase identifiers")
        if set(self.relations) & {word for pool in self.predicates for word in pool}:
            raise ValueError("Unary and binary predicates cannot share a name")
        if "train" not in self.splits:
            raise ValueError("A training split is required")
        train = self.splits["train"]
        if train.heldout_wording or "diamond" in train.motifs or max(train.depths) > 3:
            raise ValueError("Training cannot contain held-out wording, motifs, or depths")
        return self


def load_dataset_config(path: Path) -> DatasetConfig:
    with path.open() as stream:
        return DatasetConfig.model_validate(yaml.safe_load(stream))
