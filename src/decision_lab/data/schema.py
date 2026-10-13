from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from decision_lab.config import StrictModel


class Label(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class Atom(StrictModel):
    predicate: str = Field(pattern=r"^[a-z][a-z_]*$")
    arguments: tuple[str, ...]
    negative: bool

    @model_validator(mode="after")
    def validate_arity(self):
        if len(self.arguments) not in (1, 2) or any(not arg for arg in self.arguments):
            raise ValueError("Atoms require one or two nonempty arguments")
        return self

    def opposite(self) -> Self:
        return self.model_copy(update={"negative": not self.negative})

    def ground(self, bindings: dict[str, str]) -> Self:
        arguments = tuple(bindings[arg] if arg.startswith("?") else arg for arg in self.arguments)
        return self.model_copy(update={"arguments": arguments})


class Rule(StrictModel):
    conditions: tuple[Atom, ...]
    conclusion: Atom

    @model_validator(mode="after")
    def validate_conditions(self):
        if len(self.conditions) not in (1, 2):
            raise ValueError("Rules require one or two conditions")
        body_variables = {
            term for atom in self.conditions for term in atom.arguments if term.startswith("?")
        }
        head_variables = {term for term in self.conclusion.arguments if term.startswith("?")}
        if not head_variables <= body_variables:
            raise ValueError("Conclusion variables must occur in the conditions")
        return self


class Theory(StrictModel):
    entities: tuple[str, ...]
    facts: tuple[Atom, ...]
    rules: tuple[Rule, ...]

    @model_validator(mode="after")
    def validate_theory(self):
        if not self.entities or len(set(self.entities)) != len(self.entities):
            raise ValueError("Theory entities must be nonempty and unique")
        if any(term.startswith("?") for term in self.entities):
            raise ValueError("Entity names cannot be variables")
        if len(set(self.facts)) != len(self.facts) or len(set(self.rules)) != len(self.rules):
            raise ValueError("Duplicate facts or rules are not allowed")
        atoms = [*self.facts, *(a for r in self.rules for a in (*r.conditions, r.conclusion))]
        arities: dict[str, int] = {}
        for atom in atoms:
            arity = len(atom.arguments)
            if atom.predicate in arities and arities[atom.predicate] != arity:
                raise ValueError("A predicate cannot have multiple arities")
            arities[atom.predicate] = arity
            for term in atom.arguments:
                if not term.startswith("?") and term not in self.entities:
                    raise ValueError(f"Undeclared entity: {term}")
        if any(term.startswith("?") for atom in self.facts for term in atom.arguments):
            raise ValueError("Initial facts must be ground")
        return self


class Example(StrictModel):
    example_id: str
    family_id: str
    split: str
    theory: Theory
    query: Atom
    label: Label
    proof_depth: int | None
    theory_depth: int
    motif: str
    renderer_id: str
    theory_hash: str
    structure_hash: str
    facts: tuple[str, ...]
    rules: tuple[str, ...]
    question: str
    prompt: str
