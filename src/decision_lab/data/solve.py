from dataclasses import dataclass
from itertools import product

from decision_lab.data.schema import Atom, Label, Theory


@dataclass(frozen=True)
class Resolution:
    depths: dict[Atom, int]

    def answer(self, query: Atom) -> tuple[Label, int | None]:
        if any(term.startswith("?") for term in query.arguments):
            raise ValueError("Queries must be ground statements")
        if query in self.depths:
            return Label.TRUE, self.depths[query]
        if query.opposite() in self.depths:
            return Label.FALSE, self.depths[query.opposite()]
        return Label.UNKNOWN, None


class InconsistentTheoryError(ValueError):
    pass


def solve(theory: Theory) -> Resolution:
    grounded = []
    for rule in theory.rules:
        variables = sorted(
            {a for atom in rule.conditions for a in atom.arguments if a.startswith("?")}
        )
        for entities in product(theory.entities, repeat=len(variables)):
            bindings = dict(zip(variables, entities, strict=True))
            grounded.append(
                (
                    tuple(atom.ground(bindings) for atom in rule.conditions),
                    rule.conclusion.ground(bindings),
                )
            )
    depths = dict.fromkeys(theory.facts, 0)
    changed = True
    while changed:
        changed = False
        for conditions, conclusion in grounded:
            if all(condition in depths for condition in conditions):
                depth = 1 + max(depths[condition] for condition in conditions)
                if conclusion not in depths or depth < depths[conclusion]:
                    depths[conclusion] = depth
                    changed = True
    if any(atom.opposite() in depths for atom in depths):
        raise InconsistentTheoryError("Both a statement and its explicit negation are derivable")
    return Resolution(depths)
