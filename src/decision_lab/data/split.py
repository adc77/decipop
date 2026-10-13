import hashlib
import json
from itertools import permutations

from decision_lab.data.schema import Atom, Theory


def content_hash(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _atom(atom: Atom, names: dict[str, str]):
    return atom.predicate, tuple(names[arg] for arg in atom.arguments), atom.negative


def canonical_hash(theory: Theory) -> str:
    best = None
    for ordering in permutations(theory.entities):
        entities = {name: f"e{index}" for index, name in enumerate(ordering)}
        facts = sorted(_atom(atom, entities) for atom in theory.facts)
        rules = sorted(_canonical_rule(rule, entities) for rule in theory.rules)
        candidate = json.dumps([facts, rules], separators=(",", ":"))
        if best is None or candidate < best:
            best = candidate
    return hashlib.sha256(best.encode()).hexdigest()


def _canonical_rule(rule, entities):
    variables = sorted({a for atom in rule.conditions for a in atom.arguments if a.startswith("?")})
    variants = []
    for ordering in permutations(variables):
        names = entities | {name: f"?v{index}" for index, name in enumerate(ordering)}
        variants.append(
            (
                tuple(sorted(_atom(atom, names) for atom in rule.conditions)),
                _atom(rule.conclusion, names),
            )
        )
    return min(variants)
