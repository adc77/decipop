from itertools import product

import pytest

from decision_lab.data.schema import Atom, Label, Rule, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve


def atom(predicate, *arguments, negative=False):
    return Atom(predicate=predicate, arguments=arguments, negative=negative)


def test_open_world_and_explicit_negation():
    positive = atom("ready", "Aman")
    missing = atom("active", "Aman")
    resolution = solve(Theory(entities=("Aman",), facts=(positive,), rules=()))
    assert resolution.answer(positive) == (Label.TRUE, 0)
    assert resolution.answer(positive.opposite()) == (Label.FALSE, 0)
    assert resolution.answer(missing) == (Label.UNKNOWN, None)
    assert resolution.answer(missing.opposite()) == (Label.UNKNOWN, None)


def test_shortest_proof_and_multiple_conditions():
    base, middle, target = [atom(name, "Aman") for name in ("base", "middle", "target")]
    enabled = atom("enabled", "Aman")
    rules = (
        Rule(conditions=(base,), conclusion=middle),
        Rule(conditions=(middle, enabled), conclusion=target),
        Rule(conditions=(base,), conclusion=target),
    )
    theory = Theory(entities=("Aman",), facts=(base, enabled), rules=rules)
    assert solve(theory).answer(target) == (Label.TRUE, 1)
    blocked = theory.model_copy(update={"facts": (base,), "rules": rules[:2]})
    assert solve(blocked).answer(target) == (Label.UNKNOWN, None)


def test_binary_variables_and_negative_conditions():
    relation = atom("supports", "Aman", "Bella")
    condition = atom("verified", "Bella", negative=True)
    rule = Rule(
        conditions=(atom("supports", "?x", "?y"), atom("verified", "?y", negative=True)),
        conclusion=atom("trusted", "?x"),
    )
    theory = Theory(entities=("Aman", "Bella"), facts=(relation, condition), rules=(rule,))
    assert solve(theory).answer(atom("trusted", "Aman")) == (Label.TRUE, 1)
    assert solve(theory).answer(atom("trusted", "Bella")) == (Label.UNKNOWN, None)


def test_rejects_contradictions_and_unsafe_variables():
    fact = atom("ready", "Aman")
    with pytest.raises(InconsistentTheoryError):
        solve(Theory(entities=("Aman",), facts=(fact, fact.opposite()), rules=()))
    with pytest.raises(ValueError, match="Conclusion variables"):
        Rule(conditions=(atom("ready", "?x"),), conclusion=atom("active", "?y"))


def test_cycles_terminate_without_inventing_support():
    first, second = atom("ready", "Aman"), atom("active", "Aman")
    rules = (
        Rule(conditions=(first,), conclusion=second),
        Rule(conditions=(second,), conclusion=first),
    )
    empty = Theory(entities=("Aman",), facts=(), rules=rules)
    assert solve(empty).answer(first) == (Label.UNKNOWN, None)
    assert solve(empty.model_copy(update={"facts": (first,)})).answer(second) == (Label.TRUE, 1)


def test_exhaustive_two_property_worlds_against_independent_boolean_reference():
    for world in product((None, False, True), repeat=2):
        for first_head, second_head in product((False, True), repeat=2):
            known = {("base", "Aman", sign) for sign in world[:1] if sign is not None}
            known |= {("enabled", "Aman", sign) for sign in world[1:] if sign is not None}
            expected = set(known)
            if ("base", "Aman", False) in expected:
                expected.add(("middle", "Aman", first_head))
            if ("middle", "Aman", first_head) in expected and (
                "enabled",
                "Aman",
                False,
            ) in expected:
                expected.add(("target", "Aman", second_head))
            facts = tuple(atom(name, entity, negative=sign) for name, entity, sign in known)
            rules = (
                Rule(
                    conditions=(atom("base", "?x"),),
                    conclusion=atom("middle", "?x", negative=first_head),
                ),
                Rule(
                    conditions=(atom("middle", "?x", negative=first_head), atom("enabled", "?x")),
                    conclusion=atom("target", "?x", negative=second_head),
                ),
            )
            resolution = solve(Theory(entities=("Aman",), facts=facts, rules=rules))
            actual = {(a.predicate, a.arguments[0], a.negative) for a in resolution.depths}
            assert actual == expected
