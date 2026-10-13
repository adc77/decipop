import random

from decision_lab.data.render import render, render_rule
from decision_lab.data.schema import Atom, Rule, Theory


def test_renderer_preserves_variable_identity_and_explicit_negation():
    rule = Rule(
        conditions=(
            Atom(predicate="supports", arguments=("?x", "?y"), negative=True),
            Atom(predicate="ready", arguments=("?y",), negative=False),
        ),
        conclusion=Atom(predicate="active", arguments=("?x",), negative=False),
    )
    sentence = render_rule(rule, "canonical")
    assert "person 1 does not support person 2" in sentence
    assert "person 2 is ready" in sentence
    assert "person 1 is active" in sentence


def test_prompt_contains_only_rendered_task_inputs():
    fact = Atom(predicate="ready", arguments=("Aman",), negative=False)
    theory = Theory(entities=("Aman",), facts=(fact,), rules=())
    facts, rules, query, prompt = render(theory, fact, "canonical", random.Random(17))
    assert facts == ("Aman is ready.",)
    assert rules == ()
    assert query == "Aman is ready."
    assert prompt.endswith("Answer:")
    assert "Facts:\nAman is ready." in prompt
    assert "proof_depth" not in prompt
    assert "family_id" not in prompt
