import json

import pytest

from decision_lab.data.schema import Atom, Rule, Theory
from decision_lab.data.solve import solve
from decision_lab.data.split import canonical_hash
from decision_lab.publishing.dataset import normalize_row, rename_symbols
from decision_lab.studies.v5.data import independent_closure, reference_label


def test_name_normalization_preserves_all_query_answers_and_world_identity():
    theory = Theory(
        entities=("Original", "Other"),
        facts=(Atom(predicate="verified", arguments=("Original",), negative=False),),
        rules=(
            Rule(
                conditions=(Atom(predicate="verified", arguments=("?x",), negative=False),),
                conclusion=Atom(predicate="authorized", arguments=("?x",), negative=False),
            ),
        ),
    )
    row = {
        "theory": theory.model_dump(mode="json"),
        "prompt": "Original is verified. Is Original authorized? Other is not Original.",
    }
    clean = normalize_row(row, "Original", "EntityA")
    renamed = Theory.model_validate_json(json.dumps(clean["theory"]))
    assert canonical_hash(theory) == canonical_hash(renamed)
    for entity in theory.entities:
        for predicate in ("verified", "authorized", "unknown"):
            for negative in (False, True):
                query = Atom(predicate=predicate, arguments=(entity,), negative=negative)
                new_query = Atom.model_validate_json(
                    json.dumps(rename_symbols(query.model_dump(mode="json"), "Original", "EntityA"))
                )
                assert solve(theory).answer(query) == solve(renamed).answer(new_query)
                assert reference_label(independent_closure(theory), query) == reference_label(
                    independent_closure(renamed), new_query
                )
    assert "Original" not in str(clean)
    assert row["theory"]["entities"][0] == "Original"


def test_name_normalization_rejects_entity_collision():
    with pytest.raises(ValueError, match="new and distinct"):
        normalize_row({"theory": {"entities": ["Original", "EntityA"]}}, "Original", "EntityA")
