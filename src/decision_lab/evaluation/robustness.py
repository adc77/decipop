import json
import random
from collections import defaultdict

from decision_lab.data.render import build_prompt, render_atom, render_rule
from decision_lab.data.schema import Atom, Example, Label, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash, content_hash


def renamed(theory: Theory, query: Atom):
    mapping = {name: f"Entity{index}" for index, name in enumerate(theory.entities)}

    def atom(value):
        return value.model_copy(
            update={"arguments": tuple(mapping.get(term, term) for term in value.arguments)}
        )

    updated = Theory(
        entities=tuple(mapping.values()),
        facts=tuple(atom(value) for value in theory.facts),
        rules=tuple(
            rule.model_copy(
                update={
                    "conditions": tuple(atom(value) for value in rule.conditions),
                    "conclusion": atom(rule.conclusion),
                }
            )
            for rule in theory.rules
        ),
    )
    return updated, atom(query)


def causal_theory(theory: Theory, query: Atom, transformation: str):
    original_label, _ = solve(theory).answer(query)
    for index, fact in enumerate(theory.facts):
        facts = list(theory.facts)
        if transformation == "remove_premise":
            facts.pop(index)
        else:
            facts[index] = fact.opposite()
        candidate = Theory(entities=theory.entities, facts=tuple(facts), rules=theory.rules)
        try:
            if solve(candidate).answer(query)[0] != original_label:
                return candidate
        except InconsistentTheoryError:
            continue
    return None


def rebuild(row: Example, theory: Theory, query: Atom, transformation: str, rng):
    renderer = "paraphrase" if transformation == "paraphrase" else row.renderer_id
    facts = [render_atom(atom, {}) + "." for atom in theory.facts]
    rules = [render_rule(rule, renderer) for rule in theory.rules]
    if transformation == "reorder":
        rng.shuffle(facts)
        rng.shuffle(rules)
    question = render_atom(query, {}) + "."
    resolution = solve(theory)
    label, proof_depth = resolution.answer(query)
    payload = row.model_dump(mode="json") | {
        "example_id": row.example_id + ":" + transformation,
        "family_id": row.family_id + ":" + transformation,
        "split": "robust_" + transformation,
        "theory": theory.model_dump(mode="json"),
        "query": query.model_dump(mode="json"),
        "label": label.value,
        "proof_depth": proof_depth,
        "theory_depth": max(resolution.depths.values()),
        "renderer_id": renderer,
        "theory_hash": canonical_hash(theory),
        "structure_hash": content_hash({"transformation": transformation, "source": row.family_id}),
        "facts": facts,
        "rules": rules,
        "question": question,
        "prompt": build_prompt(facts, rules, question),
    }
    return Example.model_validate_json(json.dumps(payload))


def generate_variants(rows: list[Example], families: int, seed: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.family_id].append(row)
    names = sorted(grouped)
    random.Random(seed).shuffle(names)
    output = {}
    manifest = {}
    for transformation in (
        "rename",
        "reorder",
        "irrelevant",
        "paraphrase",
        "remove_premise",
        "flip_fact",
    ):
        rng = random.Random(seed)
        variants = []
        exclusions = 0
        selected = 0
        for name in names:
            family = grouped[name]
            theory = family[0].theory
            if transformation in {"remove_premise", "flip_fact"}:
                query = next(row.query for row in family if row.label == Label.TRUE)
                theory = causal_theory(theory, query, transformation)
                if theory is None:
                    exclusions += 1
                    continue
            if transformation == "irrelevant":
                theory = Theory(
                    entities=theory.entities,
                    rules=theory.rules,
                    facts=(
                        *theory.facts,
                        Atom(
                            predicate="irrelevant", arguments=(theory.entities[0],), negative=False
                        ),
                    ),
                )
            for row in family:
                updated, query = (
                    renamed(theory, row.query)
                    if transformation == "rename"
                    else (theory, row.query)
                )
                variant = rebuild(row, updated, query, transformation, rng)
                if (
                    transformation not in {"remove_premise", "flip_fact"}
                    and variant.label != row.label
                ):
                    raise RuntimeError(
                        "A label-preserving transformation changed the oracle answer"
                    )
                variants.append(
                    {
                        "example": variant,
                        "base_example_id": row.example_id,
                        "label_changed": variant.label != row.label,
                    }
                )
            selected += 1
            if selected == families:
                break
        if selected != families:
            raise ValueError(f"Insufficient eligible families for {transformation}")
        output[transformation] = variants
        manifest[transformation] = {
            "families": selected,
            "questions": len(variants),
            "ineligible_families": exclusions,
            "oracle_label_changes": sum(row["label_changed"] for row in variants),
        }
    return output, manifest
