import hashlib
import json
from collections import Counter
from pathlib import Path

from decision_lab.config import DatasetConfig
from decision_lab.data.render import build_prompt, render_atom, render_rule
from decision_lab.data.schema import Example, Label
from decision_lab.data.solve import solve
from decision_lab.data.split import canonical_hash, content_hash


def read_examples(path: Path) -> list[Example]:
    with path.open() as stream:
        return [Example.model_validate_json(line) for line in stream if line.strip()]


def has_diamond(example: Example) -> bool:
    dependencies: dict[str, set[str]] = {}
    for rule in example.theory.rules:
        dependencies[rule.conclusion.predicate] = {atom.predicate for atom in rule.conditions}
    for parents in dependencies.values():
        if len(parents) == 2:
            left, right = sorted(parents)
            if left in dependencies and right in dependencies:
                if dependencies[left] & dependencies[right]:
                    return True
    return False


def validate_dataset(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    config = DatasetConfig.model_validate(manifest["config"])
    if manifest["config_hash"] != content_hash(config.model_dump()):
        raise ValueError("Configuration hash mismatch")
    owners: dict[str, str] = {}
    identifiers: set[str] = set()
    family_owners: dict[str, str] = {}
    summaries = {}
    for split, meta in manifest["splits"].items():
        path = directory / meta["file"]
        if path.parent != directory or path.suffix != ".jsonl":
            raise ValueError("Invalid dataset manifest path")
        examples = read_examples(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["file_sha256"]:
            raise ValueError(f"Content hash mismatch: {split}")
        if content_hash([row.model_dump(mode="json") for row in examples]) != meta["sha256"]:
            raise ValueError(f"Content hash mismatch: {split}")
        families: dict[str, list[Example]] = {}
        for example in examples:
            if example.split != split or example.example_id in identifiers:
                raise ValueError("Wrong split or duplicate example identifier")
            identifiers.add(example.example_id)
            if example.family_id in family_owners and family_owners[example.family_id] != split:
                raise ValueError("A theory family crosses splits")
            family_owners[example.family_id] = split
            families.setdefault(example.family_id, []).append(example)
        spec = config.splits[split]
        for family in families.values():
            first = family[0]
            if len(family) != 3 or {row.label for row in family} != set(Label):
                raise ValueError("Each family must contain one question of each label")
            if any(row.theory != first.theory for row in family):
                raise ValueError("A family contains multiple underlying theories")
            digest = canonical_hash(first.theory)
            if digest in owners:
                raise ValueError(f"Canonical theory reused in {owners[digest]} and {split}")
            owners[digest] = split
            resolution = solve(first.theory)
            allowed_template = (
                [config.heldout_template] if spec.heldout_wording else config.train_templates
            )
            for row in family:
                if not set(row.query.arguments) <= set(row.theory.entities):
                    raise ValueError("Query contains undeclared entities or variables")
                if row.theory_hash != digest or resolution.answer(row.query) != (
                    row.label,
                    row.proof_depth,
                ):
                    raise ValueError("Incorrect theory hash, answer, or proof depth")
                if row.theory_depth != max(resolution.depths.values()):
                    raise ValueError("Incorrect theory depth")
                if row.proof_depth is not None and row.proof_depth not in spec.depths:
                    raise ValueError("Question proof depth is outside the split design")
                if row.renderer_id not in allowed_template or row.motif not in spec.motifs:
                    raise ValueError("Held-out template or motif contamination")
                expected_facts = sorted(render_atom(atom, {}) + "." for atom in row.theory.facts)
                expected_rules = sorted(
                    render_rule(rule, row.renderer_id) for rule in row.theory.rules
                )
                if sorted(row.facts) != expected_facts or sorted(row.rules) != expected_rules:
                    raise ValueError("Rendered facts or rules disagree with the formal theory")
                if row.question != render_atom(row.query, {}) + ".":
                    raise ValueError("Rendered question disagrees with the query")
                if row.prompt != build_prompt(row.facts, row.rules, row.question):
                    raise ValueError("Model prompt differs from the verified rendered input")
                if row.theory_depth > max(spec.depths):
                    raise ValueError("Theory exceeds the split's depth limit")
                allowed_entities = (
                    config.heldout_entities if spec.heldout_wording else config.train_entities
                )
                if not set(row.theory.entities) <= set(allowed_entities):
                    raise ValueError("Held-out entity vocabulary contamination")
            if split == "train" and (first.theory_depth > 3 or has_diamond(first)):
                raise ValueError("Training contains a reserved depth or diamond motif")
            if split == "test_structure" and not has_diamond(first):
                raise ValueError("Structure test is missing its reserved diamond motif")
        counts = dict(Counter(row.label.value for row in examples))
        if (
            len(examples) != meta["questions"]
            or len(families) != meta["families"]
            or counts != meta["labels"]
        ):
            raise ValueError("Manifest counts do not match dataset contents")
        summaries[split] = {"questions": len(examples), "families": len(families), "labels": counts}
    return {
        "valid": True,
        "token_length_verified": manifest["token_length_verified"],
        "splits": summaries,
    }
