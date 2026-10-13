import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from decision_lab.config import DatasetConfig, SplitConfig
from decision_lab.data.render import render
from decision_lab.data.schema import Atom, Example, Label, Rule, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash, content_hash


def _property(predicate: str, entity: str, negative: bool) -> Atom:
    return Atom(predicate=predicate, arguments=(entity,), negative=negative)


def _candidate(config: DatasetConfig, spec: SplitConfig, rng: random.Random):
    names = config.heldout_entities if spec.heldout_wording else config.train_entities
    entities = tuple(rng.sample(names, rng.randint(config.entities_min, config.entities_max)))
    predicates = rng.sample(rng.choice(config.predicates), 8)
    target = entities[0]
    depth = rng.choice(spec.depths)
    motif = rng.choice(spec.motifs)
    polarities = [rng.choice([False, True]) for _ in predicates]
    facts = {_property(predicates[0], target, polarities[0])}
    rules = []
    if motif == "relational":
        links = [entities[index % len(entities)] for index in range(depth + 1)]
        for index in range(depth):
            relation = rng.choice(config.relations)
            facts.add(
                Atom(predicate=relation, arguments=(links[index + 1], links[index]), negative=False)
            )
            rules.append(
                Rule(
                    conditions=(
                        Atom(predicate=relation, arguments=("?x", "?y"), negative=False),
                        _property(predicates[index], "?y", polarities[index]),
                    ),
                    conclusion=_property(predicates[index + 1], "?x", polarities[index + 1]),
                )
            )
        target = links[-1]
        terminal = depth
        edges = []
    elif motif == "diamond":
        edges = [((0,), 1), ((1,), 2), ((1,), 3), ((2, 3), 4)]
        terminal = 4
    else:
        edges = [((index,), index + 1) for index in range(depth)]
        if motif == "conjunction" and depth:
            facts.add(_property(predicates[7], target, polarities[7]))
            edges = [((conditions[0], 7), head) for conditions, head in edges]
        terminal = depth
    for conditions, head in edges:
        rules.append(
            Rule(
                conditions=tuple(
                    _property(predicates[index], "?x", polarities[index]) for index in conditions
                ),
                conclusion=_property(predicates[head], "?x", polarities[head]),
            )
        )
    pool = []
    for entity in entities[1:]:
        for predicate in predicates:
            pool.extend([_property(predicate, entity, False), _property(predicate, entity, True)])
    for source in entities:
        for destination in entities:
            if source != destination:
                for predicate in config.relations:
                    pool.append(
                        Atom(
                            predicate=predicate,
                            arguments=(source, destination),
                            negative=rng.choice([False, True]),
                        )
                    )
    extra_count = rng.randint(config.extra_facts_min, config.extra_facts_max)
    facts.update(rng.sample(pool, extra_count))
    theory = Theory(
        entities=entities,
        facts=tuple(sorted(facts, key=lambda a: a.model_dump_json())),
        rules=tuple(rules),
    )
    conclusion = _property(predicates[terminal], target, polarities[terminal])
    resolution = solve(theory)
    unknown = []
    present = {atom.predicate for atom in resolution.depths if len(atom.arguments) == 1}
    for predicate in sorted(present):
        for entity in entities:
            atom = _property(predicate, entity, rng.choice([False, True]))
            if resolution.answer(atom)[0] == Label.UNKNOWN:
                unknown.append(atom)
    if not unknown:
        raise ValueError("Candidate has no unknown question over an existing predicate")
    if resolution.answer(conclusion) != (Label.TRUE, depth):
        raise ValueError("Candidate does not have the requested shortest proof depth")
    if max(resolution.depths.values()) > max(spec.depths):
        raise ValueError("Candidate exceeds the split's theory-depth limit")
    return theory, (conclusion, conclusion.opposite(), rng.choice(unknown)), motif


def generate_dataset(config: DatasetConfig, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {output}")
    output.mkdir(parents=True)
    rng = random.Random(config.seed)
    seen: dict[str, str] = {}
    splits = {}
    rejections = Counter()
    for split, spec in config.splits.items():
        examples = []
        attempts = 0
        families = 0
        while families < spec.families:
            attempts += 1
            if attempts > spec.families * config.max_attempts_per_family:
                raise RuntimeError(f"Generation exhausted its attempt budget for {split}")
            try:
                theory, queries, motif = _candidate(config, spec, rng)
            except InconsistentTheoryError:
                rejections["inconsistent"] += 1
                continue
            except ValueError as error:
                rejections[str(error)] += 1
                continue
            digest = canonical_hash(theory)
            if digest in seen:
                rejections["duplicate_theory"] += 1
                continue
            template = (
                config.heldout_template
                if spec.heldout_wording
                else rng.choice(config.train_templates)
            )
            family = f"{split}-{families:06d}"
            resolution = solve(theory)
            rows = []
            for index, query in enumerate(queries):
                facts, rules, question, prompt = render(theory, query, template, rng)
                label, depth = resolution.answer(query)
                rows.append(
                    Example(
                        example_id=f"{family}-q{index}",
                        family_id=family,
                        split=split,
                        theory=theory,
                        query=query,
                        label=label,
                        proof_depth=depth,
                        theory_depth=max(resolution.depths.values()),
                        motif=motif,
                        renderer_id=template,
                        theory_hash=digest,
                        structure_hash=content_hash(
                            {"motif": motif, "depth": max(resolution.depths.values())}
                        ),
                        facts=facts,
                        rules=rules,
                        question=question,
                        prompt=prompt,
                    )
                )
            if any(len(row.prompt) > config.max_text_characters for row in rows):
                rejections["text_length"] += 1
                continue
            seen[digest] = split
            examples.extend(rows)
            families += 1
        rng.shuffle(examples)
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for example in examples:
                stream.write(example.model_dump_json() + "\n")
        splits[split] = {
            "file": path.name,
            "questions": len(examples),
            "families": families,
            "sha256": content_hash([row.model_dump(mode="json") for row in examples]),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "labels": dict(Counter(row.label.value for row in examples)),
            "proof_depths": dict(Counter(str(row.proof_depth) for row in examples)),
            "motifs": dict(Counter(row.motif for row in examples)),
            "attempts": attempts,
        }
    manifest = {
        "schema_version": config.schema_version,
        "config": config.model_dump(),
        "config_hash": content_hash(config.model_dump()),
        "splits": splits,
        "rejections": dict(rejections),
        "generator_version": "0.1.0",
        "source_hash": source_hash(),
        "label_semantics": "open-world-explicit-negation",
        "token_length_verified": False,
        "structure_hash_semantics": (
            "construction motif and maximum proof depth; not graph isomorphism"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def source_hash() -> str:
    source = Path(__file__).parent.parent
    files = {
        str(path.relative_to(source)): path.read_text() for path in sorted(source.rglob("*.py"))
    }
    return content_hash(files)
