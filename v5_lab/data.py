import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.render import build_prompt, render_atom, render_rule
from decision_lab.data.schema import Atom, Label, Rule, Theory
from decision_lab.data.solve import solve
from decision_lab.data.split import canonical_hash
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.data import prior_worlds

VARIANTS = (
    "base",
    "renamed",
    "reordered",
    "distractor",
    "negated_query",
    "wording",
    "remove_support",
    "reverse_support",
    "repair_unknown",
)


def atom(predicate, entity, negative=False):
    return Atom(predicate=predicate, arguments=(entity,), negative=negative)


def independent_closure(theory):
    known = set(theory.facts)
    grounded = []
    for rule in theory.rules:
        variables = sorted({a for c in rule.conditions for a in c.arguments if a.startswith("?")})
        for names in itertools.product(theory.entities, repeat=len(variables)):
            bindings = dict(zip(variables, names, strict=True))
            grounded.append(
                (set(c.ground(bindings) for c in rule.conditions), rule.conclusion.ground(bindings))
            )
    for _ in range(1 + len(grounded)):
        added = {head for conditions, head in grounded if conditions <= known} - known
        if not added:
            if any(a.opposite() in known for a in known):
                raise ValueError("Independent oracle found a contradiction")
            return known
        known |= added
    raise RuntimeError("Independent closure did not converge")


def reference_label(closure, query):
    if query in closure:
        return Label.TRUE
    if query.opposite() in closure:
        return Label.FALSE
    return Label.UNKNOWN


def exclusions():
    hashes, files = prior_worlds()
    for name in (
        "runs/study-v3/data/families.jsonl",
        "runs/study-v4/diagnostic/families.jsonl",
        "runs/study-v4/training/families.jsonl",
    ):
        path = Path(name)
        for line in path.read_text().splitlines():
            hashes.update(json.loads(line)["world_hashes"])
        files[name] = file_hash(path)
    return hashes, files


def remap(world, queries, mapping):
    def replace(source):
        return source.model_copy(update={"arguments": tuple(mapping[a] for a in source.arguments)})

    renamed = Theory(
        entities=tuple(mapping[e] for e in world.entities),
        facts=tuple(replace(f) for f in world.facts),
        rules=world.rules,
    )
    return renamed, tuple(replace(q) for q in queries)


def edit_fact(world, source, destination):
    facts = list(world.facts)
    if source is not None:
        facts.remove(source)
    if destination is not None:
        facts.append(destination)
    return Theory(entities=world.entities, facts=tuple(facts), rules=world.rules)


def make_family(config, rng, index, family, suite):
    names = config.train_entities
    entities = tuple(names[(index + offset) % len(names)] for offset in range(3))
    assignment = list(itertools.permutations(list(Label)))[(index // 8) % 6]
    negative = bool((index // 48) % 2)
    depth = rng.choice([3, 4]) if suite == "deeper" else rng.choice([1, 2])
    predicates = rng.sample(rng.choice(config.predicates), depth + 4)
    shared, branch = predicates[:2]
    outputs = predicates[2 : 2 + depth]
    irrelevant = predicates[-2:]
    facts = [atom(shared, entity) for entity in entities]
    critical = {}
    for entity, label in zip(entities, assignment, strict=True):
        critical[entity] = atom(branch, entity, negative ^ (label == Label.FALSE))
        if label != Label.UNKNOWN:
            facts.append(critical[entity])
        fillers = (irrelevant[0],) if label == Label.TRUE else (irrelevant[1],)
        if label == Label.UNKNOWN:
            fillers = tuple(irrelevant)
        facts.extend(atom(p, entity) for p in fillers)
    rules = []
    for sign in (False, True):
        conditions = (atom(branch, "?x", sign),)
        if suite != "signed_chain":
            conditions = (atom(shared, "?x"), *conditions)
        rules.append(Rule(conditions=conditions, conclusion=atom(outputs[0], "?x", sign)))
        for previous, following in zip(outputs[:-1], outputs[1:], strict=True):
            rules.append(
                Rule(
                    conditions=(atom(previous, "?x", sign),), conclusion=atom(following, "?x", sign)
                )
            )
    rng.shuffle(facts)
    rng.shuffle(rules)
    world = Theory(entities=entities, facts=tuple(facts), rules=tuple(rules))
    queries = tuple(atom(outputs[-1], entity, negative) for entity in entities)
    if tuple(solve(world).answer(q)[0] for q in queries) != assignment:
        raise ValueError("Closed-form and proof oracle disagree")
    target = entities[assignment.index(Label.TRUE)]
    unknown = entities[assignment.index(Label.UNKNOWN)]
    mapping = dict(zip(entities, entities[1:] + entities[:1], strict=True))
    renamed, renamed_queries = remap(world, queries, mapping)
    reordered = world.model_copy(update={"facts": world.facts[::-1], "rules": world.rules[::-1]})
    other_group = next(group for group in config.predicates if shared not in group)
    noise_predicates = rng.sample(other_group, 2)
    noise = tuple(
        atom(p, e, bool((i + j) % 2))
        for i, p in enumerate(noise_predicates)
        for j, e in enumerate(entities)
    )
    noisy = world.model_copy(update={"facts": (*world.facts, *noise)})
    variants = {
        "base": (world, queries, "canonical"),
        "renamed": (renamed, renamed_queries, "canonical"),
        "reordered": (reordered, queries, "canonical"),
        "distractor": (noisy, queries, "canonical"),
        "negated_query": (world, tuple(q.opposite() for q in queries), "canonical"),
        "wording": (world, queries, "paraphrase"),
        "remove_support": (edit_fact(world, critical[target], None), queries, "canonical"),
        "reverse_support": (
            edit_fact(world, critical[target], critical[target].opposite()),
            queries,
            "canonical",
        ),
        "repair_unknown": (edit_fact(world, None, critical[unknown]), queries, "canonical"),
    }
    rows, closures = [], {}
    for variant, (theory, statements, template) in variants.items():
        resolution = solve(theory)
        closure = independent_closure(theory)
        if closure != set(resolution.depths):
            raise ValueError("Independent and production closures disagree")
        closures[variant] = sorted(a.model_dump_json() for a in closure)
        rendered_facts = [render_atom(f, {}) + "." for f in theory.facts]
        rendered_rules = [render_rule(r, template) for r in theory.rules]
        statement_order = list(enumerate(statements))
        if variant == "reordered":
            statement_order.reverse()
        for slot, query in statement_order:
            label, proof_depth = resolution.answer(query)
            if label != reference_label(closure, query):
                raise ValueError("Independent query label mismatch")
            question = render_atom(query, {}) + "."
            rows.append(
                {
                    "example_id": f"{family}-{variant}-{slot}",
                    "family_id": family,
                    "suite": suite,
                    "variant": variant,
                    "slot": slot,
                    "entity": query.arguments[0],
                    "base_entity": entities[slot],
                    "query": query.model_dump(),
                    "label": label.value,
                    "proof_depth": proof_depth,
                    "theory": theory.model_dump(),
                    "world_hash": canonical_hash(theory),
                    "facts": rendered_facts,
                    "rules": rendered_rules,
                    "question": question,
                    "prompt": build_prompt(rendered_facts, rendered_rules, question),
                }
            )
    record = {
        "family_id": family,
        "suite": suite,
        "depth": depth,
        "assignment": [label.value for label in assignment],
        "negative_query": negative,
        "entities": entities,
        "target_slot": assignment.index(Label.TRUE),
        "unknown_slot": assignment.index(Label.UNKNOWN),
        "world_hashes": sorted({row["world_hash"] for row in rows}),
        "critical_facts": {e: f.model_dump() for e, f in critical.items()},
        "proof_closures": closures,
    }
    audit_family(rows, record)
    return rows, record


def audit_family(rows, record):
    grouped = {(r["variant"], r["slot"]): r for r in rows}
    if len(rows) != len(VARIANTS) * 3 or len(grouped) != len(rows):
        raise ValueError("Incomplete or duplicate family")
    base = [grouped["base", slot] for slot in range(3)]
    if sorted(r["label"] for r in base) != sorted(label.value for label in Label):
        raise ValueError("Base world must support all three labels")
    prefix = [r["prompt"].split("Statement:")[0] for r in base]
    if len(set(prefix)) != 1:
        raise ValueError("Query-only comparison changed premises")
    for variant in VARIANTS:
        for slot in range(3):
            expected = base[slot]["label"]
            if variant == "negated_query":
                expected = {"TRUE": "FALSE", "FALSE": "TRUE", "UNKNOWN": "UNKNOWN"}[expected]
            elif variant in ("remove_support", "reverse_support") and slot == record["target_slot"]:
                expected = "UNKNOWN" if variant == "remove_support" else "FALSE"
            elif variant == "repair_unknown" and slot == record["unknown_slot"]:
                expected = "TRUE"
            if grouped[variant, slot]["label"] != expected:
                raise ValueError("Intervention changed an unintended oracle answer")


def validate_dataset(directory, excluded):
    records = [json.loads(line) for line in (directory / "families.jsonl").read_text().splitlines()]
    families = {record["family_id"]: record for record in records}
    if len(families) != len(records):
        raise ValueError("Duplicate family identity")
    owners, total = {}, 0
    for partition in ("pilot", "diagnostic"):
        grouped = defaultdict(list)
        rows = [
            json.loads(line) for line in (directory / f"{partition}.jsonl").read_text().splitlines()
        ]
        if len({r["example_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate example identity")
        for row in rows:
            grouped[row["family_id"]].append(row)
        for name, bundle in grouped.items():
            record = families[name]
            if record["partition"] != partition:
                raise ValueError("A parent family crossed partitions")
            audit_family(bundle, record)
            hashes, known = set(), {}
            for row in bundle:
                theory = Theory.model_validate_json(json.dumps(row["theory"]))
                query = Atom.model_validate_json(json.dumps(row["query"]))
                digest = canonical_hash(theory)
                if digest != row["world_hash"]:
                    raise ValueError("Recorded world hash differs")
                hashes.add(digest)
                concrete = theory.model_dump_json()
                if concrete not in known:
                    known[concrete] = independent_closure(theory)
                    if known[concrete] != set(solve(theory).depths):
                        raise ValueError("Independent closure mismatch")
                if reference_label(known[concrete], query).value != row["label"]:
                    raise ValueError("Saved oracle label differs")
                template = "paraphrase" if row["variant"] == "wording" else "canonical"
                facts = [render_atom(f, {}) + "." for f in theory.facts]
                rules = [render_rule(r, template) for r in theory.rules]
                question = render_atom(query, {}) + "."
                if (facts, rules, question, build_prompt(facts, rules, question)) != (
                    row["facts"],
                    row["rules"],
                    row["question"],
                    row["prompt"],
                ):
                    raise ValueError("Serialized theory and rendered input differ")
            if hashes != set(record["world_hashes"]) or hashes & excluded:
                raise ValueError("World inventory mismatch or prior-world overlap")
            for digest in hashes:
                if digest in owners:
                    raise ValueError("A canonical world crossed parent families")
                owners[digest] = name
        total += len(rows)
    if set(families) != set(owners.values()):
        raise ValueError("Family registry has missing rows")
    return {
        "valid": True,
        "questions": total,
        "families": len(families),
        "unique_canonical_worlds": len(owners),
        "prior_world_overlap": 0,
    }


def generate(directory, spec, excluded, exclusion_files, tokenizer):
    if directory.exists():
        raise FileExistsError(directory)
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rng = random.Random(spec["dataset_seed"])
    owners, records, partitions, rejects = {}, [], {}, Counter()
    for partition in ("pilot", "diagnostic"):
        rows = []
        for suite, count in spec["families"].items():
            size = spec["pilot_families_per_suite"] if partition == "pilot" else count
            for index in range(size):
                family = f"v5-{partition}-{suite}-{index:06d}"
                for _ in range(spec["max_attempts_per_family"]):
                    bundle, record = make_family(config, rng, index, family, suite)
                    lengths = [
                        len(ids)
                        for ids in tokenizer([row["prompt"] for row in bundle], truncation=False)[
                            "input_ids"
                        ]
                    ]
                    if max(lengths) > spec["max_length"]:
                        rejects["token_limit"] += 1
                        continue
                    hashes = set(record["world_hashes"])
                    if hashes & (excluded | owners.keys()):
                        rejects["canonical_overlap"] += 1
                        continue
                    owners.update(dict.fromkeys(hashes, family))
                    record["partition"] = partition
                    records.append(record)
                    rows.extend(bundle)
                    break
                else:
                    raise RuntimeError(f"Generation attempt limit: {family}")
        partitions[partition] = rows
    directory.mkdir(parents=True)
    for name, rows in partitions.items():
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "families.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    audit = {
        "excluded_worlds": len(excluded),
        "exclusion_files": exclusion_files,
        "partitions": {
            k: {"questions": len(v), "families": len(v) // 27} for k, v in partitions.items()
        },
        "rejections": dict(rejects),
        "base_name_label_counts": {},
        "base_slot_label_counts": {},
    }
    for suite in spec["families"]:
        names, slots = defaultdict(Counter), defaultdict(Counter)
        for row in partitions["diagnostic"]:
            if row["suite"] == suite and row["variant"] == "base":
                names[row["entity"]][row["label"]] += 1
                slots[row["slot"]][row["label"]] += 1
        for counts in [*names.values(), *slots.values()]:
            if len(set(counts.values())) != 1 or set(counts) != {label.value for label in Label}:
                raise ValueError("Names or query slots are not exactly label-balanced")
        audit["base_name_label_counts"][suite] = {k: dict(v) for k, v in names.items()}
        audit["base_slot_label_counts"][suite] = {k: dict(v) for k, v in slots.items()}
    atomic_json(directory / "audit.json", audit)
    return audit
