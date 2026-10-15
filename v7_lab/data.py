import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import _candidate
from decision_lab.data.render import build_prompt, render_atom, render_rule
from decision_lab.data.schema import Label, Rule, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash
from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v2.data import example, parent_world, replace_fact
from v4_lab.data import triples
from v5_lab.data import atom, independent_closure, reference_label, remap
from v6_lab.data import exclusions as previous_exclusions

VARIANTS = (
    "base",
    "renamed",
    "reordered",
    "distractor",
    "negated_query",
    "wording",
    "support_edit",
    "support_edit_inverse",
)
TESTS = (
    "test_multi",
    "test_multi_transformations",
    "test_multi_depth",
    "test_multi_wording",
    "test_multi_signed_chain",
    "test_original",
    "test_counterfactual",
    "test_depth",
    "test_wording",
    "test_single_branch",
)


def exclusions(extra_paths=()):
    hashes, files = previous_exclusions()
    paths = (Path("runs/study-v6/data/families.jsonl"), *extra_paths)
    for path in paths:
        for line in path.read_text().splitlines():
            hashes.update(json.loads(line)["world_hashes"])
        files[str(path)] = file_hash(path)
    return hashes, files


def support_edit(world, query, label, shared, branch, fillers, inverse=False):
    entity = query.arguments[0]
    target = {Label.TRUE: Label.FALSE, Label.FALSE: Label.UNKNOWN, Label.UNKNOWN: Label.TRUE}
    if not inverse:
        target = {Label.TRUE: Label.UNKNOWN, Label.FALSE: Label.TRUE, Label.UNKNOWN: Label.FALSE}
    changed = target[label]
    facts = [f for f in world.facts if not (f.predicate == branch and f.arguments == (entity,))]
    others = [f for f in facts if f.arguments == (entity,) and f.predicate != shared]
    if changed == Label.UNKNOWN:
        missing = next(p for p in fillers if atom(p, entity) not in facts)
        facts.append(atom(missing, entity))
    else:
        if label == Label.UNKNOWN:
            facts.remove(others[0])
        facts.append(atom(branch, entity, query.negative ^ (changed == Label.FALSE)))
    return Theory(entities=world.entities, facts=tuple(facts), rules=world.rules), changed


def multi_bundle(
    config, rng, family, entities, assignment, negative, depth, signed, variants, target_slot=0
):
    predicates = rng.sample(rng.choice(config.predicates), depth + 4)
    shared, branch = predicates[:2]
    outputs, fillers = predicates[2 : 2 + depth], predicates[-2:]
    facts = []
    for entity, label in zip(entities, assignment, strict=True):
        facts.append(atom(shared, entity))
        if label == Label.UNKNOWN:
            facts.extend(atom(p, entity) for p in fillers)
        else:
            facts.append(atom(branch, entity, negative ^ (label == Label.FALSE)))
            facts.append(atom(rng.choice(fillers), entity))
    rules = []
    for sign in (False, True):
        conditions = (atom(branch, "?x", sign),)
        if not signed:
            conditions = (atom(shared, "?x"), *conditions)
        rules.append(Rule(conditions=conditions, conclusion=atom(outputs[0], "?x", sign)))
        for left, right in zip(outputs[:-1], outputs[1:], strict=True):
            rules.append(
                Rule(conditions=(atom(left, "?x", sign),), conclusion=atom(right, "?x", sign))
            )
    rng.shuffle(facts)
    rng.shuffle(rules)
    world = Theory(entities=entities, facts=tuple(facts), rules=tuple(rules))
    queries = tuple(atom(outputs[-1], entity, negative) for entity in entities)
    if [solve(world).answer(q)[0] for q in queries] != assignment:
        raise ValueError("Multi-entity construction and oracle disagree")
    slot = target_slot
    renamed, renamed_queries = remap(
        world, queries, dict(zip(entities, entities[1:] + entities[:1], strict=True))
    )
    noisy = world.model_copy(
        update={
            "facts": (
                *world.facts,
                *(
                    atom(p, entity, bool(index % 2))
                    for index, entity in enumerate(entities)
                    for p in rng.sample(next(g for g in config.predicates if shared not in g), 1)
                ),
            )
        }
    )
    forward, forward_label = support_edit(
        world, queries[slot], assignment[slot], shared, branch, fillers
    )
    reverse, reverse_label = support_edit(
        world, queries[slot], assignment[slot], shared, branch, fillers, True
    )
    worlds = {
        "base": (world, queries, "canonical"),
        "renamed": (renamed, renamed_queries, "canonical"),
        "reordered": (world, queries, "canonical"),
        "distractor": (noisy, queries, "canonical"),
        "negated_query": (world, tuple(q.opposite() for q in queries), "canonical"),
        "wording": (world, queries, "paraphrase"),
        "support_edit": (forward, queries, "canonical"),
        "support_edit_inverse": (reverse, queries, "canonical"),
    }
    rows, seed = [], rng.randrange(2**32)
    for variant in variants:
        theory, statements, template = worlds[variant]
        for index, query in enumerate(statements):
            row = example(
                theory,
                query,
                family,
                "multi",
                f"{variant}-{index}",
                "signed_chain" if signed else "conjunction",
                template,
                seed,
            )
            if variant == "reordered":
                row = row.model_copy(
                    update={
                        "facts": row.facts[::-1],
                        "rules": row.rules[::-1],
                        "prompt": build_prompt(row.facts[::-1], row.rules[::-1], row.question),
                    }
                )
            expected = assignment[index]
            if variant == "negated_query" and expected != Label.UNKNOWN:
                expected = Label.FALSE if expected == Label.TRUE else Label.TRUE
            if index == slot and variant in ("support_edit", "support_edit_inverse"):
                expected = forward_label if variant == "support_edit" else reverse_label
            if row.label != expected:
                raise ValueError("Multi-entity intervention affected an unintended answer")
            rows.append(row)
    return rows, {
        "component": "multi",
        "entities": entities,
        "assignment": [label.value for label in assignment],
        "depth": depth,
        "signed_chain": signed,
        "negative_query": negative,
        "target_slot": slot,
        "variants": list(variants),
    }


def generate(directory, spec, excluded, exclusion_files):
    if directory.exists():
        raise FileExistsError(directory)
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rng = random.Random(spec["dataset_seed"])
    registry, owners, splits, rejections = [], {}, defaultdict(list), Counter()

    def accept(family, partition, factory):
        for _ in range(spec["max_attempts_per_family"]):
            try:
                bundles, metadata = factory()
                hashes = {canonical_hash(r.theory) for rows in bundles.values() for r in rows}
                if hashes & (excluded | owners.keys()):
                    raise ValueError("Canonical world overlap")
            except (ValueError, InconsistentTheoryError) as error:
                rejections[str(error)] += 1
                continue
            owners.update(dict.fromkeys(hashes, family))
            registry.append(
                {
                    "family_id": family,
                    "partition": partition,
                    "world_hashes": sorted(hashes),
                    **metadata,
                }
            )
            for split, rows in bundles.items():
                splits[split].extend(r.model_copy(update={"split": split}) for r in rows)
            return
        raise RuntimeError(f"Generation attempt budget exhausted: {family}")

    def old(partition, count):
        for index in range(count):
            family = f"{spec['study']}-{partition}-old-{index:06d}"

            def factory(index=index, family=family):
                depth = rng.choice([4, 5] if partition == "depth" else [1, 2, 3])
                motif = rng.choice(["chain", "relational"])
                template = (
                    "paraphrase" if partition == "wording" else rng.choice(config.train_templates)
                )
                seed = rng.randrange(2**32)
                if partition == "single_branch":
                    setting = config.splits["train"].model_copy(update={"depths": [depth]})
                    theory, queries, motif = _candidate(config, setting, rng)
                    query = queries[rng.randrange(2)]
                    choices = [
                        replace_fact(theory, f, f.opposite())
                        for f in theory.facts
                        if solve(replace_fact(theory, f, f.opposite())).answer(query)[0]
                        == Label.UNKNOWN
                    ]
                    if not choices:
                        raise ValueError("No critical single-branch flip")
                    rows = [
                        example(w, query, family, partition, role, motif, template, seed)
                        for w, role in zip(
                            (theory, rng.choice(choices)), ("before", "after"), strict=True
                        )
                    ]
                    return {"test_single_branch": rows}, {"component": "old"}
                theory, query, unknown, critical, selector = parent_world(
                    config, depth, motif, partition == "wording", rng
                )
                matched, original, _ = triples(
                    theory,
                    query,
                    unknown,
                    critical,
                    selector,
                    family,
                    partition,
                    motif,
                    template,
                    seed,
                )
                if partition == "test":
                    bundles = {"test_original": original, "test_counterfactual": matched}
                elif partition in ("depth", "wording"):
                    bundles = {f"test_{partition}": matched}
                else:
                    bundles = {
                        "old_train" if partition == "train" else partition: original
                        if index % 2 == 0
                        else matched
                    }
                return bundles, {
                    "component": "old",
                    "curriculum": "original" if index % 2 == 0 else "counterfactual",
                }

            accept(family, partition, factory)
        print(json.dumps({"generated_old_partition": partition, "families": count}), flush=True)

    def multi(partition, groups, transfer=None):
        for size in (3, 4, 5):
            for group in range(groups):
                entities = tuple(config.train_entities[(group + i) % 8] for i in range(size))
                pattern = [0] * size if group % 6 == 0 else [rng.randrange(3) for _ in range(size)]
                for cycle in range(3):
                    family = f"{spec['study']}-{partition}-multi-{size}-{group:04d}-{cycle}"
                    assignment = [list(Label)[(value + cycle) % 3] for value in pattern]

                    def factory(
                        family=family, assignment=assignment, entities=entities, group=group
                    ):
                        variants = (
                            VARIANTS
                            if partition == "test_multi"
                            else (
                                "base",
                                ("renamed", "reordered", "support_edit", "support_edit_inverse")[
                                    group % 4
                                ],
                            )
                            if transfer is None
                            else ("wording",)
                            if transfer == "wording"
                            else ("base",)
                        )
                        depth = rng.choice([3, 4] if transfer == "depth" else [1, 2])
                        signed = transfer == "signed_chain" or (
                            transfer is None and partition != "test_multi" and group % 4 == 3
                        )
                        rows, record = multi_bundle(
                            config,
                            rng,
                            family,
                            entities,
                            assignment,
                            bool(group % 2),
                            depth,
                            signed,
                            variants,
                            group % len(entities),
                        )
                        record["sampling_block"] = family.rsplit("-", 1)[0]
                        if partition == "test_multi":
                            bundles = {
                                "test_multi": [r for r in rows if "-base-" in r.example_id],
                                "test_multi_transformations": rows,
                            }
                        else:
                            bundles = {"multi_train" if partition == "train" else partition: rows}
                        return bundles, record

                    accept(family, partition, factory)
        print(
            json.dumps({"generated_multi_partition": partition, "groups_per_size": groups}),
            flush=True,
        )

    old("train", spec["train_old_families"])
    multi("train", spec["train_multi_groups_per_size"])
    control = splits.pop("old_train")
    shared, replaced = control[: len(control) // 2], control[len(control) // 2 :]
    replacement = splits.pop("multi_train")
    queues = {label: [r for r in replacement if r.label == label] for label in Label}
    for rows in queues.values():
        rng.shuffle(rows)
    treatment = shared + [queues[r.label].pop() for r in replaced]
    if any(queues.values()) or len(control) != len(treatment):
        raise ValueError("Replacement label exposure differs")
    order = list(range(len(control)))
    rng.shuffle(order)
    for arm, rows in (("control", control), ("treatment", treatment)):
        splits[f"train_{arm}"] = [
            rows[i].model_copy(update={"split": f"train_{arm}"}) for i in order
        ]
    for partition in ("development", "selection_fit", "selection", "calibration"):
        old(partition, spec["auxiliary_old_families"])
        multi(partition, spec["auxiliary_multi_groups_per_size"])
        rng.shuffle(splits[partition])
    old("test", spec["test_old_families"])
    multi("test_multi", spec["test_multi_groups_per_size"])
    for transfer in ("depth", "wording", "single_branch"):
        old(transfer, spec["transfer_old_families"])
    for transfer in ("depth", "wording", "signed_chain"):
        multi(f"test_multi_{transfer}", spec["transfer_multi_groups_per_size"], transfer)
    directory.mkdir(parents=True)
    registry_path = directory / "families.jsonl"
    registry_path.write_text("".join(json.dumps(r) + "\n" for r in registry))
    manifests = {}
    for split, rows in splits.items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
        manifests[split] = {
            "sha256": file_hash(path),
            "questions": len(rows),
            "families": len({r.family_id for r in rows}),
            "labels": dict(Counter(r.label.value for r in rows)),
        }
    atomic_json(
        directory / "manifest.json",
        {
            "splits": manifests,
            "families_sha256": file_hash(registry_path),
            "exclusion_files": exclusion_files,
            "excluded_worlds": len(excluded),
            "unique_worlds": len(owners),
            "rejections": dict(rejections),
        },
    )
    return validate(directory, excluded)


def validate(directory, excluded):
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(directory / "families.jsonl") != manifest["families_sha256"]:
        raise ValueError("Family registry checksum differs")
    records = [json.loads(line) for line in (directory / "families.jsonl").read_text().splitlines()]
    parents, owners = {r["family_id"]: r for r in records}, {}
    if len(parents) != len(records):
        raise ValueError("Duplicate family identity")
    for record in records:
        for digest in record["world_hashes"]:
            if digest in owners or digest in excluded:
                raise ValueError("Family or prior-world overlap")
            owners[digest] = record["family_id"]
    total, observed_families = 0, set()
    for split, summary in manifest["splits"].items():
        path = directory / f"{split}.jsonl"
        if file_hash(path) != summary["sha256"]:
            raise ValueError("Dataset checksum differs")
        rows, closures = read_examples(path), {}
        if len({r.example_id for r in rows}) != len(rows):
            raise ValueError("Duplicate example identity")
        for row in rows:
            record = parents[row.family_id]
            partition = (
                "train"
                if split.startswith("train_")
                else "test"
                if split in ("test_original", "test_counterfactual")
                else "test_multi"
                if split == "test_multi_transformations"
                else split.removeprefix("test_")
                if split in ("test_depth", "test_wording", "test_single_branch")
                else split
            )
            if record["partition"] != partition or row.split != split:
                raise ValueError("Parent crossed partitions")
            if (
                canonical_hash(row.theory) != row.theory_hash
                or owners[row.theory_hash] != row.family_id
            ):
                raise ValueError("World is outside its parent")
            concrete = row.theory.model_dump_json()
            if concrete not in closures:
                closures[concrete] = independent_closure(row.theory)
            if reference_label(closures[concrete], row.query) != row.label:
                raise ValueError("Independent oracle label mismatch")
            resolution = solve(row.theory)
            if set(resolution.depths) != closures[concrete] or resolution.answer(row.query) != (
                row.label,
                row.proof_depth,
            ):
                raise ValueError("Production oracle mismatch")
            if sorted(row.facts) != sorted(render_atom(a, {}) + "." for a in row.theory.facts):
                raise ValueError("Fact rendering mismatch")
            if sorted(row.rules) != sorted(
                render_rule(r, row.renderer_id) for r in row.theory.rules
            ):
                raise ValueError("Rule rendering mismatch")
            if row.question != render_atom(row.query, {}) + "." or row.prompt != build_prompt(
                row.facts, row.rules, row.question
            ):
                raise ValueError("Serialized input mismatch")
            observed_families.add(row.family_id)
        if (
            len(rows) != summary["questions"]
            or len({r.family_id for r in rows}) != summary["families"]
        ):
            raise ValueError("Manifest count mismatch")
        total += len(rows)
    if observed_families != set(parents):
        raise ValueError("Unused family record")
    control, treatment = (
        read_examples(directory / f"train_{arm}.jsonl") for arm in ("control", "treatment")
    )
    if [r.label for r in control] != [r.label for r in treatment]:
        raise ValueError("Training label streams differ")
    if (
        len(control) != len(treatment)
        or sum(a.prompt == b.prompt for a, b in zip(control, treatment, strict=True))
        != len(control) // 2
    ):
        raise ValueError("Shared training exposure differs")
    if Counter(r.label for r in control) != Counter({label: len(control) // 3 for label in Label}):
        raise ValueError("Training labels are not balanced")
    return {
        "valid": True,
        "families": len(parents),
        "questions": total,
        "unique_worlds": len(owners),
        "prior_worlds_excluded": len(excluded),
        "training_questions_per_arm": len(control),
        "matched_label_stream": True,
        "shared_training_questions": len(control) // 2,
    }
