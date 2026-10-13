import json
import random
import shutil
from collections import Counter
from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import _candidate
from decision_lab.data.render import build_prompt, render, render_atom, render_rule
from decision_lab.data.schema import Atom, Example, Label, Rule, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash, content_hash
from decision_lab.data.validate import read_examples
from decision_lab.training.checkpoint import file_hash

ARMS = ("control", "treatment")
TESTS = ("test_original", "test_counterfactual", "test_depth", "test_wording", "test_single_branch")


def unary(predicate, entity, negative):
    return Atom(predicate=predicate, arguments=(entity,), negative=negative)


def replace_fact(theory, source, destination):
    if source not in theory.facts:
        raise ValueError("Critical fact is absent")
    facts = tuple(atom for atom in theory.facts if atom != source)
    if destination is not None:
        facts += (destination,)
    return Theory(entities=theory.entities, facts=facts, rules=theory.rules)


def parent_world(config, depth, motif, wording, rng):
    names = config.heldout_entities if wording else config.train_entities
    entities = tuple(rng.sample(names, rng.choice([3, 4])))
    predicates = rng.sample(rng.choice(config.predicates), 8)
    signs = [rng.choice([False, True]) for _ in predicates]
    links = [entities[i % len(entities)] for i in range(depth)]
    seed = unary(predicates[0], links[0], signs[0])
    facts = {seed}
    rules = []
    for i in range(depth - 1):
        if motif == "relational":
            relation = Atom(
                predicate=rng.choice(config.relations),
                arguments=(links[i + 1], links[i]),
                negative=False,
            )
            facts.add(relation)
            conditions = (
                relation.model_copy(update={"arguments": ("?x", "?y")}),
                unary(predicates[i], "?y", signs[i]),
            )
        else:
            conditions = (unary(predicates[i], "?x", signs[i]),)
        rules.append(
            Rule(conditions=conditions, conclusion=unary(predicates[i + 1], "?x", signs[i + 1]))
        )
    target = links[-1] if motif == "relational" else links[0]
    selector = unary(predicates[7], target, signs[7])
    facts.add(selector)
    query = unary(predicates[depth], target, signs[depth])
    for negative in (False, True):
        branch = selector.opposite() if negative else selector
        head = query.opposite() if negative else query
        rules.append(
            Rule(
                conditions=(
                    unary(predicates[depth - 1], "?x", signs[depth - 1]),
                    branch.model_copy(update={"arguments": ("?x",)}),
                ),
                conclusion=head.model_copy(update={"arguments": ("?x",)}),
            )
        )
    noise = [
        unary(predicate, entity, sign)
        for entity in entities
        for predicate in predicates[:7]
        for sign in (False, True)
        if entity != target and predicate != seed.predicate
    ]
    facts.update(rng.sample(noise, rng.randint(4, 8)))
    theory = Theory(
        entities=entities,
        facts=tuple(sorted(facts, key=lambda atom: atom.model_dump_json())),
        rules=tuple(rules),
    )
    if solve(theory).answer(query) != (Label.TRUE, depth):
        raise ValueError("Parent does not have its requested proof")
    distractors = [
        unary(query.predicate, entity, rng.choice([False, True]))
        for entity in entities
        if entity != target
    ]
    unknown = next((q for q in distractors if solve(theory).answer(q)[0] == Label.UNKNOWN), None)
    if unknown is None:
        raise ValueError("No original-style UNKNOWN query")
    return theory, query, unknown, seed, selector


def example(theory, query, family, split, role, motif, template, render_seed):
    resolution = solve(theory)
    label, depth = resolution.answer(query)
    facts, rules, question, prompt = render(theory, query, template, random.Random(render_seed))
    return Example(
        example_id=f"{family}-{split}-{role}",
        family_id=family,
        split=split,
        theory=theory,
        query=query,
        label=label,
        proof_depth=depth,
        theory_depth=max(resolution.depths.values()),
        motif=motif,
        renderer_id=template,
        theory_hash=canonical_hash(theory),
        structure_hash=content_hash({"motif": motif, "depth": depth}),
        facts=facts,
        rules=rules,
        question=question,
        prompt=prompt,
    )


def old_hashes(directory):
    hashes = set()
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            hashes.add(json.loads(line)["theory_hash"])
    return hashes


def generate(output: Path, settings: dict, config_path=Path("configs/dataset.yaml")):
    if output.exists():
        raise FileExistsError(output)
    config = load_dataset_config(config_path)
    rng = random.Random(settings["dataset_seed"])
    excluded = old_hashes(Path("data/pilot-v3"))
    owners = {}
    partitions = {
        "train": settings["train_families"],
        "development": settings["development_families"],
        "calibration": settings["calibration_families"],
        "test": settings["test_families"],
        "depth": settings["transfer_families"],
        "wording": settings["transfer_families"],
        "single_branch": settings["transfer_families"],
    }
    arm_rows = {arm: {} for arm in ARMS}
    records = []
    rejections = Counter()
    for partition, count in partitions.items():
        accepted = 0
        attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"Exhausted candidate budget for {partition}")
            depth = rng.choice([4, 5] if partition == "depth" else [1, 2, 3])
            motif = rng.choice(["chain", "relational"])
            template = (
                "paraphrase" if partition == "wording" else rng.choice(config.train_templates)
            )
            family = f"v2-{partition}-{accepted:06d}"
            render_seed = rng.randrange(2**32)
            try:
                if partition == "single_branch":
                    spec = config.splits["train"].model_copy(update={"depths": [depth]})
                    theory, queries, motif = _candidate(config, spec, rng)
                    query = queries[rng.randrange(2)]
                    flips = []
                    for fact in theory.facts:
                        modified = replace_fact(theory, fact, fact.opposite())
                        try:
                            if solve(modified).answer(query)[0] == Label.UNKNOWN:
                                flips.append((fact, modified))
                        except InconsistentTheoryError:
                            continue
                    if not flips:
                        raise ValueError("No single-branch broken-support flip")
                    critical, modified = rng.choice(flips)
                    rows = [
                        example(
                            t,
                            query,
                            family,
                            "test_single_branch",
                            role,
                            motif,
                            template,
                            render_seed,
                        )
                        for role, t in (("before", theory), ("after", modified))
                    ]
                    variants = [(theory, query), (modified, query)]
                    control = treatment = rows
                    intervention = "flip"
                else:
                    theory, query, unknown, critical, selector = parent_world(
                        config, depth, motif, partition == "wording", rng
                    )
                    intervention = "remove" if partition == "train" and accepted % 2 else "flip"
                    modified = replace_fact(
                        theory, critical, None if intervention == "remove" else critical.opposite()
                    )
                    false_world = replace_fact(theory, selector, selector.opposite())
                    variants = [(theory, query), (false_world, query), (modified, query)]
                    if [solve(t).answer(q)[0] for t, q in variants] != list(Label):
                        raise ValueError("Matched worlds have incorrect labels")
                    split = {
                        "test": "test_counterfactual",
                        "depth": "test_depth",
                        "wording": "test_wording",
                    }.get(partition, partition)
                    treatment = [
                        example(t, q, family, split, role, motif, template, render_seed)
                        for (t, q), role in zip(variants, ("true", "false", "unknown"), strict=True)
                    ]
                    original_split = "test_original" if partition == "test" else split
                    control = [
                        example(
                            theory, q, family, original_split, role, motif, template, render_seed
                        )
                        for q, role in zip(
                            (query, query.opposite(), unknown),
                            ("true", "false", "unknown"),
                            strict=True,
                        )
                    ]
                hashes = {canonical_hash(t) for t, _ in variants}
                if hashes & excluded or any(h in owners for h in hashes):
                    raise ValueError("Canonical overlap with another parent or v1")
            except (InconsistentTheoryError, ValueError) as error:
                rejections[str(error)] += 1
                continue
            if max(len(row.prompt) for row in control + treatment) > 3100:
                rejections["text_length"] += 1
                continue
            owners.update(dict.fromkeys(hashes, family))
            records.append(
                {
                    "family_id": family,
                    "partition": partition,
                    "depth": depth,
                    "critical_fact": critical.model_dump(),
                    "unknown_intervention": intervention,
                    "world_hashes": sorted(hashes),
                }
            )
            for arm in ARMS:
                target = arm_rows[arm]
                if partition == "train":
                    rows = control if arm == "control" else treatment
                elif partition in {"development", "calibration"}:
                    rows = control if accepted % 2 == 0 else treatment
                elif partition == "test":
                    target.setdefault("test_original", []).extend(control)
                    rows = treatment
                else:
                    rows = treatment
                target.setdefault(rows[0].split, []).extend(rows)
            accepted += 1
            if accepted % 1000 == 0 or accepted == count:
                print(
                    json.dumps({"generated_partition": partition, "families": accepted}), flush=True
                )
    output.mkdir(parents=True)
    (output / "families.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records)
    )
    for arm, splits in arm_rows.items():
        directory = output / arm
        directory.mkdir()
        metadata = {}
        for split, rows in splits.items():
            order = list(range(len(rows)))
            random.Random(settings["dataset_seed"] + len(split)).shuffle(order)
            rows = [rows[i] for i in order]
            path = directory / f"{split}.jsonl"
            path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
            metadata[split] = {
                "file": path.name,
                "file_sha256": file_hash(path),
                "questions": len(rows),
                "families": len({row.family_id for row in rows}),
                "labels": dict(Counter(row.label.value for row in rows)),
            }
        manifest = {
            "arm": arm,
            "settings": settings,
            "base_config": config.model_dump(),
            "splits": metadata,
            "v1_theories_excluded": len(excluded),
            "family_manifest_sha256": file_hash(output / "families.jsonl"),
            "generator_sha256": file_hash(Path(__file__)),
            "rejections": dict(rejections),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copyfile(output / "families.jsonl", directory / "families.jsonl")
    return validate(output)


def validate(output: Path):
    records = [json.loads(line) for line in (output / "families.jsonl").read_text().splitlines()]
    parents = {row["family_id"]: row for row in records}
    if len(parents) != len(records):
        raise ValueError("Duplicate parent family identifiers")
    excluded = old_hashes(Path("data/pilot-v3"))
    owners = {}
    for row in records:
        for digest in row["world_hashes"]:
            if digest in owners or digest in excluded:
                raise ValueError("A world overlaps another family or v1")
            owners[digest] = row["family_id"]
    total = 0
    for arm in ARMS:
        directory = output / arm
        manifest = json.loads((directory / "manifest.json").read_text())
        if file_hash(directory / "families.jsonl") != manifest["family_manifest_sha256"]:
            raise ValueError("Parent family manifest changed")
        for split, meta in manifest["splits"].items():
            path = directory / meta["file"]
            if file_hash(path) != meta["file_sha256"]:
                raise ValueError("Split checksum changed")
            rows = read_examples(path)
            families = {}
            for row in rows:
                family = parents[row.family_id]
                if row.theory_hash not in family["world_hashes"]:
                    raise ValueError("A row is outside its registered parent worlds")
                if row.theory_hash != canonical_hash(row.theory):
                    raise ValueError("Incorrect canonical hash")
                resolution = solve(row.theory)
                if resolution.answer(row.query) != (row.label, row.proof_depth):
                    raise ValueError("Incorrect oracle answer")
                if sorted(row.facts) != sorted(render_atom(a, {}) + "." for a in row.theory.facts):
                    raise ValueError("Rendered facts disagree")
                if sorted(row.rules) != sorted(
                    render_rule(r, row.renderer_id) for r in row.theory.rules
                ):
                    raise ValueError("Rendered rules disagree")
                if row.prompt != build_prompt(row.facts, row.rules, row.question):
                    raise ValueError("Prompt disagrees with rendering")
                if row.question != render_atom(row.query, {}) + ".":
                    raise ValueError("Query disagrees with rendering")
                expected_partition = {
                    "test_original": "test",
                    "test_counterfactual": "test",
                    "test_depth": "depth",
                    "test_wording": "wording",
                    "test_single_branch": "single_branch",
                }.get(split, split)
                if family["partition"] != expected_partition or row.split != split:
                    raise ValueError("A parent family crosses partitions")
                families.setdefault(row.family_id, []).append(row)
            for family in families.values():
                if split == "test_single_branch":
                    before, after = sorted(family, key=lambda r: r.example_id, reverse=True)
                    if (
                        len(family) != 2
                        or after.label != Label.UNKNOWN
                        or before.label == Label.UNKNOWN
                    ):
                        raise ValueError("Invalid single-branch pair")
                    if before.query != after.query:
                        raise ValueError("Single-branch query changed")
                elif {r.label for r in family} != set(Label) or len(family) != 3:
                    raise ValueError("Family is not balanced")
                if (
                    split.startswith("test_")
                    and split != "test_original"
                    and split != "test_single_branch"
                ):
                    if len({r.query for r in family}) != 1:
                        raise ValueError("Counterfactual test query changes across worlds")
                if arm == "treatment" and split == "train" and len({r.query for r in family}) != 1:
                    raise ValueError("Treatment training query changes")
                matched = len({r.query for r in family}) == 1 and len(family) == 3
                if matched:
                    by_label = {row.label: row for row in family}
                    before = by_label[Label.TRUE].theory
                    if any(row.theory.rules != before.rules for row in family):
                        raise ValueError("Counterfactual rules changed")
                    for label in (Label.FALSE, Label.UNKNOWN):
                        after = by_label[label].theory
                        removed = set(before.facts) - set(after.facts)
                        added = set(after.facts) - set(before.facts)
                        if len(removed) != 1 or len(added) > 1:
                            raise ValueError("Counterfactual changes more than one premise")
                        if added and added != {next(iter(removed)).opposite()}:
                            raise ValueError("Counterfactual does not flip the changed fact")
            if len(rows) != meta["questions"] or len(families) != meta["families"]:
                raise ValueError("Manifest count mismatch")
            total += len(rows)
    for split in ("development", "calibration", *TESTS):
        if file_hash(output / "control" / f"{split}.jsonl") != file_hash(
            output / "treatment" / f"{split}.jsonl"
        ):
            raise ValueError("Shared selection/calibration/test inputs differ between arms")
    return {
        "valid": True,
        "parents": len(parents),
        "unique_worlds": len(owners),
        "rows_across_arms": total,
    }
