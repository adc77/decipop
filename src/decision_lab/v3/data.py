import json
import random
from collections import Counter
from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import _candidate
from decision_lab.data.render import build_prompt, render_atom, render_rule
from decision_lab.data.schema import Label
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash
from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v2.data import example, parent_world, replace_fact


def prior_worlds():
    paths = sorted(Path("data/pilot-v3").glob("*.jsonl"))
    paths += sorted(Path("runs/study-v1/test-inputs").glob("*.jsonl"))
    paths += sorted(Path("data/v2/control").glob("*.jsonl"))
    paths += sorted(Path("data/v2/treatment").glob("*.jsonl"))
    paths.append(Path("data/v2/families.jsonl"))
    hashes = set()
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "world_hashes" in row:
                hashes.update(row["world_hashes"])
            else:
                source = row["example"] if "example" in row else row
                hashes.add(source["theory_hash"])
    return hashes, {str(path): file_hash(path) for path in paths}


def generate(directory, spec, excluded, exclusion_files, config_path=Path("configs/dataset.yaml")):
    if directory.exists():
        raise FileExistsError(directory)
    config = load_dataset_config(config_path)
    rng = random.Random(spec["dataset_seed"])
    counts = {
        "selection_fit": spec["selection_fit_families"],
        "selection": spec["selection_families"],
        "calibration": spec["calibration_families"],
        "test": spec["test_families"],
        "depth": spec["transfer_families"],
        "wording": spec["transfer_families"],
        "single_branch": spec["transfer_families"],
    }
    owners = {}
    families = []
    splits = {}
    rejections = Counter()
    for partition, count in counts.items():
        accepted = attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * spec["candidate_attempt_multiplier"]:
                raise RuntimeError(f"Candidate budget exhausted: {partition}")
            depth = rng.choice([4, 5] if partition == "depth" else [1, 2, 3])
            motif = rng.choice(["chain", "relational"])
            template = (
                "paraphrase" if partition == "wording" else rng.choice(config.train_templates)
            )
            family = f"v3-{partition}-{accepted:06d}"
            render_seed = rng.randrange(2**32)
            try:
                if partition == "single_branch":
                    split = "test_single_branch"
                    setting = config.splits["train"].model_copy(update={"depths": [depth]})
                    theory, queries, motif = _candidate(config, setting, rng)
                    query = queries[rng.randrange(2)]
                    choices = []
                    for fact in theory.facts:
                        changed = replace_fact(theory, fact, fact.opposite())
                        try:
                            if solve(changed).answer(query)[0] == Label.UNKNOWN:
                                choices.append((fact, changed))
                        except InconsistentTheoryError:
                            continue
                    if not choices:
                        raise ValueError("No single-branch critical flip")
                    critical, changed = rng.choice(choices)
                    variants = [theory, changed]
                    rows = [
                        example(world, query, family, split, role, motif, template, render_seed)
                        for world, role in zip(variants, ("before", "after"), strict=True)
                    ]
                    bundles = {split: rows}
                else:
                    theory, query, unknown, critical, selector = parent_world(
                        config, depth, motif, partition == "wording", rng
                    )
                    variants = [
                        theory,
                        replace_fact(theory, selector, selector.opposite()),
                        replace_fact(theory, critical, critical.opposite()),
                    ]
                    if [solve(world).answer(query)[0] for world in variants] != list(Label):
                        raise ValueError("Incorrect matched oracle labels")
                    split = {
                        "test": "test_counterfactual",
                        "depth": "test_depth",
                        "wording": "test_wording",
                    }.get(partition, partition)
                    matched = [
                        example(world, query, family, split, role, motif, template, render_seed)
                        for world, role in zip(variants, ("true", "false", "unknown"), strict=True)
                    ]
                    original_split = "test_original" if partition == "test" else split
                    original = [
                        example(
                            theory, q, family, original_split, role, motif, template, render_seed
                        )
                        for q, role in zip(
                            (query, query.opposite(), unknown),
                            ("true", "false", "unknown"),
                            strict=True,
                        )
                    ]
                    if partition == "test":
                        bundles = {original_split: original, split: matched}
                    elif partition in {"selection_fit", "selection", "calibration"}:
                        bundles = {split: original if accepted % 2 == 0 else matched}
                    else:
                        bundles = {split: matched}
                hashes = {canonical_hash(world) for world in variants}
                if hashes & excluded or hashes & owners.keys():
                    raise ValueError("Canonical world overlap")
                if (
                    max(len(row.prompt) for rows in bundles.values() for row in rows)
                    > spec["max_prompt_characters"]
                ):
                    raise ValueError("Prompt character limit")
            except (InconsistentTheoryError, ValueError) as error:
                rejections[str(error)] += 1
                continue
            owners.update(dict.fromkeys(hashes, family))
            families.append(
                {
                    "family_id": family,
                    "partition": partition,
                    "world_hashes": sorted(hashes),
                    "critical_fact": critical.model_dump(),
                }
            )
            for split, rows in bundles.items():
                splits.setdefault(split, []).extend(rows)
            accepted += 1
        print(json.dumps({"generated_partition": partition, "families": count}), flush=True)
    directory.mkdir(parents=True)
    (directory / "families.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in families)
    )
    inventory = {}
    for split, rows in splits.items():
        random.Random(spec["dataset_seed"] + len(split)).shuffle(rows)
        path = directory / f"{split}.jsonl"
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
        inventory[split] = {
            "file_sha256": file_hash(path),
            "questions": len(rows),
            "families": len({row.family_id for row in rows}),
            "labels": dict(Counter(row.label.value for row in rows)),
        }
    manifest = {
        "settings": spec,
        "splits": inventory,
        "families_sha256": file_hash(directory / "families.jsonl"),
        "exclusion_files": exclusion_files,
        "prior_unique_worlds": len(excluded),
        "fresh_unique_worlds": len(owners),
        "rejections": dict(rejections),
    }
    atomic_json(directory / "manifest.json", manifest)
    return validate(directory, excluded)


def validate(directory, excluded):
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(directory / "families.jsonl") != manifest["families_sha256"]:
        raise ValueError("Family registry changed")
    records = [json.loads(line) for line in (directory / "families.jsonl").read_text().splitlines()]
    parents = {record["family_id"]: record for record in records}
    if len(parents) != len(records):
        raise ValueError("Duplicate parent identifiers")
    owners = {}
    for record in records:
        for digest in record["world_hashes"]:
            if digest in excluded or digest in owners:
                raise ValueError("Family or prior-world overlap")
            owners[digest] = record["family_id"]
    total = 0
    for split, metadata in manifest["splits"].items():
        path = directory / f"{split}.jsonl"
        if file_hash(path) != metadata["file_sha256"]:
            raise ValueError("Dataset changed")
        rows = read_examples(path)
        families = {}
        for row in rows:
            parent = parents[row.family_id]
            partition = {
                "test_original": "test",
                "test_counterfactual": "test",
                "test_depth": "depth",
                "test_wording": "wording",
                "test_single_branch": "single_branch",
            }.get(split, split)
            if row.split != split or parent["partition"] != partition:
                raise ValueError("Family crosses partitions")
            if (
                canonical_hash(row.theory) != row.theory_hash
                or owners[row.theory_hash] != row.family_id
            ):
                raise ValueError("World is outside its parent")
            if solve(row.theory).answer(row.query) != (row.label, row.proof_depth):
                raise ValueError("Oracle mismatch")
            if sorted(row.facts) != sorted(
                render_atom(atom, {}) + "." for atom in row.theory.facts
            ):
                raise ValueError("Fact rendering mismatch")
            if sorted(row.rules) != sorted(
                render_rule(rule, row.renderer_id) for rule in row.theory.rules
            ):
                raise ValueError("Rule rendering mismatch")
            if row.question != render_atom(row.query, {}) + ".":
                raise ValueError("Query rendering mismatch")
            if row.prompt != build_prompt(row.facts, row.rules, row.question):
                raise ValueError("Prompt rendering mismatch")
            families.setdefault(row.family_id, []).append(row)
        for family in families.values():
            if split == "test_single_branch":
                before = next(row for row in family if row.example_id.endswith("-before"))
                after = next(row for row in family if row.example_id.endswith("-after"))
                if (
                    len(family) != 2
                    or before.label == Label.UNKNOWN
                    or after.label != Label.UNKNOWN
                ):
                    raise ValueError("Invalid single-branch pair")
                if before.query != after.query:
                    raise ValueError("Changed single-branch query")
                worlds = (before, after)
            else:
                if len(family) != 3 or {row.label for row in family} != set(Label):
                    raise ValueError("Unbalanced three-world family")
                worlds = tuple(sorted(family, key=lambda row: list(Label).index(row.label)))
            if len({row.query for row in worlds}) == 1:
                for changed in worlds[1:]:
                    source = worlds[0].theory
                    removed = set(source.facts) - set(changed.theory.facts)
                    added = set(changed.theory.facts) - set(source.facts)
                    if source.rules != changed.theory.rules or len(removed) != 1:
                        raise ValueError("Invalid critical intervention")
                    if added != {next(iter(removed)).opposite()}:
                        raise ValueError("Intervention must flip one fact")
        if len(rows) != metadata["questions"] or len(families) != metadata["families"]:
            raise ValueError("Dataset counts changed")
        total += len(rows)
    return {"valid": True, "families": len(parents), "unique_worlds": len(owners), "rows": total}
