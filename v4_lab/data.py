import json
import random
from collections import Counter
from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import _candidate
from decision_lab.data.render import build_prompt, render_atom, render_rule
from decision_lab.data.schema import Label, Theory
from decision_lab.data.solve import InconsistentTheoryError, solve
from decision_lab.data.split import canonical_hash
from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v2.data import example, parent_world, replace_fact
from decision_lab.v3.data import prior_worlds

TESTS = ("test_original", "test_counterfactual", "test_depth", "test_wording", "test_single_branch")


def exclusions(include_diagnostics=False):
    hashes, files = prior_worlds()
    paths = [Path("runs/study-v3/data/families.jsonl")]
    if include_diagnostics:
        paths.append(Path("runs/study-v4/diagnostic/families.jsonl"))
    for path in paths:
        for line in path.read_text().splitlines():
            hashes.update(json.loads(line)["world_hashes"])
        files[str(path)] = file_hash(path)
    return hashes, files


def renamed(theory, query):
    names = theory.entities
    mapping = dict(zip(names, names[1:] + names[:1], strict=True))

    def atom(source):
        return source.model_copy(
            update={
                "arguments": tuple(a if a.startswith("?") else mapping[a] for a in source.arguments)
            }
        )

    world = Theory(
        entities=names,
        facts=tuple(atom(f) for f in theory.facts),
        rules=tuple(
            r.model_copy(
                update={
                    "conditions": tuple(atom(c) for c in r.conditions),
                    "conclusion": atom(r.conclusion),
                }
            )
            for r in theory.rules
        ),
    )
    return world, atom(query)


def triples(theory, query, unknown, critical, selector, family, split, motif, template, seed):
    worlds = (
        theory,
        replace_fact(theory, selector, selector.opposite()),
        replace_fact(theory, critical, critical.opposite()),
    )
    if [solve(w).answer(query)[0] for w in worlds] != list(Label):
        raise ValueError("Matched counterfactual oracle mismatch")
    matched = [
        example(w, query, family, split, role, motif, template, seed)
        for w, role in zip(worlds, ("true", "false", "unknown"), strict=True)
    ]
    original = [
        example(theory, q, family, split, role, motif, template, seed)
        for q, role in zip(
            (query, query.opposite(), unknown), ("true", "false", "unknown"), strict=True
        )
    ]
    return matched, original, worlds


def generate(directory, spec, excluded, exclusion_files, diagnostic=False):
    if directory.exists():
        raise FileExistsError(directory)
    config = load_dataset_config(Path("configs/dataset.yaml"))
    rng = random.Random(spec["dataset_seed"] + (0 if diagnostic else 1))
    counts = (
        {"diagnostic": spec["diagnostic_families"]}
        if diagnostic
        else {
            "train": spec["train_families"],
            "selection_fit": spec["selection_fit_families"],
            "selection": spec["selection_families"],
            "calibration": spec["calibration_families"],
            "test": spec["test_families"],
            "depth": spec["transfer_families"],
            "wording": spec["transfer_families"],
            "single_branch": spec["transfer_families"],
        }
    )
    registry, owners, splits, rejects = [], {}, {}, Counter()
    for partition, count in counts.items():
        accepted = attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * spec["candidate_attempt_multiplier"]:
                raise RuntimeError(f"Generation budget exhausted: {partition}")
            depth = rng.choice([4, 5] if partition == "depth" else [1, 2, 3])
            motif = rng.choice(["chain", "relational"])
            template = (
                "paraphrase" if partition == "wording" else rng.choice(config.train_templates)
            )
            family = f"v4-{partition}-{accepted:06d}"
            seed = rng.randrange(2**32)
            try:
                theory, query, unknown, critical, selector = parent_world(
                    config, depth, motif, partition == "wording", rng
                )
                matched, original, worlds = triples(
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
                if diagnostic:
                    unknown = unknown.model_copy(update={"negative": query.negative})
                    negative_world = worlds[1]
                    noisy = Theory(
                        entities=theory.entities,
                        facts=(*theory.facts, unknown.opposite()),
                        rules=theory.rules,
                    )
                    rename_world, rename_query = renamed(theory, query)
                    _, rename_unknown = renamed(theory, unknown)
                    variants = [
                        (theory, query, "base_true"),
                        (negative_world, query, "base_false"),
                        (theory, unknown, "query_unknown_true"),
                        (negative_world, unknown, "query_unknown_false"),
                        (worlds[2], query, "premise_unknown"),
                        (rename_world, rename_query, "renamed_true"),
                        (rename_world, rename_unknown, "renamed_unknown"),
                        (noisy, query, "distractor_true"),
                    ]
                    rows = [
                        example(w, q, family, partition, role, motif, template, seed)
                        for w, q, role in variants
                    ]
                    if [r.label for r in rows] != [
                        Label.TRUE,
                        Label.FALSE,
                        Label.UNKNOWN,
                        Label.UNKNOWN,
                        Label.UNKNOWN,
                        Label.TRUE,
                        Label.UNKNOWN,
                        Label.TRUE,
                    ]:
                        raise ValueError("Diagnostic intervention oracle mismatch")
                    worlds = [w for w, _, _ in variants]
                    bundles = {"diagnostic": rows}
                elif partition == "single_branch":
                    setting = config.splits["train"].model_copy(update={"depths": [depth]})
                    theory, queries, motif = _candidate(config, setting, rng)
                    query = queries[rng.randrange(2)]
                    choices = []
                    for fact in theory.facts:
                        changed = replace_fact(theory, fact, fact.opposite())
                        if solve(changed).answer(query)[0] == Label.UNKNOWN:
                            choices.append((fact, changed))
                    if not choices:
                        raise ValueError("No single-branch critical flip")
                    critical, changed = rng.choice(choices)
                    worlds = [theory, changed]
                    bundles = {
                        "test_single_branch": [
                            example(
                                w, query, family, "test_single_branch", role, motif, template, seed
                            )
                            for w, role in zip(worlds, ("before", "after"), strict=True)
                        ]
                    }
                elif partition == "train":
                    bundles = {
                        "train_control": matched,
                        "train_treatment": original if accepted % 2 == 0 else matched,
                    }
                elif partition in ("selection_fit", "selection", "calibration"):
                    bundles = {partition: original if accepted % 2 == 0 else matched}
                elif partition == "test":
                    bundles = {"test_original": original, "test_counterfactual": matched}
                else:
                    bundles = {f"test_{partition}": matched}
                hashes = {canonical_hash(w) for w in worlds}
                if hashes & excluded or hashes & owners.keys():
                    raise ValueError("Canonical world overlap")
                if (
                    max(len(r.prompt) for rs in bundles.values() for r in rs)
                    > spec["max_prompt_characters"]
                ):
                    raise ValueError("Prompt character limit")
            except (InconsistentTheoryError, ValueError) as error:
                rejects[str(error)] += 1
                continue
            owners.update(dict.fromkeys(hashes, family))
            registry.append(
                {
                    "family_id": family,
                    "partition": partition,
                    "world_hashes": sorted(hashes),
                    "critical_fact": critical.model_dump(),
                    "treatment_component": "original" if accepted % 2 == 0 else "counterfactual",
                }
            )
            for split, rows in bundles.items():
                splits.setdefault(split, []).extend(
                    r.model_copy(
                        update={
                            "split": split,
                            "example_id": f"{family}-{split}-{r.example_id.rsplit('-', 1)[1]}",
                        }
                    )
                    for r in rows
                )
            accepted += 1
        print(json.dumps({"generated_partition": partition, "families": count}), flush=True)
    directory.mkdir(parents=True)
    registry_path = directory / "families.jsonl"
    registry_path.write_text("".join(json.dumps(r) + "\n" for r in registry))
    files = {}
    for split, rows in splits.items():
        shuffle_key = len("train") if split.startswith("train_") else len(split)
        random.Random(spec["dataset_seed"] + shuffle_key).shuffle(rows)
        path = directory / f"{split}.jsonl"
        path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
        files[split] = {
            "sha256": file_hash(path),
            "questions": len(rows),
            "families": len({r.family_id for r in rows}),
            "labels": dict(Counter(r.label.value for r in rows)),
        }
    atomic_json(
        directory / "manifest.json",
        {
            "splits": files,
            "families_sha256": file_hash(registry_path),
            "excluded_files": exclusion_files,
            "excluded_worlds": len(excluded),
            "fresh_worlds": len(owners),
            "rejections": dict(rejects),
        },
    )
    return validate(directory, excluded)


def validate(directory, excluded):
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(directory / "families.jsonl") != manifest["families_sha256"]:
        raise ValueError("Family registry changed")
    registry = [
        json.loads(line) for line in (directory / "families.jsonl").read_text().splitlines()
    ]
    parents = {r["family_id"]: r for r in registry}
    owners = {}
    if len(parents) != len(registry):
        raise ValueError("Duplicate parent family")
    for record in registry:
        for digest in record["world_hashes"]:
            if digest in owners or digest in excluded:
                raise ValueError("Family or prior-world overlap")
            owners[digest] = record["family_id"]
    total = 0
    for split, record in manifest["splits"].items():
        path = directory / f"{split}.jsonl"
        if file_hash(path) != record["sha256"]:
            raise ValueError("Dataset checksum changed")
        rows = read_examples(path)
        groups = {}
        for row in rows:
            partition = (
                "train"
                if split.startswith("train_")
                else "test"
                if split in ("test_original", "test_counterfactual")
                else split.removeprefix("test_")
            )
            if row.split != split or parents[row.family_id]["partition"] != partition:
                raise ValueError("Family crosses partitions")
            if (
                canonical_hash(row.theory) != row.theory_hash
                or owners[row.theory_hash] != row.family_id
            ):
                raise ValueError("World is outside its declared family")
            if solve(row.theory).answer(row.query) != (row.label, row.proof_depth):
                raise ValueError("Oracle mismatch")
            if sorted(row.facts) != sorted(render_atom(a, {}) + "." for a in row.theory.facts):
                raise ValueError("Fact rendering mismatch")
            if sorted(row.rules) != sorted(
                render_rule(r, row.renderer_id) for r in row.theory.rules
            ):
                raise ValueError("Rule rendering mismatch")
            if row.question != render_atom(row.query, {}) + "." or row.prompt != build_prompt(
                row.facts, row.rules, row.question
            ):
                raise ValueError("Query or prompt rendering mismatch")
            groups.setdefault(row.family_id, []).append(row)
        if split not in ("diagnostic", "test_single_branch"):
            if any(Counter(r.label for r in rs) != Counter(Label) for rs in groups.values()):
                raise ValueError("Unbalanced parent labels")
        if len(rows) != record["questions"] or len(groups) != record["families"]:
            raise ValueError("Manifest count mismatch")
        total += len(rows)
    return {
        "valid": True,
        "families": len(registry),
        "questions": total,
        "unique_worlds": len(owners),
        "prior_worlds_excluded": len(excluded),
    }
