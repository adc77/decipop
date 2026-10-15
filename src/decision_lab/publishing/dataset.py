import argparse
import hashlib
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

import yaml

from decision_lab.data.schema import Example, Theory
from decision_lab.data.solve import solve
from decision_lab.data.split import canonical_hash
from decision_lab.studies.v5.data import independent_closure, reference_label


def rename_symbols(value, source, replacement):
    pattern = re.compile(rf"\b{re.escape(source)}\b")
    if isinstance(value, str):
        return pattern.sub(replacement, value)
    if isinstance(value, list):
        return [rename_symbols(item, source, replacement) for item in value]
    if isinstance(value, dict):
        return {key: rename_symbols(item, source, replacement) for key, item in value.items()}
    return value


def normalize_row(row, source, replacement):
    if source == replacement or replacement in row["theory"]["entities"]:
        raise ValueError("Replacement must be new and distinct")
    return rename_symbols(row, source, replacement)


@lru_cache(maxsize=4096)
def resolutions(serialized):
    theory = Theory.model_validate_json(serialized)
    return solve(theory), independent_closure(theory)


def validate_row(row):
    example = Example.model_validate_json(json.dumps(row))
    resolution, closure = resolutions(example.theory.model_dump_json())
    label, depth = resolution.answer(example.query)
    if (label, depth) != (example.label, example.proof_depth):
        raise ValueError(f"Label or proof changed: {example.example_id}")
    if reference_label(closure, example.query) != label:
        raise ValueError(f"Independent oracle disagrees: {example.example_id}")
    return example


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source, destination, source_name, replacement):
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    card = (source / "README.md").read_text().split("---", 2)
    metadata = yaml.safe_load(card[1])
    metadata["tags"] = ["synthetic", "reasoning", "counterfactual", "symbolic", "decipop"]
    inventory, counts, checks = {}, {}, []
    for path in sorted(source.rglob("*.jsonl")):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open() as inputs, target.open("x") as outputs:
            for line in inputs:
                row = json.loads(line)
                if "theory" in row:
                    clean = normalize_row(row, source_name, replacement)
                    before, after = validate_row(row), validate_row(clean)
                    if count == 0:
                        if canonical_hash(before.theory) != canonical_hash(after.theory):
                            raise ValueError("Canonical world changed")
                        checks.append(str(relative))
                else:
                    clean = rename_symbols(row, source_name, replacement)
                encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
                if source_name in encoded:
                    raise ValueError(f"Source name remains in {relative}")
                outputs.write(encoded + "\n")
                count += 1
        inventory[str(relative)] = {"sha256": digest(target), "source_sha256": digest(path)}
        if relative.parts[0] == "data":
            counts.setdefault(relative.parts[1], {})[path.stem] = count
        print(json.dumps({"file": str(relative), "rows": count}), flush=True)
    total = sum(sum(splits.values()) for splits in counts.values())
    provenance = {
        "transformation": f"One entity symbol replaced consistently with {replacement}.",
        "original_source_revision": "v0.1.0",
        "labels_and_proof_depths": "All rows checked with production and independent oracles.",
        "canonical_hash_samples": checks,
        "serialized_rows": total,
        "model_scores": "Scores use original inputs; normalized inputs not re-evaluated.",
    }
    for filename, value in (
        ("sha256.json", inventory),
        ("row-counts.json", counts),
        ("normalization.json", provenance),
    ):
        (destination / filename).write_text(json.dumps(value, indent=2) + "\n")
    shutil.copyfile(source / "LICENSE", destination / "LICENSE")
    body = """# decipop worlds

Synthetic facts and rules for testing whether a small model follows changing evidence.
Each question has an exact TRUE, FALSE or UNKNOWN answer from a symbolic oracle.

[Article](https://adc77.github.io/decipop/) · [Code](https://github.com/adc77/decipop) · [Methods](https://github.com/adc77/decipop/blob/main/docs/V8_PROTOCOL.md)

## Load

```python
from datasets import load_dataset
rows = load_dataset("axondendriteplus/decipop-worlds", "v8-r2", split="test_paired")
print(rows[0]["prompt"], rows[0]["label"])
```

Choose `v7-r1`, `v7-r2`, `v8-r1` or `v8-r2`. The release contains **190,688 serialized
rows**, including repeated training questions and related transformations. They are
not independent worlds. See `row-counts.json` for every split count.

## What changed in this release

This is a name-normalized derivative of the frozen V7/V8 research data. One entity
symbol is replaced with `EntityA` throughout symbolic theories, queries and rendered
text. Labels, proof depths, IDs, split membership and family relationships are
preserved. Every question was checked with two symbolic oracles; sampled canonical
world identities were also checked. `normalization.json` records the checks, and
`sha256.json` records both source and derived file hashes. Original manifests are
not presented as checksums for these changed files.

**The published model scores use the original input text. They are not scores on
this renamed release.** Renaming preserves logical meaning but can change model
tokenization and predictions. The released checkpoints have not been re-evaluated
on the normalized inputs. Historical source and results remain at the code's
`v0.1.0` tag; the original dataset is retained privately for exact reproduction.

## Task and schema

TRUE: the queried statement follows. FALSE: its explicit negation follows.
UNKNOWN: neither follows. Missing evidence is not negative evidence. UNKNOWN is
not a low-confidence refusal. Contradictory worlds are rejected.

Rows contain `example_id`, `family_id`, `split`, `theory`, `query`, `label`,
`proof_depth`, `theory_depth`, canonical hashes, `motif`, `renderer_id`, rendered
`facts`, `rules`, `question` and `prompt`. The prompt's A/B/C choices map to
TRUE/FALSE/UNKNOWN. Names are arbitrary generated symbols. Family registries live
under `provenance/`. No teacher-model labels or customer data are included.

## Splits and overlap

`train_control` and `train_treatment` are experimental arms, not disjoint splits.
V8's training files are identical by design. Development, calibration, selection-fit
and selection use distinct families within a config. Related queries and edits
share a family and must stay together. Repetitions exclude prior canonical worlds,
but templates, vocabulary and rule skeletons are shared. Later studies respond to
earlier findings: do not merge configs into a single uncontaminated benchmark.

Question accuracy counts individual rows. Complete-world accuracy requires every
entity correct. Complete paired worlds additionally require every query correct
after an assigned edit. Complete transformation families require all included
variants. UNKNOWN edits can be identity controls; inspect known-target edits
separately. The test labels are public; tuning on them forfeits a fresh test claim.

## Intended use and limits

For controlled research on entity tracking, evidence sensitivity, calibration and
checkpoint selection. This narrow synthetic distribution does not establish
reliable business decisions, unrestricted reasoning or real-world safety.

Project-generated data: Apache 2.0. Related work includes RuleTaker and ProofWriter;
their data are not redistributed here. Model-base attribution is separate.
"""
    (destination / "README.md").write_text(
        "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + body
    )
    for path in destination.rglob("*"):
        if path.is_file() and source_name in path.read_text():
            raise ValueError(f"Source name remains in {path}")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--replacement", default="EntityA")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.source_name, args.replacement)))


if __name__ == "__main__":
    main()
