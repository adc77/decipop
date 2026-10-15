import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path.cwd()
DEST = ROOT / "artifacts/premise-release"
NAMESPACE = "axondendriteplus"
GITHUB = "https://github.com/adc77/premise"
DATASET = f"{NAMESPACE}/premise-worlds"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing release file differs: {target}")
    else:
        os.link(source, target)


def prepare_dataset():
    target = DEST / "dataset"
    configs, counts, files = [], {}, {}
    for version in (7, 8):
        for rep in ("r1", "r2"):
            name = f"v{version}-{rep}"
            source = ROOT / f"runs/study-v{version}/{rep}/data"
            splits = []
            counts[name] = {}
            for path in sorted(source.glob("*.jsonl")):
                if path.stem == "families":
                    continue
                relative = Path("data") / name / path.name
                link(path, target / relative)
                counts[name][path.stem] = sum(1 for _ in path.open())
                splits.append(dict(split=path.stem, path=str(relative)))
                files[str(relative)] = digest(path)
            configs.append(dict(config_name=name, data_files=splits))
            for filename in ("families.jsonl", "manifest.json"):
                relative = Path("provenance") / name / filename
                link(source / filename, target / relative)
                files[str(relative)] = digest(source / filename)
    metadata = dict(
        license="apache-2.0",
        language=["en"],
        task_categories=["text-classification"],
        tags=["synthetic", "reasoning", "counterfactual", "symbolic", "premise"],
        configs=configs,
    )
    total = sum(sum(splits.values()) for splits in counts.values())
    rows = "\n".join(
        f"| {config} | {split} | {count:,} |"
        for config, splits in counts.items()
        for split, count in splits.items()
    )
    card = f"""# Premise Worlds

Frozen synthetic facts-and-rules data used in Premise's V7 and V8 comparisons.
Each row asks whether a statement is TRUE, FALSE, or UNKNOWN given an explicit world.

[Code and research article]({GITHUB}) · [Methods]({GITHUB}/blob/main/docs/V8_PROTOCOL.md)

## Load

```python
from datasets import load_dataset

data = load_dataset("{DATASET}", "v8-r2", split="test_paired")
row = data[0]
print(row["prompt"], row["label"])
```

Configs: `v7-r1`, `v7-r2`, `v8-r1`, `v8-r2`. Choose the matching config for a
model's reported results. `train_control` and `train_treatment` are separate
arms, not disjoint partitions. V8's two training files are identical by design.
Selection-fit, selection, calibration and development have distinct families.

## Semantics and schema

- TRUE: the query is derivable.
- FALSE: its explicit negation is derivable.
- UNKNOWN: neither is derivable; absence does not imply negation.

Rows retain exact frozen `example_id`, `family_id`, `split`, `theory`, `query`,
`label`, proof-depth fields, rendered `facts`, `rules`, `question`, and `prompt`.
The symbolic theory contains atoms with predicates, arguments and explicit polarity.
The prompt is the actual model input. A/B/C in the prompt map to TRUE/FALSE/UNKNOWN.
Family registries and original manifests are under `provenance/`; `sha256.json`
covers the exact published source bytes. No paid teacher model supplied labels.

## Provenance and partitioning

Created by Aman / adc77 using the included generator and symbolic oracle.
Names are symbolic tokens in generated worlds, not records about real people.
No customer, freight, personal-document or external benchmark data is included.
Contradictory worlds are rejected. All transformations stay with their parent family.
Within these studies, recorded prior canonical worlds are excluded and r2 excludes r1.
Shared templates, vocabulary and rule skeletons remain; this is not abstract-template
separation or a guarantee about the pretrained base's data.

V7 tests the curriculum intervention. V8 tests paired CE versus an added fixed
label-aligned Jensen–Shannon term, from the V7 r1 selected treatment. V7 and V8
have different endpoints and fresh data. Do not merge configurations and claim a
single uncontaminated train/test split: later studies adapt to earlier findings.

## Evaluation units

Question accuracy is per row. Complete worlds require every entity correct.
V8 complete paired worlds require all queries correct before and after the assigned
edit. Complete transformation families require every included variant and query.
Rows share worlds, transformations, label-rotation blocks and intentional training
repetitions. The {total:,} serialized rows below are not independent examples or
unique worlds. UNKNOWN edits can be identity controls; separate known-target edits.

## Counts

| Config | Split | Rows |
| --- | --- | ---: |
{rows}

## Uses and limitations

For controlled synthetic reasoning, evidence sensitivity, entity tracking,
checkpoint selection and calibration research. Training/evaluation generators
are narrow. These data do not establish business decision quality, unrestricted
reasoning or human-level logical competence. The test answers are released:
do not tune on them and call the same split a fresh confirmatory evaluation.

## License

Project-generated data are released under Apache 2.0. Code, data, and model-base
attribution are tracked separately. Related work includes RuleTaker and ProofWriter;
their datasets are not redistributed in this release.
"""
    (target / "README.md").write_text(
        "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + card
    )
    shutil.copyfile(ROOT / "LICENSE", target / "LICENSE")
    save(target / "sha256.json", files)
    save(target / "row-counts.json", counts)
    return dict(repo_id=DATASET, directory=str(target), rows=total, files=files)


def prepare_model(version, rep, arm):
    source = ROOT / f"artifacts/v{version}-releases/{rep}/{arm}"
    old_manifest = json.loads((source / "artifact-manifest.json").read_text())
    for name, expected in old_manifest["files"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Historical bundle changed: {source}/{name}")
    suffix = (
        ("control" if arm == "control" else "multi-entity")
        if version == 7
        else ("paired-ce" if arm == "control" else "consistency")
    )
    name = f"premise-v{version}-{rep}-{suffix}"
    target = DEST / "models" / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in old_manifest["files"]:
        if filename != "README.md":
            link(source / filename, target / filename)
    old_release = json.loads(
        (ROOT / f"runs/study-v{version}/{rep}/huggingface-release.json").read_text()
    )[arm]
    selection = json.loads((source / "selection-freeze.json").read_text())
    step = selection["candidates"][selection["choices"][arm]["calibrated_nll"]]["step"]
    evaluation = json.loads((source / "evaluation.json").read_text())["splits"]
    rows = "\n".join(
        f"| {split} | {value['raw']['accuracy']:.2%} | "
        f"{value['all_questions_in_family_correct']:.2%} |"
        for split, value in evaluation.items()
    )
    seed = {7: {"r1": 731, "r2": 947}, 8: {"r1": 1601, "r2": 1801}}[version][rep]
    description = (
        "Old-curriculum control"
        if version == 7 and arm == "control"
        else "Multi-entity curriculum treatment"
        if version == 7
        else "Cross-entropy on paired worlds"
        if arm == "control"
        else "Paired cross-entropy plus 0.5 times label-aligned Jensen–Shannon consistency"
    )
    conclusion = (
        "V7's treatment-minus-control complete-world gains replicate in both fresh runs. "
        "This selected checkpoint is one arm, not a separate independent replication."
        if version == 7
        else "V8's added consistency loss did not pass the declared replicated advantage "
        "over paired CE: the first interval includes zero; the second is positive. "
        "This is mixed evidence, not equivalence."
    )
    header = dict(
        license="apache-2.0",
        language=["en"],
        base_model="HuggingFaceTB/SmolLM2-360M",
        datasets=[DATASET],
        tags=["premise", "research", "text-classification", "custom-code", "reasoning"],
    )
    card = f"""# Premise · V{version} {rep} · {suffix}

{description}. A full-fine-tuned SmolLM2-360M backbone with a linear
three-class decision head over the last non-padding hidden state.

[Repository]({GITHUB}) · [Dataset](https://huggingface.co/datasets/{DATASET}) ·
[Results]({GITHUB}/blob/main/reports/V{version}_RESULTS.md)

## Decisions

Given facts, universally quantified rules and a statement, predict TRUE (derivable),
FALSE (explicit negation derivable), or UNKNOWN (neither). UNKNOWN is a semantic
class, not a confidence-based refusal. No chain-of-thought or proof is generated.

## Run on CPU

Use Python 3.12 and the repository's locked environment:

```sh
git clone {GITHUB}.git
cd premise
uv sync --extra train --extra analysis --extra baseline
uv run python examples/predict.py --model {NAMESPACE}/{name}
```

The example downloads this custom inference bundle, checks its SHA-256 manifest,
and predicts the exact held-out prompts in `examples/prompts.json`. Use
`--revision <commit>` to pin a release. The standalone bundle also supports
`python infer.py --input prompts.json` after installing `requirements.txt`.
Inputs must match `prompt-template.txt`; the maximum length is 512 tokens and
overlong inputs fail rather than truncate. This is not a stock chat model or a
Transformers `pipeline` checkpoint; use the bundled loader. Model weights are FP32.

## Training and selection

- Study: V{version}, repetition {rep}, seed {seed}; dataset config `v{version}-{rep}`.
- Selected update: {step}; all arms completed 1,134 updates and 36,288 presentations.
- AdamW: backbone 2e-5, head 1e-4, weight decay .01, clip 1, warmup 5%.
- Effective batch 32, microbatch 8; training and aggregate inference use BF16 autocast.
- Candidates: 0, 250, 600, 900, 1,134. Temperature-fit and selection families are
  distinct; choose minimum calibrated selection NLL, then fit a new temperature
  on separate calibration families before opening final tests.
- V7 starts from V4's mixed head; V8 starts from V7 r1's selected treatment.
  V7's curriculum comparison changes token exposure; V8 matches tokens and pairs.

The pretrained base revision is `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.
`protocol.json`, `selection-freeze.json`, `calibration.json`, and `evaluation.json`
retain the exact original records. `public-release.json` links the historical
artifact to this public packaging; weights, tokenizer and inference code are unchanged.

## Evaluation

{conclusion}

| Fresh test | Question accuracy | Complete families |
| --- | ---: | ---: |
{rows}

Complete families require every included question correct. For `test_multi` this
means all people; for `test_paired` all people before and after the assigned edit;
for transformations every included variant and query. These endpoints differ.
Intervals in the report describe world sampling conditional on trained models,
not a population of training seeds or a joint confidence guarantee.

## Limitations

Synthetic, narrow rule distributions; shared templates and vocabulary remain.
Missing premises, distractors and wording changes still cause failures, sometimes
confident ones. Global temperature scaling does not guarantee transfer calibration.
The weights are research baselines, not validated access-control, legal, financial,
medical, freight or general business decision systems. Symbolic names are generated
tokens. The dataset contains no real records about those people.

## Verification and attribution

Historical private release passed a fresh pinned-download CPU check on eight FP32
reference prompts. This public packaging has its own manifest and release receipt;
aggregate published scores are from the original BF16 runs, not new CPU test scores.
Use `verification.json` for reference logits, decisions and numerical tolerances.

Apache 2.0; retain LICENSE and NOTICE. The pretrained model, tokenizer, dependencies
and project-authored code retain their attribution. No architecture novelty or
performance ranking against other model families is claimed.
"""
    (target / "README.md").write_text(
        "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\n\n" + card
    )
    provenance = dict(
        source_repo=old_release["repo_id"],
        source_commit=old_release["commit"],
        original_manifest_sha256=digest(source / "artifact-manifest.json"),
        weights_sha256=old_manifest["files"]["model.safetensors"],
        weights_tokenizer_and_inference_code_unchanged=True,
        github=GITHUB,
        dataset=DATASET,
        study=f"v{version}",
        repetition=rep,
        arm=arm,
        seed=seed,
        selected_update=step,
    )
    save(target / "public-release.json", provenance)
    manifest = old_manifest.copy()
    manifest["files"] = {k: digest(target / k) for k in old_manifest["files"]}
    manifest["files"]["public-release.json"] = digest(target / "public-release.json")
    save(target / "artifact-manifest.json", manifest)
    return dict(repo_id=f"{NAMESPACE}/{name}", directory=str(target), **provenance)


def main():
    dataset = prepare_dataset()
    models = [
        prepare_model(version, rep, arm)
        for version in (7, 8)
        for rep in ("r1", "r2")
        for arm in ("control", "treatment")
    ]
    save(DEST / "prepared.json", dict(dataset=dataset, models=models))
    print(json.dumps(dict(dataset_rows=dataset["rows"], model_bundles=len(models))))


if __name__ == "__main__":
    main()
