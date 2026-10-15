---
license: apache-2.0
language:
- en
task_categories:
- text-classification
tags:
- synthetic
- reasoning
- counterfactual
- symbolic
- premise
configs:
- config_name: v7-r1
  data_files:
  - split: calibration
    path: data/v7-r1/calibration.jsonl
  - split: development
    path: data/v7-r1/development.jsonl
  - split: selection
    path: data/v7-r1/selection.jsonl
  - split: selection_fit
    path: data/v7-r1/selection_fit.jsonl
  - split: test_counterfactual
    path: data/v7-r1/test_counterfactual.jsonl
  - split: test_depth
    path: data/v7-r1/test_depth.jsonl
  - split: test_multi
    path: data/v7-r1/test_multi.jsonl
  - split: test_multi_depth
    path: data/v7-r1/test_multi_depth.jsonl
  - split: test_multi_signed_chain
    path: data/v7-r1/test_multi_signed_chain.jsonl
  - split: test_multi_transformations
    path: data/v7-r1/test_multi_transformations.jsonl
  - split: test_multi_wording
    path: data/v7-r1/test_multi_wording.jsonl
  - split: test_original
    path: data/v7-r1/test_original.jsonl
  - split: test_single_branch
    path: data/v7-r1/test_single_branch.jsonl
  - split: test_wording
    path: data/v7-r1/test_wording.jsonl
  - split: train_control
    path: data/v7-r1/train_control.jsonl
  - split: train_treatment
    path: data/v7-r1/train_treatment.jsonl
- config_name: v7-r2
  data_files:
  - split: calibration
    path: data/v7-r2/calibration.jsonl
  - split: development
    path: data/v7-r2/development.jsonl
  - split: selection
    path: data/v7-r2/selection.jsonl
  - split: selection_fit
    path: data/v7-r2/selection_fit.jsonl
  - split: test_counterfactual
    path: data/v7-r2/test_counterfactual.jsonl
  - split: test_depth
    path: data/v7-r2/test_depth.jsonl
  - split: test_multi
    path: data/v7-r2/test_multi.jsonl
  - split: test_multi_depth
    path: data/v7-r2/test_multi_depth.jsonl
  - split: test_multi_signed_chain
    path: data/v7-r2/test_multi_signed_chain.jsonl
  - split: test_multi_transformations
    path: data/v7-r2/test_multi_transformations.jsonl
  - split: test_multi_wording
    path: data/v7-r2/test_multi_wording.jsonl
  - split: test_original
    path: data/v7-r2/test_original.jsonl
  - split: test_single_branch
    path: data/v7-r2/test_single_branch.jsonl
  - split: test_wording
    path: data/v7-r2/test_wording.jsonl
  - split: train_control
    path: data/v7-r2/train_control.jsonl
  - split: train_treatment
    path: data/v7-r2/train_treatment.jsonl
- config_name: v8-r1
  data_files:
  - split: calibration
    path: data/v8-r1/calibration.jsonl
  - split: development
    path: data/v8-r1/development.jsonl
  - split: selection
    path: data/v8-r1/selection.jsonl
  - split: selection_fit
    path: data/v8-r1/selection_fit.jsonl
  - split: test_counterfactual
    path: data/v8-r1/test_counterfactual.jsonl
  - split: test_depth
    path: data/v8-r1/test_depth.jsonl
  - split: test_multi
    path: data/v8-r1/test_multi.jsonl
  - split: test_multi_depth
    path: data/v8-r1/test_multi_depth.jsonl
  - split: test_multi_signed_chain
    path: data/v8-r1/test_multi_signed_chain.jsonl
  - split: test_multi_transformations
    path: data/v8-r1/test_multi_transformations.jsonl
  - split: test_multi_wording
    path: data/v8-r1/test_multi_wording.jsonl
  - split: test_original
    path: data/v8-r1/test_original.jsonl
  - split: test_paired
    path: data/v8-r1/test_paired.jsonl
  - split: test_single_branch
    path: data/v8-r1/test_single_branch.jsonl
  - split: test_wording
    path: data/v8-r1/test_wording.jsonl
  - split: train_control
    path: data/v8-r1/train_control.jsonl
  - split: train_treatment
    path: data/v8-r1/train_treatment.jsonl
- config_name: v8-r2
  data_files:
  - split: calibration
    path: data/v8-r2/calibration.jsonl
  - split: development
    path: data/v8-r2/development.jsonl
  - split: selection
    path: data/v8-r2/selection.jsonl
  - split: selection_fit
    path: data/v8-r2/selection_fit.jsonl
  - split: test_counterfactual
    path: data/v8-r2/test_counterfactual.jsonl
  - split: test_depth
    path: data/v8-r2/test_depth.jsonl
  - split: test_multi
    path: data/v8-r2/test_multi.jsonl
  - split: test_multi_depth
    path: data/v8-r2/test_multi_depth.jsonl
  - split: test_multi_signed_chain
    path: data/v8-r2/test_multi_signed_chain.jsonl
  - split: test_multi_transformations
    path: data/v8-r2/test_multi_transformations.jsonl
  - split: test_multi_wording
    path: data/v8-r2/test_multi_wording.jsonl
  - split: test_original
    path: data/v8-r2/test_original.jsonl
  - split: test_paired
    path: data/v8-r2/test_paired.jsonl
  - split: test_single_branch
    path: data/v8-r2/test_single_branch.jsonl
  - split: test_wording
    path: data/v8-r2/test_wording.jsonl
  - split: train_control
    path: data/v8-r2/train_control.jsonl
  - split: train_treatment
    path: data/v8-r2/train_treatment.jsonl
---

# Premise Worlds

Frozen synthetic facts-and-rules data used in Premise's V7 and V8 comparisons.
Each row asks whether a statement is TRUE, FALSE, or UNKNOWN given an explicit world.

[Code and research article](https://github.com/adc77/premise) · [Methods](https://github.com/adc77/premise/blob/main/docs/V8_PROTOCOL.md)

## Load

```python
from datasets import load_dataset

data = load_dataset("axondendriteplus/premise-worlds", "v8-r2", split="test_paired")
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
repetitions. The 190,688 serialized rows below are not independent examples or
unique worlds. UNKNOWN edits can be identity controls; separate known-target edits.

## Counts

| Config | Split | Rows |
| --- | --- | ---: |
| v7-r1 | calibration | 1,008 |
| v7-r1 | development | 1,008 |
| v7-r1 | selection | 1,008 |
| v7-r1 | selection_fit | 1,008 |
| v7-r1 | test_counterfactual | 900 |
| v7-r1 | test_depth | 300 |
| v7-r1 | test_multi | 1,440 |
| v7-r1 | test_multi_depth | 432 |
| v7-r1 | test_multi_signed_chain | 432 |
| v7-r1 | test_multi_transformations | 11,520 |
| v7-r1 | test_multi_wording | 432 |
| v7-r1 | test_original | 900 |
| v7-r1 | test_single_branch | 200 |
| v7-r1 | test_wording | 300 |
| v7-r1 | train_control | 12,096 |
| v7-r1 | train_treatment | 12,096 |
| v7-r2 | calibration | 1,008 |
| v7-r2 | development | 1,008 |
| v7-r2 | selection | 1,008 |
| v7-r2 | selection_fit | 1,008 |
| v7-r2 | test_counterfactual | 900 |
| v7-r2 | test_depth | 300 |
| v7-r2 | test_multi | 1,440 |
| v7-r2 | test_multi_depth | 432 |
| v7-r2 | test_multi_signed_chain | 432 |
| v7-r2 | test_multi_transformations | 11,520 |
| v7-r2 | test_multi_wording | 432 |
| v7-r2 | test_original | 900 |
| v7-r2 | test_single_branch | 200 |
| v7-r2 | test_wording | 300 |
| v7-r2 | train_control | 12,096 |
| v7-r2 | train_treatment | 12,096 |
| v8-r1 | calibration | 1,008 |
| v8-r1 | development | 1,008 |
| v8-r1 | selection | 1,008 |
| v8-r1 | selection_fit | 1,008 |
| v8-r1 | test_counterfactual | 900 |
| v8-r1 | test_depth | 300 |
| v8-r1 | test_multi | 1,512 |
| v8-r1 | test_multi_depth | 432 |
| v8-r1 | test_multi_signed_chain | 432 |
| v8-r1 | test_multi_transformations | 13,608 |
| v8-r1 | test_multi_wording | 432 |
| v8-r1 | test_original | 900 |
| v8-r1 | test_paired | 3,024 |
| v8-r1 | test_single_branch | 200 |
| v8-r1 | test_wording | 300 |
| v8-r1 | train_control | 12,096 |
| v8-r1 | train_treatment | 12,096 |
| v8-r2 | calibration | 1,008 |
| v8-r2 | development | 1,008 |
| v8-r2 | selection | 1,008 |
| v8-r2 | selection_fit | 1,008 |
| v8-r2 | test_counterfactual | 900 |
| v8-r2 | test_depth | 300 |
| v8-r2 | test_multi | 1,512 |
| v8-r2 | test_multi_depth | 432 |
| v8-r2 | test_multi_signed_chain | 432 |
| v8-r2 | test_multi_transformations | 13,608 |
| v8-r2 | test_multi_wording | 432 |
| v8-r2 | test_original | 900 |
| v8-r2 | test_paired | 3,024 |
| v8-r2 | test_single_branch | 200 |
| v8-r2 | test_wording | 300 |
| v8-r2 | train_control | 12,096 |
| v8-r2 | train_treatment | 12,096 |

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
