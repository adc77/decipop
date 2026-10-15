---
license: apache-2.0
language:
- en
base_model: HuggingFaceTB/SmolLM2-360M
datasets:
- axondendriteplus/premise-worlds
tags:
- premise
- research
- text-classification
- custom-code
- reasoning
---

# Premise · V7 r2 · multi-entity

Multi-entity curriculum treatment. A full-fine-tuned SmolLM2-360M backbone with a linear
three-class decision head over the last non-padding hidden state.

[Repository](https://github.com/adc77/premise) · [Dataset](https://huggingface.co/datasets/axondendriteplus/premise-worlds) ·
[Results](https://github.com/adc77/premise/blob/main/reports/V7_RESULTS.md)

## Decisions

Given facts, universally quantified rules and a statement, predict TRUE (derivable),
FALSE (explicit negation derivable), or UNKNOWN (neither). UNKNOWN is a semantic
class, not a confidence-based refusal. No chain-of-thought or proof is generated.

## Run on CPU

Use Python 3.12 and the repository's locked environment:

```sh
git clone https://github.com/adc77/premise.git
cd premise
uv sync --extra train --extra analysis --extra baseline
uv run python examples/predict.py --model axondendriteplus/premise-v7-r2-multi-entity
```

The example downloads this custom inference bundle, checks its SHA-256 manifest,
and predicts the exact held-out prompts in `examples/prompts.json`. Use
`--revision <commit>` to pin a release. The standalone bundle also supports
`python infer.py --input prompts.json` after installing `requirements.txt`.
Inputs must match `prompt-template.txt`; the maximum length is 512 tokens and
overlong inputs fail rather than truncate. This is not a stock chat model or a
Transformers `pipeline` checkpoint; use the bundled loader. Model weights are FP32.

## Training and selection

- Study: V7, repetition r2, seed 947; dataset config `v7-r2`.
- Selected update: 900; all arms completed 1,134 updates and 36,288 presentations.
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

V7's treatment-minus-control complete-world gains replicate in both fresh runs. This selected checkpoint is one arm, not a separate independent replication.

| Fresh test | Question accuracy | Complete families |
| --- | ---: | ---: |
| test_multi | 86.53% | 68.06% |
| test_multi_transformations | 82.96% | 10.00% |
| test_multi_depth | 86.57% | 67.59% |
| test_multi_wording | 75.46% | 50.00% |
| test_multi_signed_chain | 86.81% | 65.74% |
| test_original | 93.78% | 81.67% |
| test_counterfactual | 99.56% | 98.67% |
| test_depth | 83.33% | 68.00% |
| test_wording | 51.33% | 3.00% |
| test_single_branch | 53.50% | 11.00% |

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
