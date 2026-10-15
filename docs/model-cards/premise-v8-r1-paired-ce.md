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

# Premise · V8 r1 · paired-ce

Cross-entropy on paired worlds. A full-fine-tuned SmolLM2-360M backbone with a linear
three-class decision head over the last non-padding hidden state.

[Repository](https://github.com/adc77/premise) · [Dataset](https://huggingface.co/datasets/axondendriteplus/premise-worlds) ·
[Results](https://github.com/adc77/premise/blob/main/reports/V8_RESULTS.md)

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
uv run python examples/predict.py --model axondendriteplus/premise-v8-r1-paired-ce
```

The example downloads this custom inference bundle, checks its SHA-256 manifest,
and predicts the exact held-out prompts in `examples/prompts.json`. Use
`--revision <commit>` to pin a release. The standalone bundle also supports
`python infer.py --input prompts.json` after installing `requirements.txt`.
Inputs must match `prompt-template.txt`; the maximum length is 512 tokens and
overlong inputs fail rather than truncate. This is not a stock chat model or a
Transformers `pipeline` checkpoint; use the bundled loader. Model weights are FP32.

## Training and selection

- Study: V8, repetition r1, seed 1601; dataset config `v8-r1`.
- Selected update: 1134; all arms completed 1,134 updates and 36,288 presentations.
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

V8's added consistency loss did not pass the declared replicated advantage over paired CE: the first interval includes zero; the second is positive. This is mixed evidence, not equivalence.

| Fresh test | Question accuracy | Complete families |
| --- | ---: | ---: |
| test_paired | 87.57% | 54.23% |
| test_multi | 88.56% | 66.93% |
| test_multi_transformations | 84.95% | 12.70% |
| test_multi_depth | 88.43% | 69.44% |
| test_multi_wording | 80.32% | 65.74% |
| test_multi_signed_chain | 86.81% | 67.59% |
| test_original | 94.78% | 84.33% |
| test_counterfactual | 99.44% | 98.33% |
| test_depth | 82.67% | 69.00% |
| test_wording | 46.67% | 1.00% |
| test_single_branch | 52.00% | 14.00% |

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
