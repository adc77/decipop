# Counterfactual failure analysis

The fine-tuned models score above 96% on ID and deeper-rule tests, yet both fail all 142 fact-flip questions whose derivation becomes unsupported. This is the central limit of the experiment. High benchmark accuracy does not establish reliable logical decisions.

## Fixed fact-flip suite

The frozen suite has 300 questions from 100 families. The oracle changes 200 labels. Of those, 58 remain answerable because a directly queried fact changes sign; 142 become UNKNOWN because support for a derived statement disappears. This is a post hoc breakdown of the original frozen test suite, not an additional training or checkpoint-selection experiment.

| Model | Direct fact flips correct | Broken-support cases correct | Kept original label on broken support | Wrong with calibrated confidence ≥90% |
| --- | ---: | ---: | ---: | ---: |
| Token | 58 / 58 | 0 / 142 | 142 / 142 | 130 / 142 |
| Head | 58 / 58 | 0 / 142 | 142 / 142 | 128 / 142 |
| Scratch | 29 / 58 | 0 / 142 | 71 / 142 | 0 / 142 |

The headline 29% changed-label accuracy for each pretrained model comes entirely from the 58 direct-fact cases. Neither recognizes the 142 transitions to UNKNOWN. They keep the original TRUE/FALSE label in every one of these cases. This is evidence of failure to account for premise validity; it does not identify the exact internal heuristic.

## One verified case

The example is the first broken-support question both pretrained models get wrong in the frozen file order. It was not selected for the largest confidence error.

Original facts include:

- Bella follows Farah.
- Farah is not calm.
- Dalia knows Bella.
- Dalia is not blue.

Rules:

1. If person 1 follows person 2 and person 2 is not calm, person 1 is not blue.
2. If person 1 knows person 2 and person 2 is not blue, person 1 is not bright.

The original query, “Dalia is not bright,” is TRUE: derive Bella is not blue from Bella/Farah, then Dalia is not bright from Dalia/Bella.

Flip only “Farah is not calm” to “Farah is calm.” The first rule no longer establishes Bella is not blue, so the second rule does not establish Dalia is not bright. Its explicit negation is also unsupported: the correct answer becomes UNKNOWN. The given fact that Dalia is not blue cannot substitute for the required fact about Bella.

- Token prediction after the flip: TRUE, calibrated confidence 94.97%.
- Head prediction after the flip: TRUE, calibrated confidence 74.96%.

The oracle recomputed both labels. The following is the actual modified model input:

```text
Decide using only these facts and rules. TRUE means the statement follows; FALSE means its explicit negation follows; UNKNOWN means neither follows. Absence is not negation. Answer A for TRUE, B for FALSE, or C for UNKNOWN.

Facts:
Dalia is not blue.
Hana is not blue.
Farah is calm.
Bella follows Farah.
Hana is not kind.
Bella knows Farah.
Dalia knows Bella.
Hana knows Dalia.
Bella does not support Hana.

Rules:
Whenever person 1 follows person 2 and person 2 is not calm, person 1 is not blue; this applies to all people mentioned.
Whenever person 1 knows person 2 and person 2 is not blue, person 1 is not bright; this applies to all people mentioned.

Statement: Dalia is not bright.

Answer:
```

## What the study supports

- Fine-tuning substantially improves classification on the generated distributions over the pretrained, unigram, and majority baselines.
- The head improves ID accuracy by 0.77 percentage points, with a 95% paired family-bootstrap interval of 0.27–1.30 points for this one training seed.
- The head loses much more accuracy on combined wording/name changes: 72.93% versus the token model’s 89.33%. The separate paraphrase test also favors the token model.
- Deeper chains alone do not produce a supported accuracy drop in this dataset. Both family-bootstrap intervals for ID-minus-depth accuracy include zero.
- Both fine-tuned approaches fail the broken-support intervention. A small ID ECE does not guarantee correct or calibrated behavior under changed premises.
- Scratch training completed the full protocol but stayed at chance. Its near-zero calibrated ID ECE reflects an almost uniform, uninformative prediction distribution.
- The head provides no demonstrated inference-speed advantage: batch-1 medians are 43.18 ms versus 43.47 ms, and measured batch throughput is slightly lower for the head. The backbone dominates this workload. A single benchmark does not establish a statistically reliable speed difference.

## Dataset limits and the next controlled experiment

The generator deliberately creates solvable TRUE/FALSE queries and UNKNOWN distractor-entity queries. Exact oracle labels and a chance-level unigram baseline do not exclude higher-order shortcuts involving query entities and rule conclusions. The intervention results make that limitation concrete.

A follow-up should include near-identical theories with the same entity and query but different premise validity, spanning TRUE, FALSE, and UNKNOWN. Add inactive rules and missing or explicitly opposite antecedents to training. Keep all related counterfactual worlds in one parent family when splitting, then reserve new families and interventions for testing. Freeze a new protocol before any retraining.

The scratch run needs a separate diagnosis using development data: start with one-hop unary decisions, verify full-dataset learning, then increase complexity and exposure. Its current failure does not show that random initialization can never work; size, tokenizer, attention, optimization, and pretraining are confounded.

No follow-up training, extra seed, or new dataset was run for these suggestions. The v1 results remain unchanged.
