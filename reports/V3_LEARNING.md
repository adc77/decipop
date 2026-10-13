# V3: what selection and calibration actually changed

These are descriptive examples from the selection partition, chosen after the study using the first sorted matched family and first sorted original-style UNKNOWN. They are not additional test estimates. Numbers use recorded GPU BF16 logits; calibrated probabilities use the separate final-calibration temperature.

## One statement, three worlds

Statement: **Hana is large.**

Original facts:

- Hana is not quiet.
- Gabe is not calm.
- Evan is not large.
- Dalia is not calm.
- Hana is not warm.
- Evan is round.

Rules:

- Whenever person 1 is not warm and person 1 is not quiet, person 1 is large; this applies to all people mentioned.
- Whenever person 1 is not warm and person 1 is quiet, person 1 is not large; this applies to all people mentioned.

| World | Oracle | Early decision | Final decision | Final raw confidence | Final calibrated confidence |
| --- | --- | --- | --- | ---: | ---: |
| TRUE | TRUE | TRUE | TRUE | 99.9901% | 79.99% |
| FALSE | FALSE | TRUE | FALSE | 99.9927% | 81.05% |
| UNKNOWN | UNKNOWN | TRUE | UNKNOWN | 99.9981% | 84.82% |

FALSE world: replace **Hana is not quiet.** with **Hana is quiet.**.

UNKNOWN world: replace **Hana is not warm.** with **Hana is warm.**.

UNKNOWN means neither the statement nor its explicit negation can be derived. It is a distinct label. The early head gives the same answer across these interventions; the final head's decisions follow this family's oracle.

## Calibration cannot repair a wrong decision

Statement: **Bella is quiet.**

Facts:

- Gabe is round.
- Hana is not quiet.
- Chen is not bright.
- Hana is not large.
- Chen is not blue.
- Chen is calm.
- Bella is not bright.
- Gabe is not kind.

Rules:

- For every person mentioned: if person 1 is round and person 1 is not kind, then person 1 is not quiet.
- For every person mentioned: if person 1 is round and person 1 is kind, then person 1 is quiet.

The oracle says **UNKNOWN**. The final model says **FALSE** with raw confidence **99.9964%**. Temperature scaling reduces that confidence to **83.00%**, but leaves the wrong decision unchanged.

The temperature is 4.79268. Compute probabilities as `softmax(logits / temperature)`; the largest logit remains the largest. This example has a proof about a different person from the statement. An entity-binding failure is a hypothesis to investigate, not a mechanism established by one example.

## Why the old checkpoint selection failed

An almost uniform early head pays about `log(3) = 1.099` NLL per question. The final head makes many more correct decisions, but confident mistakes on unsupported original-style queries can push its average raw NLL above that value. Fitting a temperature on selection-fit and evaluating it on separate selection families makes the final checkpoint competitive without changing its decisions. The checkpoint choice produces the accuracy gain; calibration adjusts its confidence.

Read [V3_RESULTS.md](V3_RESULTS.md) for all fresh-test results, including wording and single-branch failures and the confidence degradation on the counterfactual suite.
