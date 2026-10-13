# V4: one family, two forms of UNKNOWN

Family `v4-test-000000` was chosen by lexicographic order before final-test inference. This is an illustrative example, not an additional estimate of test accuracy.

## test_original

Statement: **Hana is verified.**; oracle: **FALSE**.

Facts:

- Hana is ready.
- Evan is not authorized.
- Hana is enabled.
- Dalia is not authorized.
- Evan is approved.
- Dalia is verified.

Rules:

- Whenever person 1 is enabled and person 1 is ready, person 1 is not verified; this applies to all people mentioned.
- Whenever person 1 is enabled and person 1 is not ready, person 1 is verified; this applies to all people mentioned.

| Arm | Prediction | Raw confidence | Calibrated confidence | TRUE/FALSE/UNKNOWN logits |
| --- | --- | ---: | ---: | --- |
| control | FALSE | 100.00% | 82.22% | -4.5312, 8.3125, -5.7500 |
| treatment | FALSE | 91.67% | 89.29% | -6.0938, 2.8594, 0.4590 |

Statement: **Hana is not verified.**; oracle: **TRUE**.

Facts:

- Hana is ready.
- Evan is not authorized.
- Hana is enabled.
- Dalia is not authorized.
- Evan is approved.
- Dalia is verified.

Rules:

- Whenever person 1 is enabled and person 1 is ready, person 1 is not verified; this applies to all people mentioned.
- Whenever person 1 is enabled and person 1 is not ready, person 1 is verified; this applies to all people mentioned.

| Arm | Prediction | Raw confidence | Calibrated confidence | TRUE/FALSE/UNKNOWN logits |
| --- | --- | ---: | ---: | --- |
| control | TRUE | 100.00% | 82.92% | 9.4375, -3.8438, -4.7500 |
| treatment | TRUE | 86.29% | 83.54% | 2.8906, -4.5312, 1.0469 |

Statement: **Evan is not verified.**; oracle: **UNKNOWN**.

Facts:

- Hana is ready.
- Evan is not authorized.
- Hana is enabled.
- Dalia is not authorized.
- Evan is approved.
- Dalia is verified.

Rules:

- Whenever person 1 is enabled and person 1 is ready, person 1 is not verified; this applies to all people mentioned.
- Whenever person 1 is enabled and person 1 is not ready, person 1 is verified; this applies to all people mentioned.

| Arm | Prediction | Raw confidence | Calibrated confidence | TRUE/FALSE/UNKNOWN logits |
| --- | --- | ---: | ---: | --- |
| control | TRUE | 100.00% | 82.99% | 9.4375, -3.8438, -4.8125 |
| treatment | TRUE | 99.03% | 98.34% | 4.4688, -4.6875, -0.1631 |

## test_counterfactual

Statement: **Hana is not verified.**; oracle: **FALSE**.

Facts:

- Dalia is verified.
- Evan is not authorized.
- Hana is enabled.
- Dalia is not authorized.
- Evan is approved.
- Hana is not ready.

Rules:

- Whenever person 1 is enabled and person 1 is ready, person 1 is not verified; this applies to all people mentioned.
- Whenever person 1 is enabled and person 1 is not ready, person 1 is verified; this applies to all people mentioned.

| Arm | Prediction | Raw confidence | Calibrated confidence | TRUE/FALSE/UNKNOWN logits |
| --- | --- | ---: | ---: | --- |
| control | FALSE | 100.00% | 82.29% | -4.4062, 8.4375, -5.6875 |
| treatment | FALSE | 93.57% | 91.44% | -6.2500, 3.1719, 0.4922 |

Statement: **Hana is not verified.**; oracle: **TRUE**.

Facts:

- Hana is ready.
- Evan is not authorized.
- Hana is enabled.
- Dalia is not authorized.
- Evan is approved.
- Dalia is verified.

Rules:

- Whenever person 1 is enabled and person 1 is ready, person 1 is not verified; this applies to all people mentioned.
- Whenever person 1 is enabled and person 1 is not ready, person 1 is verified; this applies to all people mentioned.

| Arm | Prediction | Raw confidence | Calibrated confidence | TRUE/FALSE/UNKNOWN logits |
| --- | --- | ---: | ---: | --- |
| control | TRUE | 100.00% | 82.92% | 9.4375, -3.8438, -4.7500 |
| treatment | TRUE | 86.29% | 83.54% | 2.8906, -4.5312, 1.0469 |

Statement: **Hana is not verified.**; oracle: **UNKNOWN**.

Facts:

- Dalia is verified.
- Evan is not authorized.
- Hana is ready.
- Dalia is not authorized.
- Evan is approved.
- Hana is not enabled.

Rules:

- Whenever person 1 is enabled and person 1 is ready, person 1 is not verified; this applies to all people mentioned.
- Whenever person 1 is enabled and person 1 is not ready, person 1 is verified; this applies to all people mentioned.

| Arm | Prediction | Raw confidence | Calibrated confidence | TRUE/FALSE/UNKNOWN logits |
| --- | --- | ---: | ---: | --- |
| control | UNKNOWN | 100.00% | 84.40% | -5.0625, -5.5625, 9.0625 |
| treatment | UNKNOWN | 100.00% | 99.99% | -3.9844, -3.6719, 7.1562 |

The original-query suite keeps facts and rules fixed and changes the statement. Its UNKNOWN statement concerns an unsupported entity. The changed-premise suite keeps the statement fixed and changes facts. Its UNKNOWN world removes a necessary derivation by flipping an antecedent. Both are the same semantic label, but they require transfer across different input changes.

Training uses three-class cross-entropy: minus the log probability of the oracle label. Both arms update the same backbone and head from the same starting weights. The treatment's data requires handling both types of input change. Calibration divides logits by a positive temperature; it adjusts confidence without changing the winning label. These are observed outputs, not proof of internal reasoning.

Read [V4_RESULTS.md](V4_RESULTS.md) for the complete held-out results and limitations.

## What this preselected example shows

The treatment still fails the unsupported statement “Evan is not verified.” It predicts TRUE with 98.34% calibrated confidence, although the rules can derive a verified/not-verified conclusion only for someone with the required enabled and ready facts. Evan has neither prerequisite. Dalia's verified fact concerns Dalia; it does not answer Evan's statement.

Both arms correctly answer UNKNOWN after Hana's necessary enabled premise is negated. That contrasts two observable behaviors: recognizing missing support for the distinguished rule target and recognizing that the queried person has no support. Mixed training improves the latter to 815/1,000 on the full fresh suite, but this example belongs to the remaining 185 errors. It was chosen before inference and retained despite being a failure. A changed training mixture and fitted confidence score do not establish general proof binding.
