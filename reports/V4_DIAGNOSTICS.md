# V4 controlled diagnostics

Development-only families; no additional training and no final-test estimates.

Frozen diagnostic protocol: `593025f62c7668bb77c4d89e1374c749d57d889a4e9e7792699df6b22465bfc3`.

The pinned V3 treatment-final head scored 200 new families, eight variants each. An exact oracle checked every label; all prior V1–V3 canonical worlds were excluded.

| Variant | Accuracy | Raw high-confidence errors | Calibrated high-confidence errors |
| --- | ---: | ---: | ---: |
| base_false | 100.00% | 0 / 200 | 0 / 200 |
| base_true | 100.00% | 0 / 200 | 0 / 200 |
| distractor_true | 100.00% | 0 / 200 | 0 / 200 |
| premise_unknown | 100.00% | 0 / 200 | 0 / 200 |
| query_unknown_false | 0.00% | 200 / 200 | 0 / 200 |
| query_unknown_true | 0.00% | 200 / 200 | 0 / 200 |
| renamed_true | 100.00% | 0 / 200 | 0 / 200 |
| renamed_unknown | 0.00% | 200 / 200 | 0 / 200 |

High-confidence threshold: 90%; calibration uses the historical V3 temperature. FALSE queries in this table retain the original statement and flip its selector.

| Pair | Oracle label changes | Prediction changes | Both variants correct |
| --- | ---: | ---: | ---: |
| base_true-to-query_unknown_true | 200 / 200 | 0.00% | 0.00% |
| base_false-to-query_unknown_false | 200 / 200 | 0.00% | 0.00% |
| base_true-to-premise_unknown | 200 / 200 | 100.00% | 100.00% |
| base_true-to-renamed_true | 0 / 200 | 0.00% | 100.00% |
| query_unknown_true-to-renamed_unknown | 0 / 200 | 0.00% | 0.00% |
| base_true-to-distractor_true | 0 / 200 | 0.00% | 100.00% |

The query-entity pairs keep facts, rules, predicate, and polarity fixed. Renaming is a consistent permutation of existing names. The distractor adds an opposite query-predicate fact about another person while preserving the query's oracle label. The premise intervention flips a necessary fact without changing fact count.

These observations measure sensitivity to the controlled conditions. They do not establish how the network represents entities internally. Diagnostic families are excluded from later selection, calibration, and tests.

Raw inputs and logits: local `runs/study-v4/diagnostic` and `runs/study-v4/predictions/parent/diagnostic.jsonl`. Protocol, audit, and metrics are retained beside them.
