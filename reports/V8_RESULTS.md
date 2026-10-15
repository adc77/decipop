# V8 results: paired prediction consistency

Scientific scoring is complete. Artifact verification, private release and cleanup are recorded separately in the execution receipt.

Both arms start from the pinned V7 r1 treatment and see identical paired examples. Control uses CE; treatment adds 0.5 times label-aligned Jensen-Shannon divergence. Seeds 1601/1801 have disjoint canonical worlds. Aggregate inference uses BF16; FP32 GPU/CPU reference checks cover eight inputs per selected export.

| Seed | Control paired worlds | Treatment paired worlds | Gain (95% interval) |
| --- | ---: | ---: | ---: |
| 1601 | 205/378 | 208/378 | 0.79 pp [-1.85, 3.70] |
| 1801 | 221/378 | 235/378 | 3.70 pp [1.06, 6.35] |

Intervals resample three-world label-rotation blocks within entity-count and edit-type strata. They describe world sampling for each trained run; there is no pooled training-seed interval or joint confidence guarantee.

The declared replication check fails: the first interval includes zero. All three retention checks pass in both runs. This is mixed evidence for an advantage from the added consistency loss, not evidence of equivalence.

## r1

| Arm | Selected update | Temperature |
| --- | ---: | ---: |
| control | 1134 | 1.381232 |
| treatment | 1134 | 1.227820 |

| Suite | Control accuracy | Treatment accuracy | Treatment complete families |
| --- | ---: | ---: | ---: |
| test_paired | 87.57% | 87.47% | 55.03% |
| test_multi | 88.56% | 88.56% | 68.25% |
| test_multi_transformations | 84.95% | 84.84% | 13.23% |
| test_multi_depth | 88.43% | 87.73% | 67.59% |
| test_multi_wording | 80.32% | 81.25% | 62.96% |
| test_multi_signed_chain | 86.81% | 87.04% | 69.44% |
| test_original | 94.78% | 94.89% | 84.67% |
| test_counterfactual | 99.44% | 99.44% | 98.33% |
| test_depth | 82.67% | 83.67% | 71.00% |
| test_wording | 46.67% | 47.00% | 5.00% |
| test_single_branch | 52.00% | 50.50% | 12.00% |

Parent comparisons (differences in percentage points):

| Check | Difference | 95% world-sampling interval |
| --- | ---: | ---: |
| CE: complete paired worlds | 4.50 | [1.32, 7.67] |
| Consistency: complete paired worlds | 5.29 | [1.85, 8.47] |
| Consistency: complete base worlds | 4.23 | [0.79, 7.67] |
| Consistency: original UNKNOWN | 0.00 | [-1.33, 1.33] |
| Consistency: counterfactual UNKNOWN | 0.00 | [-1.00, 1.00] |

| Edit / person | Control both correct | Treatment both correct |
| --- | ---: | ---: |
| renamed | 1301/1512 | 1304/1512 |
| reordered | 1216/1512 | 1227/1512 |
| distractor | 1004/1512 | 976/1512 |
| negated_query | 1283/1512 | 1296/1512 |
| wording | 1123/1512 | 1143/1512 |
| support_removal/target | 278/378 | 274/378 |
| support_removal/target_known | 171/252 | 169/252 |
| support_reversal/target | 318/378 | 314/378 |
| support_reversal/target_known | 211/252 | 209/252 |
| irrelevant_edit/target | 333/378 | 328/378 |
| support_removal/unaffected | 942/1134 | 940/1134 |
| support_reversal/unaffected | 942/1134 | 942/1134 |
| irrelevant_edit/unaffected | 987/1134 | 990/1134 |
| support_removal/target_unknown | 107/126 | 105/126 |
| support_reversal/target_unknown | 107/126 | 105/126 |
| different_entity_different_label | 959/1353 | 975/1353 |

## r2

| Arm | Selected update | Temperature |
| --- | ---: | ---: |
| control | 1134 | 1.454196 |
| treatment | 1134 | 1.519593 |

| Suite | Control accuracy | Treatment accuracy | Treatment complete families |
| --- | ---: | ---: | ---: |
| test_paired | 88.33% | 89.55% | 62.17% |
| test_multi | 89.15% | 90.08% | 72.75% |
| test_multi_transformations | 85.16% | 85.42% | 12.17% |
| test_multi_depth | 88.43% | 89.12% | 71.30% |
| test_multi_wording | 77.78% | 79.17% | 53.70% |
| test_multi_signed_chain | 90.51% | 89.81% | 73.15% |
| test_original | 94.11% | 94.00% | 82.00% |
| test_counterfactual | 100.00% | 99.56% | 98.67% |
| test_depth | 80.00% | 78.33% | 62.00% |
| test_wording | 44.67% | 44.67% | 3.00% |
| test_single_branch | 52.50% | 49.50% | 9.00% |

Parent comparisons (differences in percentage points):

| Check | Difference | 95% world-sampling interval |
| --- | ---: | ---: |
| CE: complete paired worlds | 0.79 | [-2.12, 3.70] |
| Consistency: complete paired worlds | 4.50 | [1.32, 7.94] |
| Consistency: complete base worlds | 1.85 | [-1.32, 5.03] |
| Consistency: original UNKNOWN | -0.33 | [-1.00, 0.00] |
| Consistency: counterfactual UNKNOWN | -0.33 | [-1.67, 0.67] |

| Edit / person | Control both correct | Treatment both correct |
| --- | ---: | ---: |
| renamed | 1305/1512 | 1316/1512 |
| reordered | 1247/1512 | 1254/1512 |
| distractor | 988/1512 | 989/1512 |
| negated_query | 1269/1512 | 1291/1512 |
| wording | 1151/1512 | 1155/1512 |
| support_removal/target | 266/378 | 257/378 |
| support_removal/target_known | 160/252 | 160/252 |
| support_reversal/target | 317/378 | 313/378 |
| support_reversal/target_known | 211/252 | 216/252 |
| irrelevant_edit/target | 330/378 | 326/378 |
| support_removal/unaffected | 958/1134 | 978/1134 |
| support_reversal/unaffected | 948/1134 | 981/1134 |
| irrelevant_edit/unaffected | 1002/1134 | 1022/1134 |
| support_removal/target_unknown | 106/126 | 97/126 |
| support_reversal/target_unknown | 106/126 | 97/126 |
| different_entity_different_label | 986/1344 | 1013/1344 |

## Interpretation limits

This is a synthetic, oracle-supervised decision task. Matched exposure isolates the added training objective within this pipeline; it does not prove an internal reasoning mechanism, unrestricted generalization or business usefulness. Base UNKNOWN cases can be identity edits; target behavior for known-answer removals must be read separately. Calibration changes confidence, not predicted classes. Correct complete paired worlds are stricter than question accuracy.

Treatment improves over parent in both runs; CE's second-run parent interval includes zero. The added loss does not pass the declared replicated advantage over CE. No coefficient sweep or extra seeds are authorized. All scheduled weights are retained privately; optimizer/RNG state is temporary. V1–V7 scientific results remain historical, rather than new confirmatory tests.

See [protocol](../docs/V8_PROTOCOL.md), [execution](../docs/V8_EXECUTION.md), [journal](../RESEARCH_LOG.md) and runs/study-v8 for identities, raw counts and closure.

## Preselected premise-removal examples

### r1: Aman is authorized.

Base facts:

- Chen is enabled.
- Aman is enabled.
- Bella is registered.
- Chen is active.
- Bella is enabled.
- Aman is verified.
- Aman is registered.
- Chen is registered.
- Bella is active.

Rules:

- For every person mentioned: if person 1 is registered and person 1 is enabled, then person 1 is authorized.
- For every person mentioned: if person 1 is registered and person 1 is not enabled, then person 1 is not authorized.

Removed: ['Aman is enabled.']. Added irrelevant filler: ['Aman is active.'].

| Input | Gold | Parent | Paired CE | Added consistency |
| --- | --- | --- | --- | --- |
| base | TRUE | TRUE (80.92%) | TRUE (70.33%) | TRUE (56.65%) |
| support_removal | UNKNOWN | TRUE (76.30%) | UNKNOWN (56.10%) | UNKNOWN (58.84%) |

### r2: Aman is verified.

Base facts:

- Aman is ready.
- Bella is ready.
- Bella is active.
- Chen is enabled.
- Chen is registered.
- Aman is registered.
- Chen is ready.
- Bella is enabled.
- Aman is enabled.

Rules:

- For every person mentioned: if person 1 is ready and person 1 is not enabled, then person 1 is not verified.
- For every person mentioned: if person 1 is ready and person 1 is enabled, then person 1 is verified.

Removed: ['Aman is enabled.']. Added irrelevant filler: ['Aman is active.'].

| Input | Gold | Parent | Paired CE | Added consistency |
| --- | --- | --- | --- | --- |
| base | TRUE | TRUE (72.70%) | TRUE (72.24%) | TRUE (70.81%) |
| support_removal | UNKNOWN | TRUE (61.16%) | TRUE (65.06%) | TRUE (72.33%) |

Examples were fixed in the data freeze, without selecting a favorable model result. Confidence comes from retained BF16 logits and each saved temperature; no extra inference was run. These cases illustrate behavior without estimating population performance or proving a mechanism.

## Verified completion

Added consistency does not pass the full declared replicated decision rule. Treatment improves over the unchanged parent in both repetitions, but the control comparison remains mixed and transfer remains weak. Close this bounded loss experiment; any architecture study needs a separate scope.

Complete paired-world counts (parent/control/treatment): {'r1': {'parent': 188, 'control': 205, 'treatment': 208}, 'r2': {'parent': 218, 'control': 221, 'treatment': 235}}. Replicated positive parent contrasts: {'control': False, 'treatment': True}. These secondary parent checks do not add a joint confidence guarantee.

All 126 closing CPU tests pass. Final audits verify 165,344 prediction rows across 116 files. Both independent raw-count audits and all 44 final metric recomputations pass. Four selected bundles pass fresh pinned-Hub CPU reference checks; every scheduled candidate is retained privately. Original preparation, two discarded pilots, source-guard rejection and CPU seed-replay failure remain recorded. The corrected pilots explicitly consumed the original retry allowances. A later expired-JWT request was rejected before the final chunk started; one explicit operational amendment resumes update 756 through 1,134 with saved optimizer/RNG state and unchanged trajectory prefix. The failed reservation remains in the ledger; total reservation stays $9, reallocated to $8.50 compute plus $0.50 storage/transfer. A later cancelled selection call retained all 30 candidate prediction files. Those 30,240 rows were reused with the unchanged selector; separate calibration and final tests completed inside the originally reserved test call, with no completed inference repeated or extra GPU allocation.

All 9 corrected-study owned apps are stopped with zero tasks, and its temporary Volume is deleted. The original three apps and Volume were verified separately before deletion. Full optimizer/RNG states were temporary; frozen data, source, logits, trajectories and model weights remain.

Observed owned metering, including original pilot resources: $3.70114067 at 2026-10-14T21:46:09.644358+00:00; workspace billed $0E-8. This is a snapshot, not an invoice or live wallet balance.

- r1 control: `axondendriteplus/decision-lab-v8-r1-smollm2-360m-paired-ce` at `8e29cda7bc2f763a54bfbc3ac560da365f038c7b`.
- r1 treatment: `axondendriteplus/decision-lab-v8-r1-smollm2-360m-paired-consistency` at `23dfd410a8ba486e543a178fd9039a06943e2aae`.
- r2 control: `axondendriteplus/decision-lab-v8-r2-smollm2-360m-paired-ce` at `8d653fe264308b87e2201cb53d69f25b0308487b`.
- r2 treatment: `axondendriteplus/decision-lab-v8-r2-smollm2-360m-paired-consistency` at `b3067848649220c36d8ef6cd9d44e0a13d8f504e`.

Code, reports, raw records and source archives remain local, uncommitted and unpushed. The final archive manifest and external closure receipt record archival verification.
