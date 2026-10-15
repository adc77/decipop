# After V7: test a paired training objective

Status: authorized on 2026-10-14. The original proposal below is implemented under the separately frozen [V8 protocol](V8_PROTOCOL.md). Execution status and evidence are recorded in [the journal](../RESEARCH_LOG.md) and [execution guide](V8_EXECUTION.md).

V7's gain repeats on fresh seeds: treatment reaches 245/360 complete base worlds in both repetitions, versus control's 25/360 and 19/360. Both UNKNOWN-retention checks pass. Only 43/360 and 36/360 treatment families survive every transformation. Both preselected examples still answer TRUE after a necessary premise is removed, although explicit negative replacements correctly produce FALSE. Read [the results](../reports/V7_RESULTS.md).

## Question and matched comparison

Does an explicit relation between paired predictions improve faithful responses beyond training on the same paired examples with ordinary cross-entropy?

Both arms receive identical base/edited input pairs, oracle labels, batch ordering, token exposure, optimizer settings and checkpoint-selection procedure. Control applies cross-entropy to both examples. Treatment adds one Jensen–Shannon term after aligning the paired labels. Changing data augmentation only in treatment would confound the loss comparison, so both arms get every paired example.

Proposed starting bundle: r1's already verified selected treatment, `axondendriteplus/decision-lab-v7-r1-smollm2-360m-multi-entity`, commit `6aa3e01e41460dba7049c044906dc81fe4c7ba53`. Use the first-repetition convention rather than choosing the best opened test result. Both arms start from identical weights with fresh optimizers. Keep the existing model architecture.

For probabilities p and q, labels y and y′, and mean m:

```text
P = identity if y == y′; otherwise swap the two corresponding class coordinates
a = P p
m = (a + q) / 2
JS = (KL(a || m) + KL(q || m)) / 2
control_loss = (CE(p, y) + CE(q, y′)) / 2
treatment_loss = control_loss + 0.5 * JS
```

The proposed coefficient is 0.5, fixed rather than selected from final tests. Oracle labels define alignment only during supervised training; they never enter inference inputs. Unaffected people use identity alignment, while a changed answer uses the corresponding class swap. Supervised loss remains on both examples: two wrong but consistently related answers can still have zero JS. Implement the numerical calculation with log probabilities and finite-gradient checks, without silently clipping away failures. The coefficient and exact objective must be frozen after CPU feasibility checks and before GPU training.

Jensen–Shannon consistency has prior use in image classification in [AugMix](https://arxiv.org/abs/1912.02781v2) and its [original loss implementation](https://github.com/google-research/augmix/blob/master/cifar.py). Our oracle-based alignment for label-changing logical edits is a proposed adaptation, not a result established by that paper. No novelty or guaranteed robustness claim is made.

## Data and evaluation before submission

Build fresh multi-entity worlds containing support removals, explicit reversals and edits to irrelevant facts. Keep both affected and unaffected queries. Retain old-format questions in both arms to measure forgetting. Preserve class/name/position balance, constant fact-count controls, pair identity, both symbolic-oracle checks and no-truncation audits. Exclude all recorded V1–V7 canonical worlds; keep a parent and its edits in one partition and exclude the first new repetition from the second. Canonical exclusion does not remove shared templates or abstract rule skeletons.

Propose two fresh seeded comparisons, with seeds, exposure, pair counts and test sizes declared before submission. Primary endpoint: treatment-minus-control complete paired worlds, requiring every queried person correct before and after the local edit. Report target both-answer correctness and unaffected-person correctness separately. Preserve base-world performance and both old UNKNOWN-retention checks against the unchanged parent. Require a positive primary lower interval bound in both repetitions and predeclared retention margins; show each seed separately without a pooled training-seed interval or joint confidence guarantee. Freeze exact sampling blocks, multiplicity limits and margins in the protocol.

Keep separate selection-fit, selection, calibration and final tests. Include update zero and scheduled intermediate/terminal weights. Select using calibrated NLL on the shared selection partition. Retain wording, ordering, distractor, signed-rule and depth transfer tests, with saved-temperature confident errors. Do not use V7's opened tests as new confirmatory evidence.

## Completion and decision

Recheck rates, original-budget headroom, zero billed charges and disk space after V7 closure. Reserve all calls, failure recovery and storage before training; this proposal reserves nothing. Require timing gates, exact-state recovery, complete raw records, private scheduled-weight retention, fresh selected-Hub CPU checks, owned-resource cleanup and source closure. Keep full optimizer/RNG states explicitly temporary unless a new retention plan is funded.

If paired cross-entropy improves behavior and the added term does not, record that result rather than crediting the regularizer. If neither improves faithful edits, use fresh training/development evidence to choose a specific architecture experiment. Do not simultaneously change the backbone, input representation and objective. The original scratch model's chance-level result remains part of the record.
