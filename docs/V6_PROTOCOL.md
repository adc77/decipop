# V6: matched multi-entity training

Status: specification before execution. Execution states and sealed identities are recorded separately in the journal and study receipts.

## Question and arms

Does replacing half the existing curriculum with independently answerable multi-entity worlds improve fresh complete-world accuracy while retaining earlier UNKNOWN behavior?

Both arms start from private V4 mixed release `axondendriteplus/decision-lab-v4-smollm2-360m-mixed-unknown`, revision `c4ee886cd3a718036297edb56b0202b590457cf2`. Each gets a fresh AdamW optimizer, seed 631, identical schedules, 12,096 questions per epoch, three epochs, 36,288 presentations and 1,134 updates. Effective batch is 32, microbatch 8; maximum length 512 with no truncation. Backbone/head learning rates are 0.00002/0.0001, weight decay 0.01, clipping 1, and 5% warmup.

Control uses fresh V4-style original-query/changed-premise examples in equal family proportions. Treatment shares exactly half the control questions, at the same presentation positions, and replaces the other half with multi-entity questions. Both serialized label streams are identical and balanced. Token exposure is measured and can differ; this is a curriculum comparison, not isolation of entity count from every other input change.

## Fresh data and controls

Dataset seed is 631. Multi-entity worlds have three, four or five people, shared signed rules and independent support. Label compositions vary, including worlds with a single label; all three labels need not co-occur. Three separate worlds per design block rotate every entity's assigned label, balancing label/name/slot exposure. Facts and predicates are newly sampled for each world. Explicit base fact counts are three per entity; supported entities receive a random filler, unsupported entities two fillers. Missing branch facts and filler patterns remain possible cues. There is no distinguished selector identifying one target.

Multi training contains 84 three-world rotation blocks per entity count: 756 parent worlds, two variants each, 6,048 questions. The second variant cycles through renaming, reordering, forward support edit and inverse support edit. A quarter of these blocks use signed chains; others use conjunctions. Proof depths are one or two. Forward edits cycle TRUE→UNKNOWN→FALSE→TRUE; inverse edits reverse that cycle. One entity's support changes while explicit fact counts and all other oracle answers remain fixed. Query every entity. Facts/rules use a shared rendering order within a world.

Control has 4,032 old-style training families. Treatment retains the first 2,016 families before the matched shuffle. Development, selection-temperature fitting, checkpoint selection and final calibration each have 168 old-style families plus 63 multi-entity worlds: 1,008 questions with equal old/multi exposure. Each partition has separate fresh families. Auxiliary multi worlds include the same second-variant schedule and signed-chain mixture as training.

Final tests contain:

- Primary multi-entity conjunctions: 40 rotation blocks per entity count, 360 worlds and 1,440 base questions. Equal numbers of worlds have each entity count.
- Matched transformations on those same parents: base, renamed, reordered, distractor, negated query, paraphrase, forward support edit, inverse support edit; 11,520 questions.
- Separate multi-entity deeper, paraphrase and signed-chain suites: 12 rotation blocks per entity count per suite, 108 worlds and 432 questions each. Deeper proofs have depth three or four; other suites depth one or two. These do not reproduce the earlier transfer generators.
- Old-style original-query and changed-premise tests on 300 shared parents: 900 questions each. Old deeper/wording tests: 100 families and 300 questions each; old single-branch tests: 100 families and 200 questions.

All worlds and variants stay within their parent partition. Exclude canonical worlds recorded in V1–V5. Canonical exclusion does not exclude every abstract rule skeleton. The production solver and synchronous closure must agree on complete closure and labels; they share parsing/grounding utilities. Validate serialized theories, rendered prompts, proof depths, intervention locality, per-entity fact counts and inventories before freezing. Reject contradictions and overlength prompts. Retain failures and raw audit evidence.

## Selection, calibration and untouched tests

Retain updates 0, 250, 600, 900 and 1,134 per arm; update zero is the shared initial bundle. Fit one temperature per candidate on selection_fit, then select the minimum calibrated NLL on selection, breaking exact ties by earlier update. Accuracy-based selections are descriptive only. Development metrics are logged and cannot select checkpoints. Refit selected temperatures on the separate calibration partition. Temperature bounds are [0.05, 1000], with 64 convex inverse-temperature derivative-bisection iterations.

Seal selection, candidate hashes, calibration and their prediction inventories before final inference. Score both selected arms on identical untouched final inputs. Also score the unchanged parent on primary multi-entity, old original-query and old changed-premise tests; its stored temperature is descriptive and not refitted. Aggregate inference uses BF16 autocast. Exported fixed references and independent CPU reload checks use FP32. Select the first primary family `v6-test_multi-multi-3-0000-0` before inference, retaining its results even if it fails.

## Endpoints and uncertainty

Primary: treatment-minus-control fraction of complete base worlds on test_multi. A world is correct only if every queried entity is correct. Bootstrap 2,000 resamples, seed 631, 95% percentile interval. Resample three-world label-rotation blocks within each entity-count stratum to account for the balanced construction; transformations remain attached to their parent world. The estimate weights worlds equally, not individual questions. Report complete worlds separately for each entity count.

Two predeclared engineering checks: the primary interval lower bound exceeds zero, and both treatment-minus-parent UNKNOWN-accuracy intervals on old original-query and changed-premise tests have lower bounds at least −0.02. Old-test intervals resample parent families. These checks concern these fixed, single-seed models; they do not prove general decision ability or reproducibility across training seeds. Secondary comparisons, transfer results and paired entity checks are descriptive and not multiplicity-adjusted. Collapsed empirical intervals do not prove population perfection or equivalence.

Report raw/calibrated accuracy, NLL, Brier, ECE (10 bins), confusion matrices, class counts, complete families/worlds and errors with ≥90% confidence. Semantic UNKNOWN and confidence-based deferral are distinct. Report correct response to interventions and unaffected consistency separately from both-answer correctness; constant wrong predictions can be highly invariant. Any later architecture change needs its own hypothesis and baseline.

## Bounded execution and retention

One L4, four physical CPU cores, 32 GiB, one container per function, no automatic retries, startup timeout 180 seconds and scaledown two seconds. Pilot timeout 420 seconds; six training calls at most 1,050 seconds each; selection 600 seconds; final scoring 900 seconds. One 1,050-second recovery call is reserved and requires recording the failure before submission. All submitted and failed reservations remain in the ledger.

Pilot uses each arm's longest training batch for 20 discarded updates and checks the unchanged parent's fixed references. Training starts only if the slowest measured update projects 378 updates with a 15% margin and 100-second overhead below 1,020 seconds. The frozen exposure is not shortened after seeing endpoints. If the gate fails, preserve the attempt and revise the protocol before any training submission.

At conservative [published Modal rates](https://modal.com/pricing), L4 $0.80/hour, CPU $0.0473/core-hour and memory $0.008/GiB-hour, requested hourly cost is $1.2452. The full timeout/startup/scaledown compute ceiling, including unused recovery, must stay below $4; reserve a separate $0.50 for storage/egress. Check current API rates and billing before submission. This allocation is not observed spending and the workspace gross total is not this study's attributed cost.

Preserve raw inputs, oracle records, logits, trajectories, call receipts, failed logs, all scheduled candidate weights, selected inference bundles and independently verified source archives. Verify retained hashes before deleting only the owned V6 Volume after direct stopped/zero-task app checks. Record timestamped resource-ID-attributed billing. Git implementation remains local unless publication is requested; prior frozen studies and archives are preserved.
