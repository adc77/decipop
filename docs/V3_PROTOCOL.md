# V3: selection and calibration of retained checkpoints

Question: did raw development NLL select an unhelpful early checkpoint, and can selection after temperature fitting recover useful decisions on fresh synthetic families?

This is a prospective evaluation of a retrospectively chosen, restricted candidate pool. V2 motivated the hypothesis. We retained only step 250 and step 1878 from each V2 training arm, so this experiment compares those two checkpoints per arm. It cannot identify the best checkpoint across the full training trajectory. There is one training seed and no retraining. V1/V2 test inputs, outputs, and code stay frozen.

## Data separation

Generate fresh worlds using the unchanged V2 formal generator with seed 217. Exclude every canonical world recorded in V1 datasets, V1 robustness inputs, and V2 datasets and parent registry. Register all three related worlds under one parent, including worlds not used in original-style questions. Reject collisions between new parents. This prevents exact-world leakage; it does not prevent reuse of generator motifs, predicate vocabulary, or isomorphic structures.

Four distinct uses of data:

1. Selection-fit: 400 families, 1,200 questions. Fit a separate temperature for each candidate.
2. Selection: 400 other families, 1,200 questions. Choose each policy's checkpoint using these predictions only.
3. Final calibration: 400 other families, 1,200 questions. Fit the deployment temperature after checkpoint choices are fixed. This split never chooses a checkpoint.
4. Test: 1,000 shared parent families yield original-style and matched TRUE/FALSE/UNKNOWN suites of 3,000 questions each. Depth and wording each use 500 other parents and 1,500 questions. Single-branch transfer uses 500 other parents and 1,000 before/after questions.

The first three partitions mix original-style and counterfactual families equally, matching V2 development's intended mixture. All counterfactual UNKNOWN worlds flip a critical premise rather than delete it. Three-world families are class balanced. Single-branch transfer is 50% UNKNOWN. Depth uses four or five proof steps; other suites use one to three. Wording uses the reserved names and paraphrase renderer. Formal oracle, rendering, hashes, family isolation, and the 512-token limit must pass validation before any model scoring. Test inputs are fixed and hashed before selection; no test predictions may exist until the selection/calibration freeze is durable.

## Selection and calibration

Within each arm, compare three predeclared policies over the same early/final pool: minimum raw selection NLL, minimum selection NLL using the temperature fitted on selection-fit, and maximum selection accuracy. Exact ties choose the earlier checkpoint. All policies then receive a newly fitted temperature on final calibration. Positive scalar temperature changes confidence, never the predicted class.

Optimize NLL globally in inverse temperature, a convex scalar objective, using derivative bisection for 64 iterations with temperature bounds [0.05, 1000]. Check both bounds explicitly and report boundary solutions. Fit temperatures using FP64 calculations on the recorded GPU BF16-autocast logits. Do not reuse a V2 temperature. Compare raw and calibrated NLL, Brier score, ten-bin ECE, and high-confidence wrong counts. Accuracy is unaffected by temperature.

Freeze the complete candidate hashes, selection scores, choices, final calibration temperatures, and input/code identities before test scoring. Evaluate all four retained candidates once for diagnostic transparency; reuse their identical raw test logits for policies choosing the same checkpoint. Unselected candidates are descriptive diagnostics and cannot change the frozen choices.

## Outcomes

Primary comparison: within the treatment arm, calibrated-NLL selection minus raw-NLL selection in UNKNOWN accuracy on the fresh matched counterfactual test. Report the paired parent-family percentile bootstrap 95% interval with 2,000 resamples, seed 217. Also report their overall accuracy, all-three-world correctness, and calibrated NLL differences.

Secondary comparisons: accuracy selection versus raw-NLL selection in each arm; treatment versus control under each matching policy; all five test suites; every candidate's raw and calibrated results. Bootstrap parent families, keeping related questions together. Intervals cover test-family sampling only, not training-seed uncertainty. Secondary analyses are descriptive without multiplicity correction. A zero-width interval when predictions coincide is not evidence of equivalence across training runs.

Report confusion matrices and predicted class counts so a constant-label model cannot look successful through averaging. No external-domain or general decision-model capability claim follows from these synthetic tests. Calibration on the matched mixture need not transfer to depth, wording, or single-branch worlds. Since V2 results motivated this design, this is a follow-up experiment rather than an independent replication of the original hypothesis.

## Execution and artifacts

CPU data validation, tests, statistical analysis, and private artifact verification are authorized locally. All aggregate model scoring runs on one Modal L4 with four physical CPU cores and 32 GiB RAM. No endpoint or schedule. No Hugging Face write token enters Modal. Lock dependencies and snapshot source before inference.

Predeclare two GPU calls: selection/calibration (900-second worker timeout) and test/export (1,500 seconds). Reserve at most one 900-second recovery call. Include startup and scale-down allowances per call in the allocation ledger; retries must fit both that ledger and the $2 compute allocation. Reserve $1 for storage/egress, yielding a $3 study cap. These are planning allowances, not an instantaneous billing guarantee. The workspace net spend limit remains $0 and other workspace workloads are outside this study's attribution. Verify Modal's current published resource rates before execution and attribute observed charges to our app and volume IDs.

Publish one private, self-contained artifact per distinct selected candidate, with the frozen selection/calibration record, fresh results, weight and file hashes, class ordering, tokenizer, custom loader source, license, pinned dependency requirements, and FP32 reference predictions. Perform GPU reload checks and a pinned private Hugging Face download with fresh CPU inference checks. Retain all raw logits, inputs, manifests, and final evidence locally. Delete only this study's owned Modal volume after verified retention; preserve all V1/V2 releases and evidence. Git implementation remains local unless the user authorizes a commit or push.
