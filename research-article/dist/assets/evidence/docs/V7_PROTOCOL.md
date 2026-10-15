# V7: two fresh-seed replications of multi-entity training

Status: specification before execution. Sealed protocol identities and execution states are separate receipts in `runs/study-v7` and the research journal. No new robustness objective or architecture is included in this study.

## Scope and matched comparison

Reproduce V6's selected-pipeline comparison twice: repetition r1 uses data/model/bootstrap seed 731; r2 uses 947. Each starts both arms from private V4 mixed release `axondendriteplus/decision-lab-v4-smollm2-360m-mixed-unknown`, revision `c4ee886cd3a718036297edb56b0202b590457cf2`. The parent file inventory and pinned remote weight/manifest hashes must agree before freezing.

Each arm gets a fresh AdamW optimizer, the same seed within its repetition, identical label streams, 12,096 questions per epoch, three epochs, 36,288 presentations and 1,134 updates. Effective batch 32, microbatch 8, no truncation, maximum length 512. Backbone/head learning rates 0.00002/0.0001, weight decay 0.01, gradient clipping 1, 5% warmup. Control continues the old original-query/changed-premise curriculum; treatment shares exactly half its questions at the same positions and replaces half with multi-entity questions. Token exposure is measured rather than claimed equal. Other input properties change with the curriculum, so entity count is not an isolated causal mechanism.

Use the same V6 generator design: three-to-five entities, varying label compositions including homogeneous worlds, three explicit base facts per entity, three-world label-rotation blocks, shared signed rules, and the same training edit/rendering mixture. Multi training has 756 parent worlds and 6,048 questions. No distinguished target selector identifies a queried entity. Missing branch facts, fillers, templates and abstract rule skeletons can remain cues.

Exclude every recorded canonical world from V1–V6. Generate r1 first, then additionally exclude every r1 world from r2. Freeze both datasets and audit zero cross-repetition canonical overlap before any GPU submission. World transformations stay in their parent partition. Family/example namespaces include the repetition. Both synchronous closure and production forward chaining must agree on closures, labels and proof depths; they share parsing/grounding utilities. Validate serialized prompts/renderings, inventories, labels, exposure and token bounds.

## Selection and tests

Within each repetition, retain updates 0, 250, 600, 900 and 1,134 per arm. Update zero is the same parent in both pools. Fit each candidate's temperature on selection_fit, select minimum calibrated NLL on selection with earlier-step exact ties, then refit selected temperatures on separate calibration. Each auxiliary partition has 1,008 questions, equally old/multi. Development accuracy and accuracy-based selection are descriptive only. Temperature bounds [0.05, 1000], 64 inverse-temperature derivative-bisection iterations.

Seal candidate weight hashes, choices, calibration and prediction inventories before opening tests. Score both selected models on identical fresh inputs per repetition. Score the unchanged parent on primary multi-entity and both old retention suites. Its saved temperature is descriptive and is not refitted. Keep all scheduled weights, including both terminal checkpoints, but do not add terminal-checkpoint inference after seeing selection. If update zero wins, report it plainly; this compares selected pipelines rather than forcing terminal weights.

Fresh tests per repetition match V6 sizes and generators:

- Primary multi conjunctions: 360 worlds, 1,440 questions, 40 three-world rotation blocks for each entity count.
- Eight matched transformations on those parents: 11,520 questions across base, renaming, reordering, distractors, negated query, wording and both support edits.
- Separate multi deeper, wording and signed-chain suites: 108 worlds/432 questions each.
- Old original-query and changed-premise suites: 300 shared parents/900 questions each; old deeper and wording: 100 families/300 questions each; old single-branch: 100 families/200 questions.

Preselect `v7-r1-test_multi-multi-3-0000-0` and `v7-r2-test_multi-multi-3-0000-0` before inference. Retain these examples even if they fail or have homogeneous compositions. Aggregate inference uses BF16 autocast; exported fixed references and fresh independent CPU checks use FP32.

## Endpoints and uncertainty

Primary per repetition: treatment-minus-control fraction of complete base worlds. A world is correct only when all queried entities are correct. Bootstrap 2,000 samples with that repetition's declared seed, 95% percentile intervals; sample three-world rotation blocks within entity-count strata, weighting worlds equally. Show each entity count and each repetition separately.

Per repetition, retain V6's engineering checks: primary lower interval bound greater than zero, and both treatment-minus-parent UNKNOWN-accuracy interval lower bounds at least -0.02 on the two old retention suites. Old intervals resample paired parent families. The gain replicates under this check only if both fresh repetitions' primary checks pass; the complete engineering result replicates only if both repetitions pass all their checks. These are fixed-model world-sampling intervals, not training-seed uncertainty or a joint 95% guarantee. No multiplicity-adjusted significance claim is made. Show both seed results and a descriptive mean contrast; do not attach a pooled population/seed confidence interval to two repetitions. V6 is historical context, not newly sampled confirmatory evidence.

Report raw/calibrated accuracy, NLL, Brier, ECE, confusion matrices, class counts, complete families/worlds and errors at >=90% confidence. UNKNOWN is a semantic answer, separate from deferral. Show correct reactions to edits and correctness of unaffected entities, separating consistency from both-answer correctness. Secondary transfer and edit slices remain descriptive. Any post hoc analysis must be labeled and cannot alter the primary analysis or selection.

## Execution, budget and artifacts

One L4, four physical CPU cores, 32 GiB, at most one container per function, no automatic retries, startup timeout 180 seconds and scaledown two seconds. Repetitions and arms run sequentially. Per repetition: discarded/reference pilot timeout 420 seconds; six nominal training calls up to 1,050 seconds; selection 600 seconds; final scoring 900 seconds. One separate 1,050-second recovery reservation is retained per repetition; a failure must be recorded and the checkpoint inspected before recovery. Never silently erase a failed reservation or restart from initial weights. If exact-state recovery cannot satisfy the frozen scope, stop and preserve the attempt.

Pilot uses each arm's longest training batch for 20 discarded updates and checks the parent's fixed references. The slowest timing must project 378 updates with 15% margin plus 100-second overhead below 1,020 seconds. Do not shorten exposure after observing performance. A failed gate requires a new protocol before training.

The [published Modal pricing](https://modal.com/pricing) and billing API are rechecked before submission. Conservative requested hourly cost is $1.2452: L4 $0.80, four CPU cores at $0.0473 each, 32 GiB at $0.008 each. Per-repetition worst-case compute is $3.8359077778; the joint maximum is $7.6718155556 within an $8 allocation, plus $1 total storage/egress reserve. Check original $30-budget headroom and $0 billed before freezing and again before submission. The call ledger enforces both per-repetition and global allocations. Observed spending is reported separately from reservations and can lag.

Use only the study-owned Volume `decision-model-lab-v7-runs`, retaining resource IDs for scoped cleanup and cost attribution. Preserve all data, oracle audits, raw logits, trajectories, call receipts, failures, source archives, candidate weights and selected inference bundles. Transfer additional weights one at a time to separate private HF checkpoint repositories and check pinned LFS SHA/size metadata. This does not claim fresh intermediate inference verification. Selected bundles require fresh pinned Hub downloads and eight FP32 CPU references. Only after successful fresh verification may an owned duplicate weight file be atomically replaced by a hardlink to an identical retained local weight, recording hashes and the deduplication; this keeps verified bundle contents intact with bounded laptop storage.

Delete only the owned Volume after all retention/audit/private-verification checks and direct stopped/zero-task checks for every owned app. Attribute timestamped billing only to the recorded owned resource IDs. Archive source/evidence after all documentation writers have completed, independently checking every member hash and credential patterns. Preserve prior studies, source identities and archives. Git remains local unless publication is explicitly requested. A later pair-consistency objective needs its own hypothesis, budget and frozen protocol.
