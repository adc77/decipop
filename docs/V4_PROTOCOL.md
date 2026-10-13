# V4: mixed UNKNOWN training protocol

Predeclared after development-only diagnostics and a throughput pilot, before additional training and final-test inference. The original [proposal](V4_PROPOSAL.md) records the earlier design. This protocol does not redefine any V1–V3 outcome.

## Motivation and hypothesis

The V3 final-treatment head correctly recognizes changed-premise UNKNOWN cases but fails original-style unsupported queries. On 200 fresh diagnostic families, changing only the queried entity causes 400/400 errors; 200/200 necessary-premise flips remain correct. Consistent renaming preserves both correct and incorrect predictions. An irrelevant opposite-predicate fact preserves supported-query correctness. These diagnostics support a query-entity transfer problem under these controlled conditions; they do not prove the model's internal mechanism.

Hypothesis: additional training on both query-varied and changed-premise examples improves unsupported-query UNKNOWN accuracy while retaining changed-premise UNKNOWN accuracy. The intervention is the complete training-mixture change; no attribution to a single internal mechanism is planned.

## Initialization and matched arms

Both arms start from the same V3 treatment-final weights, private Hugging Face revision `d165bf46ed93deee9697143204f78bdccdddb17d`, with identical tokenizer and architecture. Historical optimizer states were deleted. Both arms use fresh AdamW optimizers, identical additional-training seed 317, deterministic algorithms, and the same example-position/family/label order.

Each arm uses 6,667 training parent families, 20,001 questions, three passes, 60,003 presentations, and 1,878 updates. Control uses changed-premise triples for every family. Treatment uses original-style triples for 3,334 families and changed-premise triples for 3,333; component assignment alternates accepted parent indices. Each component has one TRUE, FALSE, and UNKNOWN per family. Both arms share all parent theories and the full canonical-world register.

Changed-premise triples keep the statement fixed and flip either a selector or a necessary antecedent. Original-style triples retain the theory and ask a supported statement, its opposite, and an unsupported query about another entity. Predicate polarity and naming are randomized. Counts and noise settings share the existing generator. These controls do not rule out all structural shortcuts.

Backbone learning rate: 0.00002; head: 0.0001. AdamW weight decay: 0.01. Effective batch: 32; microbatch: 16; gradient clipping: 1.0. Linear warmup fraction: 0.05 followed by linear decay. Full-backbone fine-tuning with gradient checkpointing and BF16 autocast; FP32 weights/optimizer tensors. Maximum prompt length: 512 tokens, no truncation. Evaluation batch: 32. No architecture, loss, rendering augmentation, or rule curriculum differs between arms.

## Isolation and sample sizes

The diagnostic set has 200 families and 1,600 inputs. It excludes the 52,404 recorded prior canonical worlds. Its 800 canonical worlds are additionally excluded from training, selection, calibration, and tests.

| Partition | Families | Input rows |
| --- | ---: | ---: |
| Train, each arm | 6,667 shared | 20,001 per arm |
| Selection-fit | 400 | 1,200 |
| Selection | 400 | 1,200 |
| Final calibration | 400 | 1,200 |
| Paired original-query / changed-premise test | 1,000 shared | 3,000 per suite |
| Depth transfer | 500 | 1,500 |
| Wording/name transfer | 500 | 1,500 |
| Single-branch transfer | 500 | 1,000 |

Split parent families before constructing variants. Selection-fit, selection, and calibration each alternate original-style and changed-premise parent triples, 200 families of each type. The primary two suites share the same held-out parents; related questions are clustered for uncertainty estimates. Repeated arm/suite scoring is not independent data.

Training depths are 1–3, with chain and relational motifs and shared training renderers. Depth tests use 4–5 steps. Wording tests use reserved names and the reserved paraphrase renderer. Single-branch tests use separate oracle-validated theories and matched necessary-fact flips.

Canonicalization handles entity renaming and variable ordering. Exclude all recorded V1/V2/V3 and diagnostic worlds. Registry, oracle, rendering, label balance, IDs, partition ownership, and tokenizer lengths are audited before training. Canonical-world isolation does not exclude all recurring abstract motifs or predicate-renamed structures.

## Checkpoint selection and confidence

Each arm's candidate pool is additional-training updates **0, 250, 750, 1250, 1878**. Update zero is the shared starting model; there are eight additional checkpoints plus one shared parent, rather than ten distinct weight sets. Preserve every declared candidate's weights and progress. Intermediate candidates need not retain optimizer state after a newer recoverable checkpoint is committed.

Fit one scalar temperature per candidate on selection-fit using FP64 convex inverse-temperature bisection, 64 iterations and temperature bounds [0.05, 1000]. Score calibrated NLL on the distinct selection partition. Select its minimum within each arm; exact ties choose the earlier update. Accuracy selection is a predeclared descriptive comparison, with the same tie rule; its separately chosen model is not a second primary release.

After choices, fit the primary selected candidates' temperatures on distinct final-calibration families. Persist a checksummed selection/calibration freeze before test inference. No checkpoint choice, calibrator, data-generation change, or threshold is revised using test outcomes. Positive scalar temperature scaling preserves argmax decisions.

## Endpoints and uncertainty

Primary endpoint: treatment-minus-control UNKNOWN accuracy on the unsupported-query suite, one UNKNOWN query per 1,000 paired parents. Required companion: treatment-minus-control UNKNOWN accuracy on changed-premise worlds, using the same parents. A mixture aggregate cannot substitute for reporting these separately.

Use a paired parent-family bootstrap with 2,000 resamples, seed 317, central 95% percentile intervals. The primary engineering gate requires the positive-gain interval's lower bound to exceed zero. Retention requires the changed-premise difference interval's lower bound to exceed -0.02. These are provisional engineering criteria frozen for this run, not a formal power analysis; report unresolved criteria as inconclusive. Collapsed empirical intervals do not guarantee population performance.

Report overall accuracy, macro F1, all-family correctness, confusion matrices, predicted class counts, raw/calibrated NLL, Brier score, ECE with 10 bins, and errors above 90% confidence. Transfer-suite comparisons are descriptive without multiplicity correction. One additional-training seed does not measure seed uncertainty. UNKNOWN is a semantic answer, distinct from deferral.

The analysis script's exact SHA-256 is part of the frozen input audit. Raw predictions must agree with input IDs, families, labels, and finite three-class logits before analysis or release.

## Compute, recovery, and retention

The throughput pilot performed 20 discarded updates on diagnostic inputs; mean time after four warmups was 1.344517 seconds/update. The projected 626-update chunk plus 120 seconds overhead is 961.67 seconds. Allocate three bounded chunks per arm, a 1,050-second function timeout per training call, and one separately reserved 1,050-second recovery call. No automatic whole-job retries. The scientific training loop commits candidates and complete recoverable state, including optimizer, scheduler, order cursor, and RNG, before pruning previous recovery state.

Selection timeout: 600 seconds; final evaluation and GPU export/reload checks: 800 seconds. Prior diagnostic/pilot ceilings remain charged to the invocation reservation ledger. Startup ceiling: 180 seconds per call; scaledown: 2 seconds. Use one owned L4 worker at a time, four physical CPU cores, 32 GiB requested and maximum RAM, no endpoint or schedule.

API rates verified before submission: L4 $0.80/hour, CPU $0.0473/core-hour, RAM $0.008/GiB-hour; requested rate $1.2452/hour. Published pricing was also checked at [Modal pricing](https://modal.com/pricing). Conservative full-call compute ceilings including the unused recovery slot total $3.98879067, under the $4 compute cap, with a separate $1 storage/egress reserve. These ceilings are not expenditure. Before-study workspace metering was $6.20106047 and billed charges $0; the original $30 budget supports the allocation without treating the old credit balance as current.

Retain data, logits, candidate weights, chosen releases, progress/trajectories, failures, allocation ledger, source snapshots, and billing/cleanup records locally. Verify new selected bundles on GPU, then through pinned private-Hub downloads with independent CPU reference inference. Remove owned temporary storage and stop owned apps only after retention checks. Preserve historical models and source archives; never modify unrelated workspace resources.

Implementation remains local and uncommitted. Every substantive milestone and scope/operational change is appended to [RESEARCH_LOG.md](../RESEARCH_LOG.md). Reports must distinguish a frozen plan, submitted calls, completed training, verified artifacts, and published releases.
