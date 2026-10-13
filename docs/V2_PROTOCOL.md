# V2: learning premise validity

Authorized on 2026-10-13. Code remains local and uncommitted; verified models will be stored privately on Hugging Face. V1 artifacts and results remain immutable.

## Hypothesis

Training on matched worlds with changed supporting facts improves recognition of unsupported answers on fresh families. The primary comparison is treatment minus control accuracy on UNKNOWN worlds in the new counterfactual test. Secondary comparisons include all-three-world correctness, raw NLL, confidently wrong answers, original-style accuracy, and depth/wording/single-branch transfer.

## Matched arms

Both arms start from the same pinned SmolLM2-360M checkpoint and newly initialized three-class head, using seed 17 and the same optimizer, learning rates, example order indices, exposure, batch sizes, and selection rule. Neither continues a v1 checkpoint. Each receives 20,001 examples across 6,667 matched parent families and three passes, or 60,003 presentations and 1,878 updates.

Each parent theory has a supported chain, a signed selector, and explicit rules for both signs of a terminal conclusion. The control keeps the original world's facts and asks a supported query, its opposite, and an unsupported query on a distractor entity. The treatment holds one query fixed across three worlds: supported TRUE, selector-reversed FALSE, and seed-flipped or seed-removed UNKNOWN. Query sign is randomized. Removing a seed occurs in half the UNKNOWN training worlds; fact flips preserve fact count. Inactive rules occur in both arms.

The original-style control uses the new shared theories rather than reproducing the exact v1 generator. The intervention changes query selection and premise validity together. Equal questions do not imply equal input tokens or identical family dependence; those quantities are reported. This isolates a data-construction intervention within one model architecture, not a pure causal effect of one token or pretraining.

Train/development/calibration depths are 1–3. Development and calibration are identical for both arms and alternate original-style/counterfactual families. Select by development restricted-class NLL, then fit temperatures and confidence thresholds on calibration only. Freeze both selections and calibrations before final test predictions are read. No inspected v1 family enters v2.

## Fresh tests and isolation

- Original-style and counterfactual suites use the same 1,000 new parent families, three questions each. Counterfactual UNKNOWN worlds use a flipped seed, never deletion, to avoid a missing-fact-count cue.
- Depth transfer uses 500 counterfactual families at depths 4–5.
- Wording/name transfer uses 500 counterfactual families with reserved names and the existing paraphrase renderer.
- Single-branch transfer uses 500 independently generated original-rule families. One answerable query and the identical query after a critical fact flip give 1,000 examples; half are UNKNOWN. This suite has a different label balance and is scored separately.
- Group related worlds and both arms by their parent family. Check canonical hashes of every world against v1 and all other v2 partitions, not only the original world. Intended overlaps between paired suites and training arms are recorded.
- Recompute every oracle label/proof depth, verify rendering, verify matched fact changes and query identity, and audit every input against the 512-token limit.

Use 2,000 paired parent-family bootstrap resamples for the primary accuracy and full counterfactual NLL differences. Intervals quantify test-family sampling for one training seed. All other analyses are descriptive. All-world correctness prevents a constant UNKNOWN predictor from looking successful.

## Compute and execution

Use one L4, four physical CPU cores, and 32 GiB RAM, sequentially. A throughput pilot measures both arms without using held-out predictions. Reuse v1-tested training and atomic checkpoint recovery. Run detached Modal calls, keep a durable local invocation ledger, permit no automatic job retries, and retain failed-call allocations.

Maximum allocations: one 420-second pilot; eight 1,150-second training calls; one 900-second calibration/evaluation call; one 240-second fresh GPU export check; one 1,150-second retry allowance. Add the 180-second startup and two-second scale-down allowances to each call. At the frozen requested-resource rate of $1.243584/hour this totals approximately $4.87, below the $5 compute allocation. Keep a separate $1 credit reserve for storage/accounting lag. Reverify provider rates and actual remaining credits before submission.

Do not launch full training if measured throughput, memory, or forecast cannot complete both arms within these bounds. Keep evaluation capacity reserved during every submission. A failed call is not free, and a retry cannot silently expand the allocation. If a container is interrupted, preserve/clear orphan selections before resuming from a verified complete checkpoint.

## Completion

Retain generator/config hashes, dataset and input manifests, per-arm protocols, throughput evidence, training/selection logs, raw predictions, calibration, statistics, scientific plots, failed-call records, and observed costs. Export each head with custom inference code and verify an independent pinned private Hub download. Delete only the v2 working Volume after local retention and both Hub checks pass. Do not commit or push implementation without a new instruction.

The scratch-model diagnosis remains subsequent work; it is not included in this two-arm v2. No accuracy improvement is promised.
