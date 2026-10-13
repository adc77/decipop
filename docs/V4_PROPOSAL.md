# V4 proposal: learn both forms of UNKNOWN

Created 2026-10-13. Authorized on 2026-10-13; implementation and diagnostic preparation started. This remains the design draft; the eventual frozen protocol and execution records determine what actually ran.

## Question

Can the existing small decision head learn to track the queried entity and recognize unsupported queries while preserving its ability to respond to changed premises?

[V3](../reports/V3_RESULTS.md) gives the reason to ask: the final treatment achieved 100% UNKNOWN accuracy on fresh changed-premise families but 0% on original-style unsupported queries. It also achieved only 43.4% wording accuracy and 51.0% single-branch accuracy. Better selection recovered useful weights; those weights still generalize poorly across these conditions.

The entity-binding hypothesis comes from a descriptive [Bella/Gabe example](../reports/V3_LEARNING.md). We must measure controlled interventions before claiming it explains the broader failure.

## Phase A: diagnose before retraining

Use the pinned V3 treatment-final bundle, revision `d165bf46ed93deee9697143204f78bdccdddb17d`. Start with 200 new diagnostic parent families. Keep them development-only and exclude their canonical worlds from every later selection, calibration, and final test partition.

Generate and oracle-check controlled variants:

| Intervention | What is fixed | What it measures |
| --- | --- | --- |
| Ask about an unsupported person | Facts, rules, predicate, and polarity | Whether changing only the query's entity changes the answer appropriately |
| Consistent entity renaming | Formal world and label | Sensitivity to names rather than logical relationships |
| Add irrelevant facts about another person | Query and its complete oracle proof status | Interference from an unrelated proof or distractor |
| Flip a necessary antecedent | Query, rule structure, fact count | Retention of the premise-change behavior observed in V3 |

For query-entity changes, include supported positive and negative cases paired with UNKNOWN. The oracle must confirm that neither polarity is derivable for the unsupported person. Keep variants in the same parent family. Reject contradictions, alternate proofs that defeat the intended intervention, and accidental label changes in invariance checks.

Record accuracy by label, all-variants-correct rate, oracle-required prediction changes, consistency on label-preserving pairs, and raw/calibrated confidence on errors. Save inputs, proofs, logits, and example identifiers. A model changing its prediction is useful only when it changes to the correct label.

CPU generation and tests can precede GPU work. Any diagnostic inference call gets a resource/timeout reservation first. Use results to finalize the training design, record the decision, and freeze the actual V4 protocol before launching training. Do not treat these inspected diagnostic families as unseen final-test evidence.

## Phase B: one controlled training comparison

Start both arms from identical retained V3 treatment-final weights and tokenizer. Initialize the same new optimizer in both arms; the historical optimizer state was deleted, so this is additional fine-tuning from retained weights, not an exact continuation of V2 optimization. Use the same seed, architecture, loss, learning-rate schedule, sequence limit, batch size, checkpoint schedule, and number of presentations.

| Arm | Proposed training mix |
| --- | --- |
| Control | Changed-premise TRUE/FALSE/UNKNOWN examples only |
| Treatment | Approximately half changed-premise examples and half query/entity-varied TRUE/FALSE/UNKNOWN examples |

Use the same pool of parent theories wherever the two example forms are feasible. Query-varied triples ask supported TRUE and FALSE statements and an unsupported statement about a different entity in the unchanged theory. Counterfactual triples keep the query fixed and change an antecedent. Balance all three labels within each component; control naming, query position, proof depth, predicate frequency, and fact/rule counts to reduce trivial cues.

Working target: 6,667 training families, 20,001 questions per arm, three passes, effective batch 32, and 1,878 updates per arm. This matches earlier exposure. The precise mixture allocation and feasibility checks belong in the frozen protocol. If diagnostics or throughput require a different exposure, revise the draft before freezing and use the same revised exposure for both arms.

The intervention being tested is the complete change in training mixture. It does not isolate entity binding from every other difference introduced by query-varied examples. Do not attribute a gain to a particular internal mechanism without a separate ablation.

Do not add a larger model, new architecture, auxiliary loss, single-branch curriculum, or paraphrase augmentation to the treatment simultaneously. Those would obscure whether the proposed data mixture helped. Keep shared training wording/rule-structure settings fixed; use the harder conditions as transfer tests.

## Data isolation and evaluation

Exclude all recorded V1/V2/V3 canonical worlds and Phase A diagnostic worlds. Split parent families before creating queries, counterfactuals, renamings, or distractors. Never put related variants into different partitions. Canonical-world exclusion is necessary but still permits recurring abstract motifs; report that limitation.

Working partition targets, to be finalized with generation feasibility:

| Partition | Parent families | Purpose |
| --- | ---: | --- |
| Training | 6,667 | Matched training pool for both arms |
| Selection-fit | 400 | Fit candidate temperatures |
| Selection | 400 | Choose a checkpoint using calibrated NLL |
| Final calibration | 400 | Fit the chosen checkpoint's final temperature |
| Paired primary test | 1,000 | Score both UNKNOWN types on the same held-out parents |
| Depth transfer | 500 | Longer proof chains |
| Wording/name transfer | 500 | Reserved renderer and entity vocabulary |
| Single-branch transfer | 500 | Reserved rule structure |

Shared selection/calibration inputs contain the same fixed mixture of both example types for both arms. Predeclare the saved candidate steps, tie-break rule, temperature bounds, and scoring code. Calibrated NLL is the primary selector; accuracy selection can be a predeclared descriptive comparison. Do not select from final-test results.

Freeze selected weights and final temperatures before opening final tests. Measure raw and calibrated NLL, Brier score, ECE, accuracy, per-label accuracy, confusion matrices, paired intervention correctness, and complete-family correctness. Calibration cannot repair wrong labels; report confidence changes separately from decision changes.

Primary endpoint: treatment-minus-control UNKNOWN accuracy on unsupported query/entity examples, with a paired parent-family bootstrap interval. Report retained changed-premise performance as a required companion result, including any regression. The concrete experiment targets both UNKNOWN types; their aggregate alone can hide a tradeoff.

As a provisional engineering gate, require the primary gain's 95% interval to exclude zero and a 95% lower bound on changed-premise UNKNOWN accuracy change above -2 percentage points. This is a draft acceptance rule, not a power analysis or a guarantee that 1,000 families suffice. Freeze the rule and bootstrap procedure before inference; report outcomes as inconclusive if the intervals do not resolve it. Other suites remain descriptive transfer results.

A single-seed study measures family-sampling uncertainty for fixed trained models. Seed replication is a separate scoped extension after the primary comparison and its cost gate, not a result promised by this draft.

## Compute and budget gate

Propose a maximum $5 credit allowance for V4, split into at most $4 compute and $1 storage/egress reserve. This is a proposed allocation, not a current-price estimate or a claim about available credits. Check current credits, current rates, and diagnostic/pilot throughput before freezing the actual ledger. Fit inside the unspent original project budget and the owner's $0 net spend setting.

Use bounded Modal batch functions, one owned L4 worker at a time, explicit CPU/RAM/timeouts, and no automatic whole-job retry or always-on endpoint. Reserve calls before submitting them; failed calls consume their reservation. Account for startup, diagnostics, pilots, both arms, evaluation, export checks, and any recovery allowance.

If the projected calls exceed the allocation, revise exposure symmetrically or reduce scope before training; do not launch and hope. Retain enough intermediate candidates to make the checkpoint-selection comparison meaningful. Freeze their schedule and storage pruning rules. Remove temporary optimizer states and owned storage only after local artifact retention and verification.

## Deliverables and decision after V4

Deliver a diagnostic report, frozen protocol, data/solver audits, matched training trajectories, selected-model evaluations, raw predictions, plots, cost attribution, and cleanup receipts. Preserve weights and candidate-selection evidence locally. Any private model publication must include pinned revisions and fresh verification; metadata must describe both gains and failures.

If both UNKNOWN types improve without material changed-premise loss, the next justified step is targeted wording/rule-structure augmentation in a separate comparison. If the mixture fails, inspect the diagnostic interventions and learning curves before changing architecture. A scratch-model curriculum remains a separate research question; its V1 failure is already recorded.

Append every substantive step and scope change to [RESEARCH_LOG.md](../RESEARCH_LOG.md), including negative outcomes. Do not overwrite V1/V2/V3 results, frozen identities, model revisions, or archived source to accommodate the new study.
