# Decision Model Lab: research journal

Last updated: 2026-10-13. Owner: Aman / adc77.

This is the running record of what we tried, why we tried it, what happened, and what follows. The early discussion is backfilled from the conversation; study results are backed by the linked reports and evidence. We did not retain a verbatim transcript of every early research discussion or every web page consulted. Dates below identify study records, not invented timestamps for earlier messages.

**Current state:** V1–V4 are complete. V4 raised unsupported-query UNKNOWN accuracy from 0% to 81.5% while retaining 100% changed-premise UNKNOWN accuracy; both declared engineering gates passed. Depth regressed and wording/rule transfer remains weak. All declared V4 candidates are retained locally, and both selected releases passed private pinned fresh-download verification. Eleven selected study bundles are private on Hugging Face. All owned V1–V4 temporary Volumes were deleted after verification; V4's five apps are stopped with zero tasks. V4's observed owned cost is $2.32520053 at 2026-10-13T10:54:33.272419Z, with $0 workspace billed charges. Implementation, reports, compact evidence, and this journal are published on the private `feat/data-foundation` branch. A verified private Hugging Face research archive preserves raw records, historical source archives, and all retained weight paths; see [publication details](reports/PUBLICATION.md). The next step is a proposed stronger multi-entity diagnostic; no V5 GPU job has been submitted.

## Project decisions before V1

| Decision | Reason and consequence |
| --- | --- |
| Use Modal within $30 starting credits | Run bounded GPU batch experiments with explicit resource limits, call reservations, and cleanup. Avoid an always-on service. |
| Keep the project independent of Sidecar | Personal data, infrastructure, accounts, and research objective; no freight or company data. |
| Reject the music project | It did not match the research ambition. |
| Reject the coding-agent VM/sandbox project | Providing an execution environment was not the model research the owner wanted. |
| Explore digital business decisions, then reject the advertising direction | The owner chose a broader controlled study of small decision models instead of an ad-specific product. |
| Fine-tune first, then build a tiny model from scratch | Compare learned decisions, output architecture, confidence, and generalization before proposing architectural novelty. |
| Finish the agreed scope without a calendar deadline | A negative research result is acceptable; an unfinished pipeline is not completion. |
| Use exact-label synthetic facts and rules | A symbolic oracle provides labels and interventions without paid teacher-model labeling. |
| Store models privately on Hugging Face | Preserve verified weights and inference bundles with pinned revisions. Keep credentials out of chat and research artifacts. |
| Start with a reviewed GitHub plan | [The initial plan repository](https://github.com/adc77/decision-model-lab) contains the authorized plan commit. Later implementation has remained local. |
| Permit local CPU tests for this personal project | The owner explicitly granted an exception to the machine's general restriction on running project code locally. GPU experiments used Modal. |
| Set Modal net spend limit to $0 | The owner set it and supplied the billing screenshot. The intended budget is credits, with no paid compute overage. Historical reports document storage/accounting caveats separately. |

See [PLAN.md](PLAN.md) for the original V1 scope. It is a historical plan, not the status of all subsequent studies. Ideas mentioned in discussion, including other decision-model systems, are motivation rather than architectures reproduced or evaluated here.

## What the models decide

Each input contains facts, universally quantified rules, and a statement. TRUE means the statement follows; FALSE means its explicit negation follows; UNKNOWN means neither follows. The generator rejects contradictory worlds. Absence is not negation.

UNKNOWN is a correct semantic answer, not a low-confidence refusal. Deferral is a separate confidence-based decision. Temperature scaling changes confidence while preserving the predicted class.

The pretrained base is `HuggingFaceTB/SmolLM2-360M`, revision `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. V1 compares full fine-tuning with a decision token, full fine-tuning with a three-class head, and a roughly 14M-parameter transformer initialized randomly. The scratch comparison also changes size, tokenizer, and attention; it does not isolate the effect of pretraining.

## Study timeline

### V1: establish the baselines

Recorded completion: 2026-10-12 in [LOCAL_PROGRESS.md](reports/LOCAL_PROGRESS.md). Final billing observation: `2026-10-12T18:13:57.640278+00:00`.

All three models received 20,001 training questions over three passes: 60,003 presentations and 1,878 optimizer updates each, seed 17. The complete dataset contains 31,503 questions in 10,501 families. The path `data/pilot-v3` names this V1 dataset iteration; it is unrelated to the later V3 selection study.

| Model | Original held-out accuracy | Broken-support interventions correct |
| --- | ---: | ---: |
| Fine-tuned decision token | 96.60% | 0 / 142 |
| Fine-tuned decision head | 97.37% | 0 / 142 |
| Tiny scratch transformer | 33.33% | 0 / 142 |

Both fine-tuned models correctly handled all 58 direct-fact label reversals, but retained their previous TRUE/FALSE answers on every broken-support case whose oracle answer became UNKNOWN. Many wrong answers were highly confident. Wording accuracy was 89.33% for the token model and 72.93% for the head. The head showed no demonstrated inference-speed advantage.

The scratch model passed a tiny memorization gate but stayed at chance on the full study. This does not establish that scratch training cannot work; diagnosing it requires a separate experiment.

Operationally, two head calls failed/interrupted and their budget reservations were retained. A GPU gate verified an identical next CUDA update after checkpoint reload. Recovery completed the required scope. All 50 final V1 CPU tests passed; all three releases passed GPU reload and independent private-Hub CPU verification.

**Decision:** investigate missing-premise generalization with matched TRUE/FALSE/UNKNOWN worlds before making the model larger.

Evidence: [results](reports/RESULTS.md), [failure analysis](reports/FAILURE_ANALYSIS.md), [learning walkthrough](reports/LEARNING.md), [releases and costs](reports/COSTS_AND_RELEASES.md), and [compact records](reports/evidence/protocol.json). Full records: `runs/study-v1`; source archives: `artifacts/source-snapshot`.

### V2: teach changes in premise validity

Completed on 2026-10-13 local time. Billing observation: `2026-10-12T21:37:07.306268+00:00`.

Two arms shared 6,667 parent theories, initialization, optimization, and training exposure. The control asked original-style queries. The treatment held the statement fixed while changing premises to produce TRUE, FALSE, or UNKNOWN. Development and calibration inputs were shared between arms; final tests used held-out matched families.

The frozen raw-NLL selector chose update 250 for both arms. Both selected heads predicted TRUE on every counterfactual test input: 33.33% overall accuracy, 0% UNKNOWN accuracy, and 0% complete-family accuracy. The primary treatment-minus-control gain was zero. The collapsed bootstrap interval is not proof of equivalence.

Later treatment weights reached 83.26% development accuracy, versus 50.87% for the final control. Raw NLL still favored the almost uniform early checkpoint because the late head made costly confident errors. These final weights had not yet established fresh-test performance. The completed V2 result remains the negative result for its preselected early models.

All nine Modal calls completed without a failed call or retry. All 54 CPU tests passed. Both selected bundles passed private-Hub verification. Early selected and final weights were retained; intermediate weights and optimizer recovery states were not retained.

**Decision:** test checkpoint selection and calibration on new families, using the retained early/final pool. Do not retroactively change V2's selected checkpoint or claim a fresh-test result from its development examples.

Evidence: [results](reports/V2_RESULTS.md), [walkthrough](reports/V2_LEARNING.md), [releases and costs](reports/V2_RELEASES.md), [frozen protocol](docs/V2_PROTOCOL.md), and [comparison identity](reports/v2-evidence/comparison.json). Full records: `runs/study-v2`; source archives: `artifacts/v2-source-snapshot`.

### V3: repair selection and measure calibration transfer

Completed on 2026-10-13. Billing observation: `2026-10-13T05:45:50.592880+00:00`.

No new training. Four retained candidates: control/treatment at update 250 and update 1,878. Fresh selection-fit, selection, final-calibration, and test families excluded the 41,804 recorded prior canonical worlds. The study used 3,700 families and 13,600 input rows; scoring four candidates produced 54,400 prediction rows. These counts do not represent 54,400 independent questions.

Raw-NLL selection chose the early heads. Calibrated-NLL and accuracy selection chose the final heads. Choices and final-calibration temperatures were sealed before test scoring. The pool had only two checkpoints per arm and one training seed.

| Fresh suite | Final treatment accuracy | UNKNOWN accuracy | Entire family correct |
| --- | ---: | ---: | ---: |
| Original-style queries | 66.63% | 0.00% | 0.00% |
| Changed-premise worlds | 99.93% | 100.00% | 99.80% |
| Deeper proofs | 87.53% | 93.40% | 77.40% |
| Changed wording/names | 43.40% | 76.00% | 0.00% |
| Single-branch rules | 51.00% | 58.40% | 23.40% |

The selected final treatment correctly answered 2,998/3,000 changed-premise questions and all 1,000 changed-premise UNKNOWN questions. This is a strong result on that specific generated distribution. Its original-style UNKNOWN failure and weak transfer remain unresolved.

The fitted treatment temperature was 4.79268. It improved NLL on several difficult suites but worsened already excellent changed-premise NLL from 0.0049 to 0.1948 and ECE from 0.0006 to 0.1753. Calibration on one mixture need not improve every component distribution.

The [worked example](reports/V3_LEARNING.md) separates the two issues. The final head follows the three premise interventions for “Hana is large.” It incorrectly answers FALSE for “Bella is quiet.” when the available negative proof concerns Gabe. Tracking the queried entity is a plausible explanation to test, not an established internal mechanism.

Both GPU calls completed; no failed call or retry. All 59 CPU tests passed. The final audit checked all 54,400 rows and frozen hashes. All four bundles passed GPU reload and fresh private-Hub CPU verification. The owned app stopped and its temporary Volume was deleted after verification.

**Decision:** selection is no longer the immediate bottleneck. Next, diagnose query/entity transfer and train on both forms of UNKNOWN in a controlled comparison. Keep wording and rule-structure transfer visible.

Evidence: [results](reports/V3_RESULTS.md), [walkthrough](reports/V3_LEARNING.md), [releases and costs](reports/V3_RELEASES.md), [frozen protocol](docs/V3_PROTOCOL.md), [selection freeze](reports/v3-evidence/selection-freeze.json), and [full evidence inventory](reports/v3-evidence/inventory.json). Full records: `runs/study-v3`; source archives: `artifacts/v3-source-snapshot`.

## Costs and remaining-budget interpretation

These are historical observations, not live balances or final invoices. The workspace reported $0 billed at each saved snapshot.

| Study | Owned-resource metered subtotal | Observation UTC | Accounting note |
| --- | ---: | --- | --- |
| V1 | $2.72688182 | 2026-10-12 18:13:57 | Workspace metered total was $2.77152523; provider aggregation views differed. |
| V2 | $2.13302126 | 2026-10-12 21:37:07 | Owned app/Volume scope only; snapshot recorded $24.63 workspace credits remaining. |
| V3 | $0.32294453 | 2026-10-13 05:45:50 | Saved workspace metered delta was $0.33; it differs from the owned subtotal. |

Do not treat the old $24.63 balance as today's balance, add a workspace aggregate to owned app costs, or equate submission ceilings with expenditure. Billing can update after execution and cleanup. Check current credits and resource-attributed metering before the next GPU study. The precise observations, failed-call allocations, and cleanup receipts are in each release report.

V3 allowed $2 compute plus $1 storage/egress reserve. Its two submitted call ceilings totaled $0.95479616, separate from the $0.32294453 observed usage. Historical requested resources were one L4, four physical CPU cores, and 32 GiB RAM. Any future quote must recheck provider prices and actual throughput.

## Where everything is retained

| Record | Location |
| --- | --- |
| Current narrative and decision history | This journal |
| Original scope and implementation instructions | [PLAN.md](PLAN.md), [setup](docs/SETUP.md), [architecture](docs/ARCHITECTURE.md) |
| Human-readable results and limitations | `reports/RESULTS.md`, `reports/V2_RESULTS.md`, `reports/V3_RESULTS.md` |
| Compact protocols, receipts, metrics, cleanup, billing evidence | `reports/evidence`, `reports/v2-evidence`, `reports/v3-evidence` |
| All raw data, predictions, logs, and execution ledgers | Local ignored `data`, `runs/study-v1`, `runs/study-v2`, `runs/study-v3` |
| Selected/final weights and verified release bundles | Local ignored `artifacts`; private Hugging Face revisions in each release report |
| Exact historical source and final reports | Local ignored `artifacts/source-snapshot`, `artifacts/v2-source-snapshot`, `artifacts/v3-source-snapshot` |

At the end of V3, only the plan was on GitHub and the nine private Hub bundles preserved model inference artifacts. After V4, the owner authorized publication: the implementation and journal are now on GitHub, eleven selected bundles are verified, and the separate private research archive covers retained raw records and weights. See [publication details](reports/PUBLICATION.md) for exact scope and revisions.

Historical archive manifests describe their archived files. New journal/README edits do not change the archives, and are not claimed to be members of the completed V3 archive. Never silently update an old protocol hash to make changed source appear identical to what ran.

The three release reports list all nine repositories and exact verified commits. The most relevant next-study starting artifact is [V3 treatment-final](https://huggingface.co/axondendriteplus/decision-lab-v3-smollm2-360m-treatment-final), pinned at `d165bf46ed93deee9697143204f78bdccdddb17d`.

## Next study: V4 authorized, preparation started

Read [V4_PROPOSAL.md](docs/V4_PROPOSAL.md). First build controlled diagnostic pairs on fresh development families: change only the queried person, rename entities consistently, and add irrelevant facts. Then compare matched additional training from the same final-treatment weights: changed-premise data alone versus a balanced mixture of changed-premise and query/entity-varied examples.

The research question is whether this mixture recovers unsupported-query UNKNOWN accuracy while retaining premise-change accuracy. Wording and single-branch tests remain held-out transfer checks. Architecture, loss, and wording augmentation stay fixed in this comparison. A failure should guide a separate follow-up, rather than silently expanding this study.

### 2026-10-13: documentation checkpoint

- Owner requested a durable log of the project journey and the next step.
- Backfilled the choices and V1–V3 results from the conversation and verified local reports.
- Added this journal, a V4 proposal, a README entry point, and repository instructions to update the journal after substantive work.
- Checked documentation links and whitespace. No scientific source, frozen protocol, historical report, model, or source archive was changed.
- No new GPU call, model upload, Git commit, or push was performed for this checkpoint.

### 2026-10-13: V4 authorized and CPU preflight

- Owner authorized the diagnostic and matched mixed-data training comparison.
- New scientific code lives in `v4_lab`; existing `src/decision_lab` files remain unchanged. Both arms will start from the pinned V3 treatment-final weights with fresh, matched optimizers.
- The current billing snapshot reports $6.20106047 workspace metered usage and $0 billed. A conservative subtraction from the original $30 leaves $23.79893953 before further adjustments. This is a budget gate, not an exact live credit balance.
- API rates are L4 $0.80/hour, CPU $0.0473/core-hour, and RAM $0.008/GiB-hour, slightly higher than rounded per-second web rates. The V4 ledger uses the API rates and a $4 compute ceiling plus $1 reserve.
- All 64 CPU tests passed in 25.78 seconds, including controlled diagnostic invariants, matched data order, exact recovery, and the complete selection/calibration/scoring/export path on tiny models.
- Local parent preparation initially assumed separate `vocab.json`/`merges.txt` files. The verified release instead contains the complete fast tokenizer in `tokenizer.json`; preparation was corrected to its actual required files. No GPU call was submitted for this failed preparation. The incomplete local hardlinks were removed without changing the source bundle.
- Current evidence: `runs/study-v4/billing-before.json`, `billing-rates.json`, `cpu-tests-preflight.log`, and the diagnostic preparation/audit records. Diagnostic inference and training outcomes are pending.

### 2026-10-13: V4 controlled diagnostic result and throughput adjustment

- Diagnostic protocol: `593025f62c7668bb77c4d89e1374c749d57d889a4e9e7792699df6b22465bfc3`. All 1,600 inputs across 200 families passed formal and tokenizer checks; 52,404 prior canonical worlds were excluded.
- The parent scored 200/200 supported TRUE, 200/200 supported FALSE, and 200/200 changed-premise UNKNOWN correctly. It failed all 400 query-entity-only UNKNOWN cases and all 200 renamed UNKNOWN cases, retaining its old predictions. All 600 errors had raw confidence at least 90%. The supported renaming and distractor conditions remained 200/200 correct each.
- This establishes the controlled behavioral gap; it does not prove an internal entity-binding mechanism. See `reports/V4_DIAGNOSTICS.md`, `runs/study-v4/diagnostics.json`, and raw logits.
- Diagnostic call completed in 51.97 client seconds. The subsequent discarded 20-update pilot measured 1.344517 seconds/update after warmup.
- Revised future training/recovery timeouts from 950 to 1,050 seconds, selection from 900 to 600, and final evaluation from 1,200 to 800. Exposure, candidate steps, and all scientific comparisons remain unchanged. The full reservation ceiling stays $3.98879067, including recovery; prior call allocations are preserved.
- Before-diagnostics source archive: `artifacts/v4-source-snapshot/source-before-diagnostics.tar.gz`, SHA-256 `40433308b02e8e6844d18ab07cb4b54213cdf000760daf3e99eb25dd93875713`; all 211 members verified.
- The additional-training protocol is now written in `docs/V4_PROTOCOL.md`; final training-data generation and freeze are the next step. Training and final tests have not yet run.

### 2026-10-13: V4 additional-training freeze

- Training protocol identity: `2ab1532953c1f8b4c2dfaab8d1b3a49ddc39a32cbeeff5e5413e6ed55a631650`.
- Formal/data audits passed for 10,367 parent families, 53,602 input rows including both training arms, and 30,601 unique canonical worlds. Excluded 53,204 prior and diagnostic worlds. Maximum tokenized input length is 346; no truncation.
- Each arm has 20,001 questions, balanced labels, identical family/label order, and three passes planned. Treatment uses 3,334 original-style and 3,333 changed-premise families; control uses changed-premise examples throughout.
- The frozen five-candidate schedule includes the shared initial model and updates 250, 750, 1250, and 1878 per arm. Selection-fit, selection, final calibration, and tests use distinct families.
- Primary endpoints, paired bootstrap, engineering gates, analysis-script hash, parent-file hashes, input hashes, and scientific source hashes are sealed before training. No final-test predictions have been inspected.
- `runs/study-v4/training-protocol.json`, `training-audit.json`, `training-preflight.json`, and `training-preparation.log` record the freeze. The next operation is bounded GPU training, followed by frozen selection and test scoring.

### 2026-10-13: V4 GPU training and retention preparation

- Started the matched training controller at approximately `2026-10-13T08:00:44Z`. Owned app: `ap-WeSjCGe4mvRMFwdwZzEQo0`; owned Volume: `decision-model-lab-v4-runs`, ID `vo-01ehjGaSgNFao6icxzq3Yl`.
- Before-training source archive: `artifacts/v4-source-snapshot/source-before-training.tar.gz`, SHA-256 `1299374ecf587f64c1b4838cab5706952038570658296288c7ba949d2c5f39c1`; all 214 members verified. Later retention/reporting helpers are preserved in the eventual final archive and are not claimed to be members of this earlier archive.
- Control reached update 400 with finite losses and a durable update-250 candidate. Final-test predictions remain uninspected; low training loss does not establish transfer accuracy.
- Retention helpers verify atomic downloads and hashes, preserve the complete declared candidate pool, and hardlink selected local weights to avoid unnecessary duplicate storage. Two additional CPU operations tests passed in 0.90 seconds.
- At `2026-10-13T08:05:13.070521Z`, selected `v4-test-000000` by lexicographic family order for the later descriptive walkthrough, before final-test inference. The example-selection record is local and distinct from the primary statistical endpoint.
- Current raw evidence: `runs/study-v4/modal-training.log`, `allocation-ledger.json`, `modal-ownership.json`, and per-call progress records. Training, checkpoint selection, final tests, publication, and V4 cleanup are still in progress or pending.

### 2026-10-13: V4 interpretation limit noted before final tests

- Source inspection of `src/decision_lab/v2/data.py` identifies a concrete remaining cue: `parent_world` places a unique selector predicate on the query's rule target, and noise excludes that selector predicate. Original-style supported TRUE/FALSE queries concern this target; unsupported UNKNOWN queries concern another entity.
- Consequently, a query-to-selector entity match combined with the existing premise heuristic could solve much of V4 without general proof binding. This is a possible shortcut, not a mechanism established inside the model.
- The current frozen comparison is unchanged. It tests whether the training mixture closes the measured behavioral gap, while wording/single-branch suites check some transfer. A later stronger test would place multiple independently answerable entities with different labels in the same world and remove the distinguished-target cue.
- This caveat was recorded during control training, before model selection or final-test predictions. The V4 primary endpoints and engineering gates remain as frozen.

### 2026-10-13T08:53:50.248058+00:00: V4 control training complete

- All three control chunks completed the frozen 60,003 presentations and 1,878 updates. Final weights and declared intermediate candidates are durable on the owned Volume; temporary recovery state is retained until verified release.
- Treatment training has started and reported finite loss at update 100. No held-out selection or final-test predictions have yet been scored.
- Exact chunk durations and reservations are retained in `runs/study-v4/training-call-control-*.json` and `allocation-ledger.json`.

### 2026-10-13T09:23:09Z: V4 treatment second pass complete

- Treatment completed 40,002 presentations and 1,252 updates; the final pass remains. Control already completed all 60,003 presentations and 1,878 updates. Training losses are not held-out generalization estimates.
- The finish controller waits for both completed trajectories, then runs the frozen selection, final calibration, final tests, raw/candidate retention, statistical analysis, audit, and private publication with fresh CPU verification. Cleanup and resource-attributed billing follow verification.
- All 66 CPU tests passed before final scoring. Current lint and formatting checks pass for 106 Python files; the dependency lock and Git whitespace checks pass. Later operational helpers do not alter the frozen scientific source or analysis script.
- README now accurately marks V4 as running. No implementation commit or push has been made.

### 2026-10-13T09:38:25.934465+00:00: V4 execution milestone

Both training arms completed the frozen exposure. Per-call records and the invocation ledger confirm completion; checkpoint selection and untouched final tests follow.

### 2026-10-13T09:46:54.478929+00:00: V4 execution milestone

Completed **checkpoint selection**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T09:53:38.279968+00:00: V4 execution milestone

Completed **final test scoring and GPU export verification**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T09:55:55.185928+00:00: V4 execution milestone

Completed **raw record retention**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T10:05:02.763047+00:00: V4 execution milestone

Completed **complete candidate retention**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T10:06:15.977355+00:00: V4 execution milestone

Completed **selected release retention**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T10:06:17.458021+00:00: V4 execution milestone

Completed **frozen statistical analysis and plots**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T10:06:18.072581+00:00: V4 execution milestone

Completed **numerical learning walkthrough**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T10:06:34.732839+00:00: V4 execution milestone

Completed **scientific and retention audit**. The exact command and output are retained in `runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. No Git commit or push is part of this controller.

### 2026-10-13T10:13:37.088954+00:00: V4 publication transfer adjustment

- The initial private upload was progressing at roughly 0.7 MB/s. Terminated only its identified owned Python process; the finish controller stopped on that subprocess exit. This was an operational interruption, not a failed GPU call, changed checkpoint, or scientific rerun. The initial log and private creation receipt are preserved.
- Resumed `v4_release.py` with `HF_XET_FIXED_UPLOAD_CONCURRENCY=4`, retaining the receipt and exact release inventory. Installed `hf-xet` is 1.7.0. The [official Xet guide](https://huggingface.co/docs/hub/xet/using-xet-storage) documents fixed upload concurrency; no credentials were printed or sent to Modal.
- Publication, pinned fresh CPU verification, resource cleanup, final billing, and archival still must complete. All ten GPU calls succeeded without a retry; the completed 48,000-row audit and frozen scientific files are preserved.

### 2026-10-13: V4 completed, verified, and closed

**Request and hypothesis.** The owner authorized the V4 comparison after V3 exposed two forms of UNKNOWN with different behavior. Controlled diagnostics confirmed that the V3 parent correctly detected missing premises but failed all 600 unsupported-query/renamed-UNKNOWN cases, with raw confidence at least 90%. V4 tests whether a matched training mixture closes that gap while retaining premise sensitivity. It does not introduce a new architecture or prove an internal entity-binding mechanism.

**Frozen execution.** Both arms started from the pinned V3 treatment-final bundle, used fresh AdamW optimizers, seed 317, the same family/label order, and completed 60,003 presentations and 1,878 updates. Control used changed-premise examples; treatment used 3,334 original-style and 3,333 changed-premise training families. Ten GPU calls completed without a failed call or retry. The shared initial checkpoint plus eight trained checkpoints are all retained. Calibrated-NLL selection chose update 1,878 in each arm. Final temperatures were 6.034315882573097 for control and 1.1300427719087858 for treatment; selection and calibration were sealed before final tests.

| Fresh outcome | Control | Mixed-data treatment |
| --- | ---: | ---: |
| Unsupported-query UNKNOWN | 0 / 1,000 | 815 / 1,000 |
| Original-query overall | 2,000 / 3,000 | 2,812 / 3,000 |
| Original-query complete families | 0 / 1,000 | 814 / 1,000 |
| Changed-premise UNKNOWN | 1,000 / 1,000 | 1,000 / 1,000 |
| Changed-premise overall | 3,000 / 3,000 | 2,997 / 3,000 |
| Depth overall | 1,385 / 1,500 | 1,200 / 1,500 |
| Wording overall | 634 / 1,500 | 638 / 1,500 |
| Single-branch overall | 526 / 1,000 | 516 / 1,000 |
| Single-branch supported TRUE/FALSE | 240 / 500 | 59 / 500 |
| Single-branch UNKNOWN | 286 / 500 | 457 / 500 |

The primary unsupported-query UNKNOWN gain is 81.5 percentage points, family-bootstrap 95% interval [79.1, 83.8025]. The changed-premise UNKNOWN gain is zero, with a collapsed empirical [0, 0] interval. Both predeclared engineering gates passed. Intervals concern test-family sampling for these fixed models, not training-seed uncertainty; a collapsed interval is not proof of perfect population performance. Transfer comparisons are descriptive and not multiplicity-adjusted.

**Negative findings and interpretation.** Depth accuracy drops 12.33 percentage points. Wording remains weak, with zero completely correct wording families for both arms. The mixed model predicts UNKNOWN on 849/1,000 single-branch questions although only 500 oracle labels are UNKNOWN; it correctly answers only 59/500 supported statements. It still makes 745 wording and 460 single-branch errors with calibrated confidence at least 90%. Original-query UNKNOWN errors occur for both positive statements (85/489) and negative statements (100/511), so negation alone does not explain the gap. These label counts were computed after testing and reconciled with the frozen aggregates; they did not change selection or the primary endpoints.

The preselected example `v4-test-000000` remains a failure: the treatment answers TRUE to unsupported “Evan is not verified.” with 98.34% calibrated confidence, while correctly detecting the changed-premise UNKNOWN about Hana. Its selection happened before inference; it was retained rather than replaced with a success. The distinguished-selector target cue was already recorded before final testing. V4 therefore establishes a targeted behavioral improvement with important transfer failures, not a general decision engine.

**Verification and operational record.** All 69 final CPU tests passed in 26.20 seconds. The three added regression cases check direct lookup of every owned app and rejection of an active app or remaining task. Lint, formatting of 108 Python files, dependency lock, and whitespace checks passed. The final audit verified 48,000 prediction rows, source/protocol and selection identities, full training exposure, candidate retention, selected hashes, and unchanged prior scientific files. Each selected bundle passed 8/8 GPU reload checks and 8/8 checks from an independent pinned private-Hub download in a separate CPU process. Aggregate scoring uses BF16 autocast; artifact references use FP32.

The private upload had one deliberate operational interruption and resumed from its receipt with four upload streams, as recorded in the preceding transfer entry. Initial cleanup successfully deleted the owned Volume but its recent-app-list verifier missed the older diagnostic/pilot apps. The corrected direct lifecycle/task check verified all five stopped with zero tasks and the Volume absent at 2026-10-13T10:45:21.411650Z. Initial helper source and failure log are retained; no earlier stop/delete stdout receipts are invented. A mistaken `pytest` console-entry-point test launch failed collection because root V4 modules were not on its import path; the documented `python -m pytest` invocation passed all 69 tests. The initial month-to-date hourly billing query exceeded Modal's seven-day limit; the corrected report uses V4's execution-day range. None of these operational corrections reran training, selection, or final inference, or changed frozen scientific/analysis source.

**Private releases.** `axondendriteplus/decision-lab-v4-smollm2-360m-control` at `f6fb417b7a7851fa3492412490ee7e39afa35815`; `axondendriteplus/decision-lab-v4-smollm2-360m-mixed-unknown` at `c4ee886cd3a718036297edb56b0202b590457cf2`. Private flags, pinned commits, complete inventories, and fresh CPU reference outputs were verified. Local selected bundles use hardlinks to matching retained candidates. Recovery optimizer state was discarded with the working Volume after verification. No unrelated workspace resource was modified and no implementation commit or push was made.

**Costs.** At 2026-10-13T10:54:33.272419Z, resource-ID-attributed V4 usage is $2.32520053: L4 $1.49386081, CPU $0.35330427, memory $0.47803545. All five owned apps appear in the report; no additional owned storage/egress rows appear in this observation. It is not a final invoice. Workspace billed charges are $0. Workspace gross rose from $6.20106047 to $9.89950463, a $3.69844416 increase that includes other workloads and is not V4's exact cost. The conservative original-$30 margin is $20.10049537, not a live credit-wallet query. Submitted compute ceilings total $3.5626555555555557; the full ceiling including unused recovery remains $3.988790666666667, plus the $1 storage/egress reserve.

**Retained evidence.** Read [results and post-test interpretation](reports/V4_RESULTS.md), [the preselected numerical walkthrough](reports/V4_LEARNING.md), [verified releases and costs](reports/V4_RELEASES.md), and [the frozen protocol](docs/V4_PROTOCOL.md). Full inputs, logits, training records, failed operational logs, and all candidates remain in `runs/study-v4`; selected private downloads and releases are in `artifacts/v4-*`. Compact evidence is indexed in `reports/v4-evidence/inventory.json`. The closure step saves the final source/report archive and independently verifies its members, then records its SHA-256 in `artifacts/v4-source-snapshot/manifest-final-with-reports.json`. That external manifest avoids embedding a circular archive hash in this journal. The pre-execution archives and all V1–V3 evidence remain preserved. Bulk intermediate weights and raw runs are local and ignored by Git; the GitHub plan is not their backup.

**Next proposed step, not submitted.** Generate fresh worlds with several independently answerable entities, different TRUE/FALSE/UNKNOWN labels in the same world, and no unique selector that singles out the target. Use entity permutations and interventions that change one entity's support while preserving others. Begin with oracle/data audits and a frozen diagnostic of the retained models, keeping wording and rule-structure checks. Review that result before another training curriculum or architecture change. No V5 data freeze, GPU submission, or release is claimed.

## How to update this journal

Append a dated entry after each substantive work session or experiment. Preserve previous outcomes; label corrections and new interpretations explicitly. Promote a proposal to started/completed only when execution evidence supports it. Update the current-state paragraph as the project advances.

Each entry should capture:

1. The request, hypothesis, and reason this step follows the previous result.
2. What changed and what stayed fixed; data partitions, seeds, parent weights, and frozen identities.
3. Execution status, call IDs, failures, retries, tests, and evidence paths.
4. Main results and negative results, with denominators and uncertainty limits.
5. Observed costs with timestamp and scope, separate from budget reservations.
6. Retained artifacts, private pinned releases, and cleanup evidence.
7. Supported interpretation, unresolved questions, and the next proposed step.

Never log credentials, authentication links, or secret values. A plan, submitted call, completed run, verified artifact, and published release are distinct states; record the one actually reached.

### 2026-10-13T12:18:24.027514+00:00: remote publication authorized and checked

The owner requested pushing the implementation and research history to GitHub, with Hugging Face uploads where needed. The existing GitHub repository is private and its remote contains only the original plan. All eleven selected private model repositories were rechecked at their exact recorded revisions, including fresh manifest downloads and SHA-256 checks. The current CPU suite passed all 69 tests in 26.99 seconds; lint, formatting of 134 Python files, dependency lock, and whitespace checks passed. Publication will use the existing feature branch, preserve historical source archives, and separately back up raw study records and weights not already represented by pinned releases. No new GPU experiment is authorized or submitted by this publication step. Completion receipts will be appended after verified transfers.

### 2026-10-13T12:36:32.616030+00:00: GitHub implementation pushed; bulk backup commit retry

The full V1–V4 implementation, protocols, tests, reports, compact evidence, and journal were committed as `c22b5f005751c68b313472aa46855a4ef4b69f3a` and pushed to the private `feat/data-foundation` branch. Both Git identities are `adc77 <amandogra2016@gmail.com>`. Remote branch SHA was checked. Generated SVGs retain their historical bytes; `.gitattributes` excludes their generated path whitespace from Git whitespace warnings. All 106 archived scientific/build files checked against the V4 final source manifest remain unchanged.

The private bulk backup stages 1,457 non-cache records in a 115,953,564-byte gzip archive and maps all 41 retained local weight paths. Eight unique blobs require new upload: six V4 intermediate heads, the retained V1 interrupted head, and the original scratch checkpoint representation. Other weights map to exact selected model revisions whose remote weight SHA-256 metadata was also verified. Raw datasets, logits, failed operational logs, and all nine historical source archives are included. Previously discarded optimizer states and V2 intermediates cannot be reconstructed and are not claimed as backed up.

The initial Hugging Face data transfer reached 100%, but the final commit request failed with `requests.exceptions.ConnectionError` / `BrokenPipeError`. Direct inspection found only the initial `.gitattributes` commit, so no successful backup commit was claimed. Resumed the exact staged files using the private repository ownership receipt and four Xet upload streams; initial and resume logs remain separately retained in `runs/publication-2026-10-13`. Final verification is pending. No GPU calls were made. [The next diagnostic proposal](docs/NEXT_EXPERIMENT.md) is written, without freezing or submitting a new study.

### 2026-10-13T12:38:26.792387+00:00: remote research backup verified and publication completed

The private Hugging Face research archive `axondendriteplus/decision-lab-research-archive` is committed at `06c0fb2050f4727d84f6873445794df21f416134`, verified at 2026-10-13T12:37:02.577139+00:00. The resumed transfer reused all previously uploaded content and reported zero new upload bytes. The archive manifest SHA-256 is `87815f401d9a170bd7691c427f2b670f47ef78daeebc702d4ec862dca2a599e0`. A fresh pinned download of `records.tar.gz` passed the bundle checksum and all 1,457 member hashes. All eight newly uploaded weight blobs passed pinned remote LFS size/SHA-256 checks; other retained weights map to selected releases whose pinned manifests and remote weight hashes were checked. This is remote metadata verification of the additional weights, not a claim of fresh inference tests for intermediate checkpoints. All 41 retained weight paths are represented. Eleven selected inference bundles retain their prior fresh-download CPU verification.

GitHub holds the V1–V4 source and compact evidence on `feat/data-foundation`, starting with implementation commit `c22b5f005751c68b313472aa46855a4ef4b69f3a`. This publication update adds the backup utility, complete archive manifest, checksum receipts, summarized failure/retry evidence, and [next diagnostic proposal](docs/NEXT_EXPERIMENT.md). Original upload logs remain local; their hashes and failure traceback are recorded in publication evidence. Earlier scientific files, frozen outcomes, source archives, and historical closure receipts are preserved. The archive excludes SDK caches, compiled Python, and the publication operation itself; the latter has compact evidence in Git. No GPU calls, new training, model selection, or V5 experiment occurred. Git authorship remains the owner identity.

Next: build and oracle-audit fresh multi-entity worlds, then freeze and run an inference-only comparison of retained models. Review query/entity consistency, interventions, and confident errors before choosing another training curriculum or an architecture change.
