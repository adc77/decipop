# Decision Model Lab: research journal

Last updated: 2026-10-15. Owner: Aman / adc77.

This is the running record of what we tried, why we tried it, what happened, and what follows. The early discussion is backfilled from the conversation; study results are backed by the linked reports and evidence. We did not retain a verbatim transcript of every early research discussion or every web page consulted. Dates below identify study records, not invented timestamps for earlier messages.

**Current state:** V1–V8 scientific execution and owned-resource cleanup are complete. The project is now Premise: `https://github.com/adc77/premise` is public with the implementation on `main`. All eight selected V7/V8 model packaging copies and the four-config `axondendriteplus/premise-worlds` dataset are public at verified pins; 190,688 serialized dataset rows are not independent worlds. The public release passed 126 CPU tests, 64 fresh model-reference inputs, four dataset-loader configurations, a clean-checkout article rebuild and the documented inference command with implicit HF authentication disabled. V7’s curriculum gain replicated; V8’s added-loss advantage did not pass its declared two-run check, and transformation reliability remains weak. No further experiment or GPU job was submitted. Original scientific identities, archived sources, historical private bundles and failures remain preserved. The local article at `http://localhost:8765` uses the owner’s site style, simpler explanations, an actual post hoc held-out support-removal case, and public GitHub/HF links. Public release manifests and checks are in `release/`; this publication does not deploy the personal website.

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

### 2026-10-13T14:08:31.734985+00:00: V5 multi-entity diagnostic prepared, audited, and frozen

The owner authorized the next diagnostic and then requested continuation. Built separate `v5_lab` scientific code, a configuration, frozen protocol, and tests without editing V1–V4 scientific files. Diagnostic data contains 480 parent families (288 conjunction, 96 deeper, 96 signed-chain), nine variants and three queries each: 12,960 questions. A separate 36-family pilot contains 972 questions. Every base world contains one TRUE, one FALSE, and one UNKNOWN query about the same predicate; all entities have three explicit facts. Base names and query slots are exactly label-balanced; polarity is balanced. Canonical exclusion covers 83,805 recorded prior worlds. Independent synchronous closure and production forward chaining agree; serialized prompts, labels, intervention locality, and hashes passed validation. Maximum prompt length is 355 tokens for each model.

Preparation caught and corrected three issues before any GPU call: YAML initially interpreted unquoted TRUE/FALSE class names as booleans; strict tuple schemas require JSON-aware deserialization for saved arrays; and canonical world hashes cannot key ground-name closure caches across renaming. Added a serialization/corrupted-prompt regression and resumed the same retained generated data with explicit `--resume`; data was not replaced. Initial sources and failed logs remain under `runs/study-v5`. The final CPU suite passed 99 tests in 29.86 seconds; lint and formatting passed. Protocol identity is `6038ac97693f50a8392825714fe48978e0072a3cee5ad05747290fb01bb4d4d8`.

Three unchanged pinned models: V3 treatment-final, V4 control, V4 mixed. Stored temperatures are reused; no fitting, selection, training, or scientific endpoint inspection occurred. Current billing observation is $11.19950463 gross workspace metering and $0 billed, leaving a conservative $18.80049537 original-budget margin, not a live wallet query. API and published prices support conservative ceilings. Total worst-case reserved compute, including an unused recovery call, is $1.1227553333333333 under the $1.50 allocation; a separate $0.50 reserve covers storage/egress. Next: archive the source, run the bounded timing/reference pilot, then submit full inference only if the timing gate passes. No new code commit or remote publication is part of this step.

### 2026-10-13T14:12:42.764083+00:00: V5 pilot completed; full diagnostic submitted

Owned pilot app `ap-FOFkYK0qefMDfnRu3w8HM4` completed without retry. All three retained releases passed 8/8 fixed GPU reference checks. On separate pilot inputs, throughput was 92.3358, 93.5527, and 92.8401 questions/second for V3 parent, V4 control, and V4 mixed. Pilot logits and scientific endpoint metrics were not retained or inspected. The timing gate supports the frozen full scope with a 1.5x margin and 90-second overhead. The before-inference source archive contains 305 independently checked members, SHA-256 `f7e0b996b03a69160cf71180ddc7608c869e0908148ca7ace00e7d41c13dda22`. All three local candidate manifests and weight hashes were also checked against their exact private remote revisions. Full diagnostic inference is now being submitted; completion and results are not yet claimed. No training or new model upload occurred.

### 2026-10-13: V5 completed, verified, and closed locally

**Request and execution.** The owner authorized the multi-entity diagnostic and requested continuation. Pilot `ap-FOFkYK0qefMDfnRu3w8HM4` and diagnostic `ap-aBLUIfWgZGT494imqfyB7J` both completed without a failed GPU call or retry. Three unchanged pinned heads were scored: V3 treatment-final, V4 control, V4 mixed. Their saved temperatures were reused; there was no training, selection, calibration fitting or new release. Main inference took 131.9000, 128.7032 and 128.3953 seconds respectively. Full client times, reference checks and call reservations remain in the execution records and allocation ledger.

**Frozen scope.** Protocol identity remains `6038ac97693f50a8392825714fe48978e0072a3cee5ad05747290fb01bb4d4d8`: 480 diagnostic parent families, nine variants, three entity queries per variant, 12,960 questions per model and 38,880 retained prediction rows. The separate pilot has 36 families and 972 questions per model. Serialized audits covered 13,932 questions, 516 families and 2,580 unique canonical worlds, with zero overlap against the recorded prior-world exclusion set. Labels, entity-name/slot balance, polarity, equal explicit fact counts and intervention locality were checked. The ordinary, deeper and signed-chain suites are distinct generated conditions; signed chains do not reproduce the earlier single-branch generator.

| Base suite | V3 parent accuracy | V4 control accuracy | V4 mixed accuracy | Complete worlds, every model |
| --- | ---: | ---: | ---: | ---: |
| Conjunction | 288/864 | 289/864 | 291/864 | 0/288 |
| Deeper | 96/288 | 96/288 | 96/288 | 0/96 |
| Signed chain | 97/288 | 97/288 | 96/288 | 0/96 |

**Failure and interpretation.** All three models get zero complete base worlds across all 480 families. On conjunction worlds the mixed head predicts UNKNOWN 860/864 times, including TRUE recall 3/288, FALSE recall 0/288 and UNKNOWN recall 288/288. There are 567/864 errors with at least 90% confidence after its saved temperature (568 before temperature scaling). It changes its prediction in only 8/864 different-entity query pairs; just 3/864 pairs have both answers correct. After repairing the originally UNKNOWN entity's support, it changes that target's answer in 0/288 conjunction families. High renaming or unaffected-entity consistency therefore does not establish correctness. UNKNOWN is the model's semantic class, not a confidence-based refusal.

The primary complete-world estimate is 0%, with empirical parent-family bootstrap interval [0%, 0%]. This is a fixed-model result on generated families, not proof of population-wide failure or equivalence, and not training-seed replication. Secondary comparisons are descriptive without multiplicity adjustment. All three labels co-occur in base worlds; rule templates, filler choices and missing branch facts may still provide cues. The independent closure audit shares parsing and grounding utilities with the production solver. Temperature transfer need not calibrate this distribution. The preselected first-family failure is retained in [the results](reports/V5_RESULTS.md), including its support edits; no successful example replaced it.

**Verification.** All 99 CPU tests passed in 29.86 seconds before GPU execution. Final scientific verification reconciled all 38,880 raw prediction rows with oracle labels and identities, checked finite three-class logits and regenerated the same frozen aggregates. An independent NumPy aggregation of base raw rows matched all nine model/suite confusion matrices, constant-prediction counts and complete-world counts. All 106 archived historical scientific/build files remain unchanged. All models passed 8/8 pilot GPU reference checks, and pinned remote release manifests/weight hashes matched local retained bundles. New operational helpers and documentation are outside the frozen scientific package; their final source is archived separately.

**Operational corrections.** The three pre-inference preparation defects and their failed logs/source snapshots are recorded above and preserved. Initial cleanup failed when the CLI was asked to stop an already-completed app; no Volume deletion occurred. A direct-check helper then hung after calling an async SDK client on a different event loop. Terminated only its identified owned local Python PID 199 with SIGTERM, preserving its source and warning log. Running those checks on Modal's SDK loop succeeded. At 2026-10-13T14:33:50.402949Z both owned apps were directly verified stopped with zero tasks and the owned Volume `decision-model-lab-v5-runs` was verified absent after deletion. Predictions had passed checksum verification before deletion. No unrelated workspace resources were modified. None of these corrections reran inference or altered the scientific freeze.

**Costs.** At 2026-10-13T14:34:11.317978Z, owned resource metering is $0.21995289: L4 $0.13866666, CPU $0.03625556, memory $0.04443078, egress $0.00059989. Both owned apps appear in the scoped report; no owned Volume charge row appears in this observation. Workspace billed charges remain $0. The first report defaulted to now and omitted the current hour's incomplete bucket, showing zero owned rows; that is preserved as an incomplete observation, not free compute. The corrected explicit UTC-day range includes the execution hour. These figures are observed metering, not a final invoice. Workspace gross is $11.66010451, leaving a conservative $18.33989549 against the original $30 budget, not a live credit-wallet query. Gross changes include other workloads and later reconciliation, so they are not V5's exact cost.

The two submitted compute ceilings total $0.6447368888888889. An unused recovery ceiling leaves the original full reserved maximum at $1.1227553333333333; a separate $0.50 storage/egress reserve was not a spend. No recovery call was submitted.

**Retained evidence and publication state.** [V5 results](reports/V5_RESULTS.md), [the frozen protocol](docs/V5_PROTOCOL.md), plots and [compact evidence](reports/v5-evidence/inventory.json) preserve the findings and operational corrections. Full generated inputs, proof records, logits, timing logs, failed preparation/cleanup records and selected model hardlinks remain in `runs/study-v5`. The before-inference source archive remains unchanged. Final source, reports and compact evidence are saved to `artifacts/v5-source-snapshot/source-final-with-reports.tar.gz`, with independent member hashes in `manifest-final-with-reports.json`; the external manifest and `runs/study-v5/closure-receipt.json` avoid a circular archive hash here. Raw study records are retained locally, not included in this source archive. No new Git commit/push or Hugging Face upload occurred. V1–V4's earlier remote publication and backup remain intact; they do not include V5.

**Next proposed step, not started.** [V6](docs/V6_PROPOSAL.md) compares matched continuation on the existing curriculum with a mixture containing independently answerable multi-entity worlds. Vary entity count and world label composition, preserve fresh split boundaries, and keep earlier behavior/transfer checks. Use the same initialization, question exposure and selection/calibration procedure in both arms; report any token-exposure difference. V5's opened diagnostic becomes historical evidence rather than the new confirmatory test. Recheck budget and freeze the new protocol before any submission. An architecture experiment remains a separate hypothesis and baseline.

### 2026-10-13T15:16:04.905722+00:00: V6 implementation and CPU verification

The owner authorized the proposed matched multi-entity training experiment. Built separate V6 scientific code and a protocol; V1–V5 scientific sources remain preserved. Both arms use the pinned V4 mixed head, fresh optimizers and identical label streams. Each receives 12,096 training questions over three epochs: 36,288 presentations and 1,134 updates. Treatment shares half the control examples at identical positions and replaces half with multi-entity worlds. Variable label compositions and three-to-five entities avoid requiring all three labels in every world. Whole worlds and transformations stay in their partition; V1–V5 canonical worlds are excluded. A three-world label-rotation block is the multi-test bootstrap unit, stratified by entity count. Primary is complete-world treatment-minus-control; retention checks compare UNKNOWN behavior with the unchanged parent on fresh old-style tests.

Five new CPU tests passed in 16.30 seconds, including local-edit locality/counts, serialized integrity, matched label exposure, blocked bootstrap, exact resumed-vs-uninterrupted training, selection and export. After import/style fixes, the full suite passed all 104 tests in 39.06 seconds. Initial and final logs are retained. Current workspace gross observation is $11.66010451 and billed charges $0; the conservative original-budget margin is $18.33989549. Maximum reserved compute is below $4 including one unused recovery, plus a $0.50 storage/egress reserve. Prices were checked against the Modal billing API and official pricing page.

Local storage has about 12 GiB free. Selected bundles will remain local; additional scheduled weights will be transferred one at a time to private Hugging Face storage and verified by pinned remote size/SHA metadata before deleting only their owned staging copies. This preserves candidate weights without filling the laptop. The owner previously specified Hugging Face for model storage; existing account authentication is reused without exposing credentials. Git implementation remains local. Data preparation/auditing and source archival are in progress; no GPU call has been submitted.

### 2026-10-13T15:20:54.798962+00:00: V6 data frozen and pilot submitted

Protocol identity `e9bcc7639951e6b2f017b19e938e716d93b6975abb63e7165ee212fa3835b6c5` covers 45,080 serialized questions, 6,996 families and 14,366 unique canonical worlds; 86,385 recorded prior worlds were excluded. Production and synchronous oracle audits, serialized rendering/labels, paired training exposure and token limits passed. Primary name/slot label counts are exactly balanced and world label compositions vary, including single-label worlds. Input tokens per epoch are 2,538,473 for control and 2,651,685 for treatment, a 4.46% difference; exposure is matched by questions and labels, not tokens. Maximum recorded prompt length is 355 tokens. All 104 CPU tests passed. The before-inference source archive SHA-256 is `82b26c0d92d7db3ecd7806e2825ee5811cd7529584aeb98e2dbd0c7519f61f20`, with 370 independently checked members. The bounded timing/reference pilot is submitted; no training outcome or final-test score is claimed.

### 2026-10-13T15:21:21.088380+00:00: V6 pilot CLI correction

The preceding entry prematurely said submitted: the first launch rejected `--profile` after the app-specific arguments before creating any app or GPU call. No ledger reservation or GPU execution occurred. Preserved `pilot-cli-initial.log` and relaunched with the Modal option before the app path. This changes only the CLI invocation; frozen code, data, model and budget are unchanged. Submission is confirmed only by the owned app and call ledger.

### 2026-10-13T15:30:38.739627+00:00: V6 pilot passed and training started

Pilot app `ap-vyuVWtpW4DrEZCfK1bRkfx` completed without a failed GPU call or retry. The pinned parent passed 8/8 fixed GPU reference checks. Worst-length batches measured 1.28398998475 seconds/update for control and 1.2592967815 for treatment (280 and 279 tokens maximum). Frozen timing projections are 658.1504463708251 and 647.4163109180495 seconds per chunk, below the 1,020-second gate. Discarded pilot weights are not training candidates. Training app `ap-9gdczWwiUjEKBaNCTtTz5G` has submitted control chunk one; update 100 is logged. No final-test predictions have been inspected. Local source/data identities remain frozen. The earlier CLI-option failure created no GPU reservation; it remains a separate operational record.

### 2026-10-13T15:46:51.325114+00:00: V6 packaging failure and explicit source amendment

Failed control call `control-1` retained its $0.42613511111111113 reservation. It saved update 250 (8,000 presentations) before candidate export failed on the nested parent `code/` directory. The initial flat CPU fixture missed that layout. The remote marker/progress and partial candidate contents were retained and inspected; the worker must still verify the complete weights/optimizer/progress hashes before actual resume. Original source copies and failed logs are preserved.

The original frozen protocol `e9bcc7639951e6b2f017b19e938e716d93b6975abb63e7165ee212fa3835b6c5` remains unchanged. Active amended protocol is `099611c7f4eb70c9bc12f2fd44d0279c5bfc81e0cf0913b6e5d5e0035b3f44af`, linked to the original identity and file SHA. Only the V6 study/protocol source files changed: flat asset packaging, explicit checkpoint-source identity during state restore, and amendment verification. Data, model, optimizer/update math, schedule, exposure, selection/calibration and endpoints are unchanged. After nested-layout and deliberately interrupted-packaging regression coverage, all 105 CPU tests passed in 47.65 seconds; resumed final weights match uninterrupted training exactly and partial assets are preserved. The single reserved recovery will restore the complete original state; no restart, data regeneration, new endpoint or unreserved GPU call is planned.

### 2026-10-13T15:52:04.853824+00:00: V6 reserved recovery submitted

Recovery app `ap-7wZTkDyHcF421G32QL2eeq` is submitted under amended protocol `099611c7f4eb70c9bc12f2fd44d0279c5bfc81e0cf0913b6e5d5e0035b3f44af`. The original failed ledger entry remains failed, with its ceiling retained. The current workspace gross snapshot is $11.84010451 with $0 billed; remaining original-budget margin is $18.15989549. The before-recovery source archive has 375 independently checked members, SHA-256 `0ac135dfb55cfe478cea7f299a80d5e43f782a29ac5b32939cbd27579eac4503`. An earlier archive attempt overlapped a journal write and failed independent member verification; its incomplete archive and failure log were preserved, then the snapshot was repeated after the writer completed. No source manifest was invented for that failed archive. Recovery completion and restored optimizer/RNG are not claimed until the worker returns its verified receipt.

### 2026-10-13T16:49:29.081617+00:00: V6 execution milestone

Completed **training-resumed**; exact command output is retained in `runs/study-v6/finish-training-resumed.log`. No Git commit or push is part of this controller.

### 2026-10-13T16:49:29.087207+00:00: V6 execution milestone

Both arms completed the frozen training exposure; checkpoint selection and untouched final tests follow.

### 2026-10-13T16:58:28.134260+00:00: V6 execution milestone

Completed **selection**; exact command output is retained in `runs/study-v6/finish-selection.log`. No Git commit or push is part of this controller.

### 2026-10-13T17:06:28.734489+00:00: V6 execution milestone

Completed **test**; exact command output is retained in `runs/study-v6/finish-test.log`. No Git commit or push is part of this controller.

### 2026-10-13T17:10:05.168491+00:00: V6 execution milestone

Completed **records**; exact command output is retained in `runs/study-v6/finish-records.log`. No Git commit or push is part of this controller.

### 2026-10-13T17:10:13.325438+00:00: V6 execution milestone

Completed **analysis**; exact command output is retained in `runs/study-v6/finish-analysis.log`. No Git commit or push is part of this controller.

### 2026-10-13T17:10:20.347692+00:00: V6 execution milestone

Completed **report**; exact command output is retained in `runs/study-v6/finish-report.log`. No Git commit or push is part of this controller.

### 2026-10-13T17:22:04.133098+00:00: V6 scientific execution complete; artifact closure pending

Both arms completed 36,288 question presentations, 1,134 optimizer updates and three epochs. Input tokens are 7,615,419 for control and 7,955,055 for treatment (+4.46%). The active amended protocol is `099611c7f4eb70c9bc12f2fd44d0279c5bfc81e0cf0913b6e5d5e0035b3f44af`; the unchanged initial protocol and original failure source remain separately preserved. Recovery verified every saved weight/optimizer/progress hash, restored optimizer/RNG state, and continued from update 250 through 628 without changing the failed call to success. All 105 CPU tests passed, including exact interrupted-versus-uninterrupted final weights in the regression.

Selection identity `8b06abd1ab7318e67385435fb8311013f2524574f6dc44b946528f2e55f291e1` was sealed before final testing. Calibrated selection NLL chose control step 0 (0.8904870358) and treatment step 1,134 (0.1815701121); final temperatures are 6.5513434977 and 1.1905159378. Accuracy selection would choose control step 900 but is descriptive only. All trained control checkpoints have worse calibrated selection NLL than the starting model. This is a selected-pipeline comparison, not two terminal checkpoints. Independent raw-logit counting confirms 21/360 versus 214/360 complete worlds and 481/1,440 versus 1,231/1,440 correct questions; selected control and unchanged parent have identical raw predictions on all shared baseline tests.

Primary gain is 53.61 percentage points, 95% blocked interval [48.61, 58.34] across 360 worlds and 120 three-world label-rotation blocks. Both predeclared checks pass. Old original UNKNOWN improves from 221/300 to 229/300: +2.67 points, paired 95% interval [0.33, 5.33]. Old changed-premise UNKNOWN stays 300/300 in both, with an empirical [0,0] interval; that is not population perfection. Multi deeper/wording/signed-chain question accuracy is 84.26%/77.55%/85.42%. Transformations remain fragile: only 9/360 treatment families are completely correct across all eight variants. Different-entity/different-label pairs have both answers correct in 766/1,227, versus 10/1,227 for control. This demonstrates better query-dependent answers without establishing an internal binding mechanism.

The preselected homogeneous base world is correct for treatment, but replacing `Aman is verified.` with a filler removes support and treatment still says TRUE at 79.26% calibrated confidence. Explicitly negating that same fact yields the correct FALSE. Old wording and single-branch accuracy remain 46.67%/51.00%, with 122/300 and 76/200 errors at at least 90% calibrated confidence. Separate **post hoc descriptive** slicing of existing logits reconciles with every frozen aggregate: TRUE-to-UNKNOWN edited answers are correct in 80/120; UNKNOWN-to-FALSE both-answer correctness is 62/120. No new inference, checkpoint selection or endpoint change occurred.

[V6 results](reports/V6_RESULTS.md), [execution guide](docs/V6_EXECUTION.md), raw data/logits/trajectories under `runs/study-v6`, and before-inference/before-recovery source archives retain the evidence. Final lint, formatting, lock and whitespace checks passed. V5's frozen models/data and 137 historical scientific/report files were rechecked unchanged. Eight trained weight files are transferring individually to private checkpoint storage; final audits, selected-bundle fresh-Hub verification, cleanup, billing and the closing archive are still pending. Git changes remain local. [V7](docs/V7_PROPOSAL.md) is only a proposal to replicate the gain with fresh seeds before a controlled robustness objective or architecture experiment; no next training or allocation is submitted.

### 2026-10-13T18:09:57.148732+00:00: V6 artifact recovery and numerical verification

Private checkpoint storage stopped at 2026-10-13T17:45:35Z when a temporary DNS failure prevented a Modal storage block download. Six trained checkpoints had already been uploaded and verified by pinned private LFS metadata. Preserved `finish-candidates.log`, the original controller failure, and the 80 MiB partial treatment-step900 download under `failed-storage-downloads`, with a receipt. DNS resolution was rechecked successfully before resuming. `v6_finalize.py` resumes only candidate retention and the remaining export/audit/private-verification/cleanup/billing stages. No GPU call, data regeneration, selection or final inference is repeated.

Independent metric recomputation preserves the original worker outputs. All nonfloat fields, decisions and counts match; ordinary float differences are tiny. Three transformation-suite AURC values differ across Mac/Linux by at most 6.0163355875e-7 because nearly tied confidence scores can sort differently. Preserved every difference, the original strict audit helper, and a separate correction receipt. `v6_metric_audit.py` permits at most 1e-12 absolute differences for ordinary floats. Exceptional AURC values must lie inside explicit extrema from ordering errors within confidence groups spanning at most 32 float64 epsilons (7.1054273576e-15), with order across groups fixed. All three original values satisfy these bounds. Frozen scientific source, analysis, metrics and endpoints are unchanged.

The first new synthetic audit fixture had no UNKNOWN rows, producing an undefined UNKNOWN metric; its failed log is retained. Corrected the fixture's class mapping so the metric is defined. Five rejection tests then passed, covering changed counts, nonroundoff scores, nontied AURC corruption and values outside tied-order bounds. The full closing CPU suite passed **110 tests in 50.05 seconds**. Current closure remains pending: checkpoint transfer, selected private-Hub verification and cleanup must still complete before deleting the working Volume. Git remains uncommitted and unpushed; no next experiment is submitted.

### 2026-10-13T18:12:31.307763+00:00: V6 execution milestone

Completed **candidates-resumed**; exact command output is retained in `runs/study-v6/finish-candidates-resumed.log`. No Git commit or push is part of this controller.

### 2026-10-13T18:13:46.495653+00:00: V6 execution milestone

Completed **releases**; exact command output is retained in `runs/study-v6/finish-releases.log`. No Git commit or push is part of this controller.

### 2026-10-13T18:13:59.386521+00:00: V6 execution milestone

Completed **audit**; exact command output is retained in `runs/study-v6/finish-audit.log`. No Git commit or push is part of this controller.

### 2026-10-13T18:16:37.463254+00:00: V6 execution milestone

Completed **private-releases**; exact command output is retained in `runs/study-v6/finish-private-releases.log`. No Git commit or push is part of this controller.

### 2026-10-13T18:16:52.344805+00:00: V6 execution milestone

Completed **cleanup**; exact command output is retained in `runs/study-v6/finish-cleanup.log`. No Git commit or push is part of this controller.

### 2026-10-13T18:16:58.502532+00:00: V6 execution milestone

Completed **billing**; exact command output is retained in `runs/study-v6/finish-billing.log`. No Git commit or push is part of this controller.

### 2026-10-13T18:18:58.641104+00:00: V6 verified retention, private releases and cleanup completed

Completed only the already authorized V6 scope. Final audit verifies 69,208 prediction rows in 55 files, exact paired exposure, retained scheduled candidates, original recovered trajectory prefix and partial packaging assets. All 110 closing CPU tests passed in 50.05 seconds; closing lint (176 Python files), format, lock and whitespace checks passed. Original and amended scientific protocols/data/parent/settings remain unchanged. The V4 audit covers 90 scientific/build files; the separate V5 check covers its frozen models/data and 137 historical scientific/report files. These inventories overlap and must not be added together.

Selected control is privately published at `axondendriteplus/decision-lab-v6-smollm2-360m-control`, commit `cde5272c428ecad79b07b374361c1e46c3dd2c30`. Selected treatment is at `axondendriteplus/decision-lab-v6-smollm2-360m-multi-entity`, commit `07d879161e56969669e699f4e5c2b4cc0a70f077`. Both pass all eight fixed FP32 CPU reference inputs after fresh pinned Hub downloads, in addition to the GPU export checks. All eight additional scheduled weights are privately retained at `axondendriteplus/decision-lab-v6-checkpoints`, final commit `eb6869a4265e3155f37b15fe8a78f8d04a13b440`, verified by pinned LFS size/SHA metadata; intermediate inference verification is not claimed. Ten candidate entries include both original-parent aliases. Selected local bundles and fresh verification downloads remain available.

At `2026-10-13T18:16:58.148492+00:00`, resource-ID-attributed V6 metering is **$1.75521017** (L4 $1.12766473, CPU $0.26669272, memory $0.36085272), with **$0 billed**. Metering can lag and is not an invoice. Workspace gross observation is $13.41010451, giving a conservative $16.58989549 original-budget margin rather than a live-wallet claim. The original $3.8359077778 reservation includes the failed control call and its single reserved recovery; no further GPU retry occurred. Cleanup verified every owned app directly as stopped/zero tasks, deleted only Volume `decision-model-lab-v6-runs` (`vo-bNjgm8akdamuhjln5YX6vS`), then rechecked absence and stopped states. Unrelated resources were not changed.

Retained failures include the original packaging call/source, the invalid overlapping snapshot attempt, the pre-submission CLI error, the DNS-interrupted 80 MiB download and the initial numerical-audit fixture. Numerical comparison permits only documented roundoff/near-tie ordering bounds; original results remain untouched. [V6 results](reports/V6_RESULTS.md), [execution guide](docs/V6_EXECUTION.md), compact evidence and the final source snapshot capture the work. The independently verified final archive SHA and inventory live in `artifacts/v6-source-snapshot/manifest-final-with-reports.json` and the external `runs/study-v6/closure-receipt.json`, avoiding a circular source/archive hash in this journal. Full raw data/logits/trajectories remain local, outside the source archive and the earlier V1–V4 remote backup. V5/V6 Git changes remain uncommitted and unpushed. No merge, next training or new allocation occurred.

**Next proposed:** replicate the large gain on fresh training/data seeds and preserve separate seed results, then test one controlled robustness objective if the weakness repeats. [V7](docs/V7_PROPOSAL.md) is a proposal only. The observed gain does not establish general decision ability or an internal binding mechanism.

### 2026-10-14T10:10:26.689730+00:00: V7 authorized, implemented, CPU-tested and frozen

The owner authorized V7 after reviewing the next step. Implemented isolated `v7_lab` scientific code with V6's generator, training, checkpoint selection/calibration and final-test design. Seeds are 731 and 947, with fresh optimizer/data streams within each matched pair and the same pinned V4 mixed initialization. Generator-equivalence tests confirm that namespacing does not change prompts, labels, worlds or proof depths at the same seed. The new data excludes recorded V1–V6 worlds and additionally excludes all r1 worlds from r2. Per repetition: 45,080 serialized questions, 6,996 families and 14,366 unique canonical worlds. Each arm receives 12,096 questions/epoch, 36,288 presentations and 1,134 updates. Canonical exclusion covers 100,751 old worlds for r1 and 115,117 for r2. Both oracles/rendering audits passed; maximum prompts are 353 and 354 tokens, with no truncation. Input tokens per epoch: r1 control 2,535,330/treatment 2,655,786; r2 control 2,537,575/treatment 2,657,154. These are measured differences, not matched-token claims.

Three initial V7 tests passed in 18.27 seconds. Expanded coverage tests failed-call reservation caps, exact uninterrupted/resumed training, shared exposure, per-repetition freshness and namespacing, V6-generator equivalence, selected export reload and safe disk deduplication/partial-download retention. The full CPU suite passed **119 tests in 71.05 seconds**. Lint, formatting (193 Python files), dependency-lock and whitespace checks passed after ordinary formatting corrections. No historical scientific file was edited. Source archival and the separate historical checksum check follow preparation before submission.

Master protocol identity `46afd8ff9cb9dcb48fe3adb4de3203b56a3af92a11a40cb432c924434b0aff73` links r1 `f07e4f34618900fe4a571f47f380737efe7919897a5465704c92a138c3154c71` and r2 `94550cc844cb3a116a64211bac4be473610aa0b835f8b493f70b2478ab9f808f`. Both datasets are frozen before any GPU call. Primary/retention checks are shown per seed; there is no pooled training-seed interval or joint 95% guarantee. Update-zero candidates remain eligible. No terminal-checkpoint test is added after selection, and no pair-consistency objective or architecture is trained under V7.

The billing API reports $13.39010451 gross metering and $0 billed, leaving a conservative $16.60989549 margin against the original $30 budget, not a live-wallet balance. Published Modal prices and API rates were rechecked: L4 $0.80/hour, physical CPU $0.0473/core-hour, memory $0.008/GiB-hour. Joint maximum compute is $7.6718155556 under an $8 allocation, with $1 storage/egress reserve. One recorded recovery allowance per repetition is included; failed calls remain reserved, and exact-state recovery requires inspecting the checkpoint. No automatic GPU retries or always-on endpoint are used. Live billing headroom is rechecked before each stage.

Only the owned V7 Volume can be deleted after raw-data/candidate/selected-model retention, fresh private-Hub FP32 CPU reference checks and direct stopped/zero-task app checks. Additional weights transfer one at a time to private HF stores; selected duplicate files can be atomically replaced by identical local hardlinks only after fresh download verification, preserving bundle bytes while respecting about 8 GiB free disk. Credentials remain outside the worker and chat. Git implementation stays local; no commit/push or new objective/architecture is authorized by this protocol. Next: verify the pre-execution source archive, run both discarded/reference timing pilots, then execute the full frozen scope if both gates pass. No GPU submission or result is yet claimed.

### 2026-10-14T10:18:32.802630+00:00: V7 pre-scoring pilot image failure

The first r1 pilot failed before scoring or discarded updates: the V7 verifier requires `uv.lock` and `docs/V7_PROTOCOL.md`, but the Modal image omitted them. The initial CPU fixture mocked protocol verification and missed the worker packaging contract. Preserved the original app source and failed log/call reservation. Added those two unchanged files to the image. Frozen scientific code, settings, data, model and master identity remain unchanged. Use r1's one reserved recovery allocation to rerun the discarded/reference pilot; no further r1 GPU retry is permitted. The failed pilot stays failed in the ledger. A separately verified before-recovery source archive records the operational correction before submission.

### 2026-10-14T10:23:20.810223+00:00: V7 execution milestone

Completed **r1-pilot-recovery**. Exact output is retained in `runs/study-v7/finish-r1-pilot-recovery.log`. No Git commit or push is part of this controller.

### 2026-10-14T10:27:57.840180+00:00: V7 execution milestone

Completed **r2-pilot**. Exact output is retained in `runs/study-v7/finish-r2-pilot.log`. No Git commit or push is part of this controller.

### 2026-10-14T10:29:16.176679+00:00: V7 both pilots passed; matched training started

The corrected source archive has 500 independently checked members, SHA-256 `1fa11e85cfa698bfbd742821571c8dd8dd1b9b4c4418de0716b3aafc50ac109a`; the original before-execution archive has 499 members, SHA-256 `0b81395fff571f84eaf690b9607da08388dd55f3524b9aaa18ab6862fc2710b3`. Neither historical archive/identity was rewritten. Worker checks verify the unchanged lock/protocol files, frozen source/data and pinned parent. Both pilots passed all eight CUDA reference inputs. Mean update times: r1 control 1.2829524965/treatment 1.3549980611 seconds; r2 control 1.3115105271/treatment 1.3792282801 seconds. Corresponding worst chunk projections are 689.02/699.55 seconds, below the fixed 1,020-second gate. Pilot weights were discarded.

The failed r1 pilot remains failed with its reservation retained. The reserved `pilot-recovery` completed; r1 has no remaining GPU retry allowance. R2 retains its one reserved recovery allowance. The controller now runs r1's six nominal matched training calls, followed by fixed selection/tests, then r2's corresponding scope. All data and endpoints were frozen before either pilot; no test prediction or scientific result has been inspected. Git remains local.

### 2026-10-14T11:35:10.910714+00:00: V7 execution milestone

Completed **r1-train**. Exact output is retained in `runs/study-v7/finish-r1-train.log`. No Git commit or push is part of this controller.

### 2026-10-14T11:43:56.701582+00:00: V7 execution milestone

Completed **r1-selection**. Exact output is retained in `runs/study-v7/finish-r1-selection.log`. No Git commit or push is part of this controller.

### 2026-10-14T11:52:40.477625+00:00: V7 execution milestone

Completed **r1-test**. Exact output is retained in `runs/study-v7/finish-r1-test.log`. No Git commit or push is part of this controller.

### 2026-10-14T11:56:23.698387+00:00: V7 execution milestone

Completed **r1-records**. Exact output is retained in `runs/study-v7/finish-r1-records.log`. No Git commit or push is part of this controller.

### 2026-10-14T11:56:32.087688+00:00: V7 execution milestone

Completed **r1-analysis**. Exact output is retained in `runs/study-v7/finish-r1-analysis.log`. No Git commit or push is part of this controller.

### 2026-10-14T11:57:33.426861+00:00: V7 r1 scientific execution completed; r2 training started

R1 selected control update 900 and treatment update 1,134; both arms completed 1,134 updates/36,288 presentations. The sealed selection identity is `b6c7489a6a87c93f856b2ed53943d7636606ea7e5de0e33303963f9a48ecbfbc`. Direct argmax/group counts from retained primary rows agree with the analysis: control 25/360 complete worlds, treatment 245/360, unchanged parent 25/360. Primary gain is 61.11 points, blocked 95% interval 57.22–65.00; both old UNKNOWN-retention checks pass. Original UNKNOWN counts: parent 227/300, treatment 241/300, paired gain 4.67 points with interval 2.33–7.33. Changed-premise UNKNOWN is 300/300 for both. Treatment has only 43/360 complete transformation families, 41.67% old wording accuracy and 45.50% old single-branch accuracy; respective saved-temperature confident errors are 481/11,520 transformation queries, 143/300 wording queries and 81/200 single-branch queries. These weaknesses are retained. Both exported bundles passed eight CUDA references. Raw-record retention copied 137 JSON/JSONL files. R2 is now training on seed 947 with the unchanged frozen design. R1 alone is not the global replication result; final numerical audit, private candidate retention, fresh Hub CPU checks, cleanup and archival remain pending. Git remains local.

### 2026-10-14T12:41:17.458846+00:00: V7 independent counts and artifact-retention overlap

Added `v7_raw_audit.py`, a standard-library check of serialized input/prediction identity, labels, complete-world/family counts and old UNKNOWN contrasts. R1 passed; its immutable receipt hashes the counted input/logit files. Final closure now requires both receipts and unchanged raw-file hashes. Added a standalone results figure builder; it has not generated a figure yet and adds no model inference. These are operational verification/reporting additions, outside the frozen scientific code. Routine formatting corrections were applied; lint passes. The full 119-test evidence remains the pre-execution suite, not a claim that later wrappers are covered by those tests.

`v7_retain_r1.py` runs completed r1 private retention while r2 trains, with separate stage logs/receipt and zero additional GPU calls. The reporting barrier waits for that writer, fails on a preserved retention failure and bounds waiting to one hour; subsequent controller artifact checks remain idempotent. This avoids overlapping artifact writes. At this observation, r1 control updates 250, 600 and selected 900 are privately retained by pinned LFS SHA/size verification. Both controllers must finish before the final source archive. Scientific identities, data, endpoint results and historical files are unchanged; Git remains local.

### 2026-10-14T12:52:38.932074+00:00: V7 disk headroom restored without removing artifacts

Free disk fell to about 3.1 GiB during local checkpoint staging. Inspected only this project's artifacts and relevant cache sizes; no unrelated file/cache was changed. `v7_deduplicate_v6.py` verified that both V6 fresh private-Hub downloads had previously passed all eight CPU references and that their complete bundle inventories matched the retained local bundles. It atomically hardlinked the duplicate weight files to the byte-identical local weights, then verified every bundle member again and confirmed the historical Hub verification receipt SHA remained unchanged. All paths/model bytes/results/receipts remain present; only duplicate physical storage was reclaimed. Observed free-space change is 2,895,470,592 bytes, about 2.7 GiB, with about 5.8 GiB free afterward. The V7 receipt retains both V6 private commits, manifest/weight hashes and same-inode checks. No new historical download or inference is claimed. R2 treatment is entering epoch three; global closure remains pending.

### 2026-10-14T13:02:32.078801+00:00: V7 execution milestone

Completed **r2-train**. Exact output is retained in `runs/study-v7/finish-r2-train.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:12:22.524687+00:00: V7 execution milestone

Completed **r2-selection**. Exact output is retained in `runs/study-v7/finish-r2-selection.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:21:06.694214+00:00: V7 execution milestone

Completed **r2-test**. Exact output is retained in `runs/study-v7/finish-r2-test.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:25:17.574694+00:00: V7 execution milestone

Completed **r2-records**. Exact output is retained in `runs/study-v7/finish-r2-records.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:25:25.429915+00:00: V7 execution milestone

Completed **r2-analysis**. Exact output is retained in `runs/study-v7/finish-r2-analysis.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:28:08.289676+00:00: V7 both fresh-seed scientific comparisons completed

R2 selected control and treatment update 900 under sealed selection `a6d81855a856a370df162f26a78175a039eba56fc483e235097c6f75dc4a6edd`; tests were unopened at sealing. Both arms completed 1,134 updates and 36,288 presentations. Direct raw counts give control 19/360 complete primary worlds, treatment 245/360, unchanged parent 23/360. Primary gain is 62.78 points with blocked 95% interval 59.17–66.39. Both retention checks pass: original UNKNOWN is parent 228/300 versus treatment 247/300 (gain 6.33 points, interval 3.67–9.67); changed-premise UNKNOWN is parent 300/300 versus treatment 299/300 (gain -0.33 points, interval -1.00–0.00), preserving that error rather than claiming perfect retention. Only 36/360 treatment families are correct across every transformation; old wording accuracy is 51.33% and single-branch accuracy is 53.50%. Both r2 exports passed eight CUDA references. All 137 raw JSON/JSONL records were retained. The separate standard-library raw audit passed.

Both declared primary/retention engineering checks now pass in both fresh seeds. The descriptive mean primary contrast is 61.94 points; no pooled population/training-seed interval or joint 95% guarantee is asserted. Generated and visually inspected standalone `reports/V7_RESULTS.png` and `.pdf` from the verified counts and frozen per-seed intervals. The main controller waits at reporting for independent r1 private artifact retention before its idempotent rechecks, preventing concurrent artifact writes. Private candidate retention, final metric audits, fresh selected-Hub CPU verification, cleanup, costs and final archival are still pending. No further V7 GPU scoring is planned and no next objective/architecture has been trained. Git remains local.

### 2026-10-14T13:28:52.113474+00:00: V7 r1 retention milestone

Completed **r1 candidates** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v7/retention-r1-candidates.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T13:30:12.043449+00:00: V7 r1 retention milestone

Completed **r1 releases** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v7/retention-r1-releases.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T13:30:27.780406+00:00: V7 r1 retention milestone

Completed **r1 audit** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v7/retention-r1-audit.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T13:33:08.215943+00:00: V7 forty test metric sets recomputed; r1 retention audit passed

`v7_metric_precheck.py` locally recomputed all 40 selected-model test metric sets from frozen raw logits and saved temperatures. All comparisons passed the documented exact-count/type, 1e-12 ordinary roundoff and explicit near-tie AURC-order bounds; original worker metrics are unchanged. The immutable global receipt records every compared prediction hash. This is a metric precheck, not a claim that r2 private storage/release verification is complete. R1 retained all ten candidate entries with eight additional scheduled weights in its private checkpoint store, pinned final commit `844f06b3c2f06891b16dfc31fd43f2a9136d2ea5`. Its full final audit verifies 69,208 prediction rows across 55 files, state/trajectory/exposure, candidate storage and selected bundle inventories. R1 fresh selected-Hub CPU checks are running; r2 artifact retention and global cleanup/closure remain pending. No GPU run or Git publication was added.

### 2026-10-14T13:33:46.204436+00:00: V7 r1 retention milestone

Completed **r1 publish** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v7/retention-r1-publish.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T13:33:56.133355+00:00: V7 execution milestone

Completed **report**. Exact output is retained in `runs/study-v7/finish-report.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:34:07.437329+00:00: V7 execution milestone

Completed **r1-candidates**. Exact output is retained in `runs/study-v7/finish-r1-candidates.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:34:14.455279+00:00: V7 execution milestone

Completed **r1-releases**. Exact output is retained in `runs/study-v7/finish-r1-releases.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:34:28.473096+00:00: V7 execution milestone

Completed **r1-audit**. Exact output is retained in `runs/study-v7/finish-r1-audit.log`. No Git commit or push is part of this controller.

### 2026-10-14T13:34:33.113808+00:00: V7 execution milestone

Completed **r1-publish**. Exact output is retained in `runs/study-v7/finish-r1-publish.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:11:40.338839+00:00: V7 closing CPU regression and r1 pinned recheck passed

The full closing CPU suite passed **119 tests in 71.21 seconds**, with separate evidence from the original 119-test preparation run. Later operational scripts were additionally exercised on real retained artifacts: both raw-count audits, all 40 metric recomputations, historical duplicate-weight preservation and the explicit r1 private checkpoint recheck. The tightened store recheck preserved private commit `844f06b3c2f06891b16dfc31fd43f2a9136d2ea5` and candidate-inventory SHA `5bd9bbafa9c88aee6f398c19ab6d072f3b81bd61339b2e925ea95362d954f2e7`, without an additional weight upload. Its receipt records the tested storage-wrapper source SHA.

Added exact fact differences and saved-temperature confidences to both preselected examples, without new inference. In r1, active is replaced by irrelevant enabled; treatment keeps TRUE at 53.56% although the oracle becomes UNKNOWN. In r2, quiet is replaced by irrelevant blue; treatment keeps TRUE at 71.48%. Both explicit negative replacements correctly yield FALSE above 99% confidence. This supports the retained weakness rather than replacing preselected cases. Wrote `docs/V8_PROPOSAL.md` for a future matched paired-data loss comparison, citing the original AugMix paper/implementation as motivation and marking the label-changing adaptation as an untested hypothesis. No V8 code, dataset, GPU call or allocation is submitted. R2 now has four trained control checkpoints privately retained; treatment transfers and global closure remain pending. Git remains local.

### 2026-10-14T14:45:37.455585+00:00: V7 execution milestone

Completed **r2-candidates**. Exact output is retained in `runs/study-v7/finish-r2-candidates.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:47:03.734309+00:00: V7 execution milestone

Completed **r2-releases**. Exact output is retained in `runs/study-v7/finish-r2-releases.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:47:21.985605+00:00: V7 execution milestone

Completed **r2-audit**. Exact output is retained in `runs/study-v7/finish-r2-audit.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:50:02.160613+00:00: V7 execution milestone

Completed **r2-publish**. Exact output is retained in `runs/study-v7/finish-r2-publish.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:50:10.399733+00:00: V7 execution milestone

Completed **historical**. Exact output is retained in `runs/study-v7/finish-historical.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:50:31.870241+00:00: V7 execution milestone

Completed **cleanup**. Exact output is retained in `runs/study-v7/finish-cleanup.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:50:37.702104+00:00: V7 execution milestone

Completed **costs**. Exact output is retained in `runs/study-v7/finish-costs.log`. No Git commit or push is part of this controller.

### 2026-10-14T14:56:25.886513+00:00: V7 all retained models and owned cleanup verified

All four selected private bundles pass fresh pinned-Hub CPU checks on eight inputs each. R1 control/treatment revisions are `82620838100d55290e309e5c561d7827acd70507` / `6aa3e01e41460dba7049c044906dc81fe4c7ba53`; r2 revisions are `df8e085ffd931fac19c4362d205e7bbf78b65501` / `63d1df3c1648217398243b7b943a9d1cd7362ef0`. Each local and fresh verified bundle remains complete after byte-identical hardlink deduplication. Both ten-entry private candidate stores retain eight additional trained weights each; final pins are r1 `844f06b3c2f06891b16dfc31fd43f2a9136d2ea5` and r2 `f02056903ef00dc6402e29865103140448a5181d`. All 138,416 prediction rows in 110 files are audited. Complete optimizer/RNG states are temporary, not permanently retained.

As disk headroom narrowed again, additionally hardlinked only V4 treatment's previously verified fresh duplicate to its byte-identical local weight. Complete bundles and the historical verification receipt stayed unchanged; observed reclamation was 1,446,150,144 bytes. The receipt records the unchanged pinned V4 revision and reusable helper SHA. No unrelated files/caches were changed.

All nine owned apps were directly checked stopped with zero tasks, only the recorded V7 Volume `vo-n7C4eHw1NUOunUaS9dAIiJ` was deleted, and absence/stopped states were rechecked. Historical checks preserve 262 prior source/report/build files, all frozen models/data and recorded exclusion registries. Observed owned metering at 2026-10-14T14:50:37.324652+00:00 is $3.53624060, with $0 billed. Workspace gross is $16.98010451, giving a conservative original-budget margin of $13.01989549; this is not a live-wallet balance or invoice. Nineteen GPU ledger records include the sole preserved failed pilot and successful reserved rerun; no further GPU retry occurred.

The primary gain and all engineering checks replicate across both seeds, with persistent transformation, wording and missing-support failures. The result concerns selected pipelines and fixed-model world-sampling intervals, not a general decision engine or identified reasoning mechanism. Both controllers have exited; closing documentation/style/source checks and the final independent archive/closure follow. No next experiment or allocation is submitted, and Git remains local.

### 2026-10-14T15:01:12.790555+00:00: V7 closing source checks passed; final independent archival

Closing lint, formatting (202 Python files), dependency-lock and whitespace checks all pass. Git HEAD remains `f3950097e6133a9df026cce1629ca54103a25b2b`; no V5–V7 source commit or push occurred. Both execution/retention writers exited before this final documentation and archive step. Scientific V7 identities remain master `46afd8ff9cb9dcb48fe3adb4de3203b56a3af92a11a40cb432c924434b0aff73`, r1 `f07e4f34618900fe4a571f47f380737efe7919897a5465704c92a138c3154c71`, r2 `94550cc844cb3a116a64211bac4be473610aa0b835f8b493f70b2478ab9f808f`; no scientific source, data or endpoint was amended.

The final closure command checks both 119-test logs, the 138,416-row audits, both independent raw-count receipts and their input hashes, all 40 metric input hashes, reviewed figure hashes, private selected verification, historical preservation, complete call reservations and owned cleanup. It exports compact metadata/source/log evidence, independently checks each final archive member and scans credential patterns. The final archive SHA/inventory is stored in `artifacts/v7-source-snapshot/manifest-final-with-reports.json`; the external `runs/study-v7/closure-receipt.json` records completion, avoiding a circular journal/archive hash. Complete raw inputs/logits/trajectories and local models remain outside the source archive and the earlier remote V1–V4 backup. The next paired-loss hypothesis is proposal only. No documentation writer runs during or after final source archival.

### 2026-10-14: V8 implementation authorized

The owner authorized the final controlled paired-loss experiment. Implementation is underway locally; no V8 GPU call is submitted yet. Both arms will receive identical paired data and start from the pinned V7 r1 treatment. The new comparison changes only the additional label-aligned Jensen-Shannon term. Planned checks distinguish support removal, explicit reversal, irrelevant edits and unaffected people, with two fresh repetitions. We will freeze source/data/selection rules after CPU feasibility checks, inspect live Modal billing, retain all evidence and stop this fine-tuning line if neither arm improves faithful edits. Git publication is not authorized. Local disk space is tight; only byte-identical owned artifact copies may be hardlinked, preserving every path and checksum.

### 2026-10-14: V8 CPU feasibility and budget preflight

All 125 CPU tests passed in 112.02 seconds, including both-arm exact paired-training resume, candidate selection/export/reload and intervention/blocked-bootstrap checks. Targeted initial checks passed five tests; the final full suite includes six V8 tests. The initial direct pytest collection error is retained in `artifacts/v8-cpu-initial.log`; corrected module invocation is recorded separately. Lint/format and whitespace checks passed. Live billing at 2026-10-14T15:58:31.908347Z shows $0 billed and $13.04989549 conservative margin against the original $30 budget. The allocation is $8 compute plus $1 reserve; maximum frozen call compute is $7.67181556. Private parent authentication and pinned revision checks pass. Owned historical weight deduplication preserved 72 paths and every hash while replacing 18 byte-identical physical copies with hardlinks; observed free-space gain 14,863,339,520 bytes. Fresh-data audits are proceeding before joint freeze and GPU submission.

### 2026-10-14T16:05:58.355690+00:00: V8 execution milestone

Completed **r1-pilot**. Exact output is retained in `runs/study-v8/finish-r1-pilot.log`. No Git commit or push is part of this controller.

### 2026-10-14T16:08:43.118964+00:00: V8 execution milestone

Completed **r2-pilot**. Exact output is retained in `runs/study-v8/finish-r2-pilot.log`. No Git commit or push is part of this controller.

### 2026-10-14: V8 pretraining position correction

A post-freeze data inspection caught edit type correlated with target slot for three-person worlds. The scientific-source guard then prevented the next training stage before GPU submission. Both original pilots completed and discarded weights; no full training, selection or final tests were run. Preserve the original study/source/data identities and initial two pilot reservations under `runs/study-v8-original` and the unchanged first source archive. The corrected generator uses independent target-slot rotation, adds explicit position coverage/balance validation and excludes all original prepared worlds. New families use a balanced namespace and receive a new freeze. Corrected pilots consume the two existing retry allowances, leaving no further recovery calls; the original $8 compute plus $1 reserve remains the cap. The correction is an exposed preparation error, not a failed GPU result or a tuned scientific outcome. Exact original resource cleanup and input/source evidence are retained.

### 2026-10-14: corrected V8 seed replay stopped at CPU generation

The first corrected preparation reused seeds 1103/1301 while excluding every world from their original preparation. Replaying that RNG stream exhausted the 200-attempt cap on the first old-format family. No GPU call or new freeze was submitted. Preserve the partial parent/preflight, attempted config, failed generation log and the 126-pass CPU log under `runs/study-v8-position-seed-failure`. Fix the corrected seeds to 1601/1801 before regeneration; no model results were examined or selected. The original study and pilot ledger remain unchanged. `artifacts/v8-preparation/seed-replay-correction.json` records the distinction.

### 2026-10-14T16:28:42.167369+00:00: V8 execution milestone

Completed **r1-pilot**. Exact output is retained in `runs/study-v8/finish-r1-pilot.log`. No Git commit or push is part of this controller.

### 2026-10-14T16:31:28.454821+00:00: V8 execution milestone

Completed **r2-pilot**. Exact output is retained in `runs/study-v8/finish-r2-pilot.log`. No Git commit or push is part of this controller.

### 2026-10-14: corrected V8 freeze and full training submitted

Corrected study identity `c590e9b9727c5dfa81fce210a79b1be40b2783b01ce1dcb55393630a7858075d`; r1 `c3dd59fa24ec9c61658603eef1ed560b5b1094712936ed31fb4bc2354ea88b62`, r2 `9c843f1daa2333aae4bcafb9cc5b719adf484e3294b77199602b85308a3edc71`. Both repeat datasets pass oracle/partition/rendering and target-position coverage checks; zero cross-repetition canonical overlap. Each repetition serializes 50,264 question rows; max prompt length 355/512. Identical per-arm input tokens per epoch are 2,650,958 (r1) and 2,649,010 (r2); three epochs per arm. Exclusion registries cover 145,759 prior worlds for r1 and 153,897 for r2, including original preparation and r1. Current 126-test CPU suite passes in 197.61 seconds. The corrected pre-execution archive has 661 verified members and SHA `c41ba92d85bab233240b45498903019a28a37b7f5c17615313bea2641b73a640`; the original 659-member archive remains unchanged. Both corrected pilots verify the eight parent references and fit the timing gate, with worst seconds per update 1.421/1.347. Full r1 training was submitted at 2026-10-14T16:31:28.455655Z. The original two pilots remain in the joint ledger; revalidation consumes both retry allowances. Main run, model retention, independent raw audits, private CPU checks, cleanup and closure are pending.

### 2026-10-14: preselected V8 examples and r1 control completion

The final report will render the two premise-removal families already named in the corrected data freeze, showing unchanged-parent, paired-CE and consistency predictions/confidence from saved test logits. The renderer passed a local check with synthetic logits and frozen inputs; no model inference or scientific parameter change was performed. The check is retained in `runs/study-v8/example-renderer-check.json`. This is a presentation helper added after the corrected initial source archive; final source archival retains its executed version. The first control arm completed all 1,134 updates and 36,288 presentations; the matched consistency arm is running. Held-out model scoring remains pending.

### 2026-10-14T17:35:28.607165+00:00: V8 execution milestone

Completed **r1-train**. Exact output is retained in `runs/study-v8/finish-r1-train.log`. No Git commit or push is part of this controller.

### 2026-10-14T17:48:58.721184+00:00: V8 execution milestone

Completed **r1-selection**. Exact output is retained in `runs/study-v8/finish-r1-selection.log`. No Git commit or push is part of this controller.

### 2026-10-14T17:59:37.806685+00:00: V8 execution milestone

Completed **r1-test**. Exact output is retained in `runs/study-v8/finish-r1-test.log`. No Git commit or push is part of this controller.

### 2026-10-14T18:03:21.368662+00:00: V8 execution milestone

Completed **r1-records**. Exact output is retained in `runs/study-v8/finish-r1-records.log`. No Git commit or push is part of this controller.

### 2026-10-14T18:03:31.493624+00:00: V8 execution milestone

Completed **r1-analysis**. Exact output is retained in `runs/study-v8/finish-r1-analysis.log`. No Git commit or push is part of this controller.

### 2026-10-14T18:03:32.788611+00:00: V8 execution milestone

Completed **r1-raw-counts**. Exact output is retained in `runs/study-v8/finish-r1-raw-counts.log`. No Git commit or push is part of this controller.

### 2026-10-14: V8 first-seed scientific scoring and independent counts

Seed 1601 selects update 1,134 in both arms. Fresh complete paired-world counts: unchanged parent 188/378, paired CE 205/378, added consistency 208/378. Treatment minus control is +0.79365 points, blocked 95% interval [-1.85185, +3.70370]; the declared positive-lower-bound check fails. Both treatment UNKNOWN-retention checks and base-world retention pass. Versus parent, paired CE gains +4.49735 points [1.32275, 7.67196], consistency +5.29101 [1.85185, 8.46561], conditional on this trained run and fresh world sampling. The independent raw-count audit passes. These are first-seed results; private candidate retention/release checks, full final audit and second-seed scoring are pending. Seed 1801 training began at 2026-10-14T18:03:32.800094Z without changing data, loss, selector or scope. No coefficient tuning or replacement seed is submitted.

### 2026-10-14T18:42:47.129379+00:00: V8 r1 retention milestone

Completed **r1 candidates** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v8/retention-r1-candidates.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T18:44:01.968130+00:00: V8 r1 retention milestone

Completed **r1 releases** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v8/retention-r1-releases.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T18:44:18.534563+00:00: V8 r1 retention milestone

Completed **r1 audit** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v8/retention-r1-audit.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14T18:46:32.570904+00:00: V8 r1 retention milestone

Completed **r1 publish** while the independent r2 GPU execution proceeds. Exact output is retained in `runs/study-v8/retention-r1-publish.log`. No GPU scoring/training is repeated. The main controller will recheck the idempotent retention stages before global cleanup; Git remains local.

### 2026-10-14 UTC / 2026-10-15 local: expired-JWT interruption and explicit operational amendment

The long-lived local client disconnected; the ephemeral r2 training app stopped. Its later auth refresh failed with Connection lost and returned a cached expired JWT. `AttemptStart` rejected the planned `r2/treatment-3` request in 1.19 seconds, before its worker began. Preserve that failed reservation in full. Fresh authenticated read-only inspection confirms treatment state/progress at update 756, all 756 prior trajectory entries, its checkpoint completion marker under the unchanged scientific identity, absent 900/1,134 treatment candidates, and all six then-owned corrected apps stopped with zero tasks. Stored credentials still work; no reauthentication or extra credential is needed. The first inspection used a wrong marker filename and failed read-only; the corrected receipt is `runs/study-v8/jwt-failure-inspection.json`.

Operational amendment `2c357dabeaf6fab1efb8d2715132087fa1567ee3480d5cc0bfafc51ef7eacb3f` links the unchanged scientific freeze and permits one exact saved-state continuation for the remaining 378 updates. This explicitly deviates from the original no-further-recovery rule after both position revalidations consumed its retry allowances. Total reservation remains $9: reallocate to $8.50 compute/$0.50 transfer/storage, with maximum requested compute $8.09795067 including the failed reservation and the additional call. Live billing shows $0 billed and $9.94989549 original-budget margin. No parent/data/loss/seed/optimizer/exposure/selection/endpoint changes, completed-step replay, coefficient tuning or replacement seed are permitted. The recovery uses a fresh detached bounded app, while a task-scoped idle-sleep inhibitor protects the remaining local controller; owned apps are explicitly stopped after collection. Preserve the original freeze/doc/failed log and a separate pre-recovery source archive.

### 2026-10-14T20:10:29.003640+00:00: V8 execution milestone

Completed **r2-auth-recovery**. Exact output is retained in `runs/study-v8/finish-r2-auth-recovery.log`. No Git commit or push is part of this controller.

### 2026-10-14 UTC / 2026-10-15 local: retained selection outputs and planned-call completion

The exact training continuation reached 1,134 updates/36,288 presentations, preserving the prior 756 trajectory entries and verified optimizer/scheduler/RNG state. The subsequent r2 selection call received a cancellation signal and was killed after completing all 30 selection-fit/selection/development files; cause is unconfirmed. Preserve its failed reservation and original log. All eight then-owned corrected apps are stopped with zero tasks. No current r2 selection freeze or separate-calibration output exists; copied parent selection metadata in candidate folders is historical.

Read-only retention validates all 30 files/30,240 rows against frozen IDs, labels and finite logits. A reconstruction test on the completed r1 data produces identical choices and prediction SHA inventories, with maximum calibrated-NLL difference 5.55e-17 and no additional inference. The r2 continuation reuses all completed selection outputs, applies the same original fitting/choice functions, and scores only the separate calibration partition before unchanged final evaluation. This is folded into the originally reserved 900-second test call; no extra GPU call, allocation, training, seed or scientific method change is introduced. Persist a spawned function-call ID in a detached app, then collect read-only without resubmitting. The failed selection remains in the ledger; final reports expose both interruptions.

### 2026-10-14T21:00:47.329690+00:00: V8 execution milestone

Completed **r2-final-evaluation-submit**. Exact output is retained in `runs/study-v8/finish-r2-final-evaluation-submit.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:16:55.225503+00:00: V8 execution milestone

Completed **r2-records**. Exact output is retained in `runs/study-v8/finish-r2-records.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:04.508863+00:00: V8 execution milestone

Completed **r2-analysis**. Exact output is retained in `runs/study-v8/finish-r2-analysis.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:05.751071+00:00: V8 execution milestone

Completed **r2-raw-counts**. Exact output is retained in `runs/study-v8/finish-r2-raw-counts.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:10.781803+00:00: V8 execution milestone

Completed **report**. Exact output is retained in `runs/study-v8/finish-report.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:20.966974+00:00: V8 execution milestone

Completed **r1-candidates**. Exact output is retained in `runs/study-v8/finish-r1-candidates.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:27.850740+00:00: V8 execution milestone

Completed **r1-releases**. Exact output is retained in `runs/study-v8/finish-r1-releases.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:42.993923+00:00: V8 execution milestone

Completed **r1-audit**. Exact output is retained in `runs/study-v8/finish-r1-audit.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:17:47.085456+00:00: V8 execution milestone

Completed **r1-publish**. Exact output is retained in `runs/study-v8/finish-r1-publish.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:41:06.333833+00:00: V8 execution milestone

Completed **r2-candidates**. Exact output is retained in `runs/study-v8/finish-r2-candidates.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:42:22.178035+00:00: V8 execution milestone

Completed **r2-releases**. Exact output is retained in `runs/study-v8/finish-r2-releases.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:42:37.190581+00:00: V8 execution milestone

Completed **r2-audit**. Exact output is retained in `runs/study-v8/finish-r2-audit.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:45:35.374138+00:00: V8 execution milestone

Completed **r2-publish**. Exact output is retained in `runs/study-v8/finish-r2-publish.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:45:44.796259+00:00: V8 execution milestone

Completed **historical**. Exact output is retained in `runs/study-v8/finish-historical.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:46:04.197421+00:00: V8 execution milestone

Completed **cleanup**. Exact output is retained in `runs/study-v8/finish-cleanup.log`. No Git commit or push is part of this controller.

### 2026-10-14T21:46:10.012932+00:00: V8 execution milestone

Completed **costs**. Exact output is retained in `runs/study-v8/finish-costs.log`. No Git commit or push is part of this controller.

### 2026-10-14T22:04:04.566234+00:00: V8 verified completion

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

### 2026-10-15T09:01:24.576197+00:00: local research article

Request: prepare a precise, substantial research article for review on localhost. Built `research-article/dist` as dependency-free HTML/CSS/JavaScript with an editorial layout, methods, uncertainty, the V1–V8 study sequence, the replicated V7 curriculum gain, mixed V8 consistency-loss evidence, limitations, and original-paper references. Interactive charts read saved audited counts; both premise-removal examples were fixed before evaluation and display retained logits at saved temperatures. No model inference or GPU job was run.

Verification: desktop browser checks at 1440 px and mobile at 390 px; no page overflow, no runtime errors, correct second-run figure counts, correct saved predictions for both removals and original worlds, all local evidence links return 200, and print CSS applies. Browser screenshots and a print PDF are under `artifacts`. JavaScript syntax and Python builder lint/format checks pass. Thirty curated evidence records are copied with a SHA-256 manifest by `research-article/build.py`; final presentation verification is separate from historical source archives. The local server binds loopback and serves only the article directory. No external fonts, analytics, network model calls, Git commit/push, model visibility change, or public hosting. No new Modal cost.

Next: owner review of the local article; model/code public packaging remains separate. The local preview is intentionally left available for reading.

### 2026-10-15T09:13:47.661506+00:00: article restyled to match axondendrite.com

Request: the first local design did not match the owner’s style; inspect axondendrite.com and revise. Read the live homepage and two posts, inspected computed typography/layout and the site stylesheet, and captured reference screenshots. Rebuilt the article as a narrow dark post with a 36rem outer column, 528px desktop text width, #111 background, system sans-serif body at 16.5px, 23.2px title, muted text, understated figures, lowercase headings, and a direct first-person introduction. The initial design is retained under `artifacts/article-initial-design`; no historical scientific result or archive was rewritten.

Verification: JavaScript syntax; browser desktop 1440px and mobile 390px with no horizontal overflow; four interactive result/example checks; 18 local evidence links returning 200; no observed page errors after the fix; print styles and a new PDF under `artifacts/research-article-dark.pdf`. A transient variable-name error introduced during restyling was caught in the browser and corrected before handoff. Exact audit counts and saved prediction confidences are unchanged. Updated 30-file curated evidence snapshot; separate restyling receipt records presentation hashes. Local server reused on port 8765 and left available for review. No Git commit/push, public hosting, model visibility changes, model inference, or GPU work.

### 2026-10-15T09:42:37.507678+00:00: Premise release authorized and prepared

Request: remove budget framing; make the article intuitive; use a genuine recorded case; rename the project; make a representative GitHub repository, selected models and dataset public; add model cards and working links. Selected the name Premise and renamed the private repository to `adc77/premise`. Prepared the complete V5–V8 implementation, reports, compact evidence, public README, reproduction guide, eight V7/V8 model cards, and the exact four-config synthetic dataset (190,688 serialized rows). Historical package names and archived scientific source remain intact.

The article now explains what a world and each scoring unit mean, moves technical details behind disclosure controls, removes budget/accounting framing, and uses the actual held-out Farah support-removal case `v8-r2-balanced-test_multi-multi-3-0012-0`. It is explicitly marked as a post hoc illustration, not a preselected audit example or real operational record. The page shows complete input facts, original rules, exact prompts, saved predictions and the proof that breaks. No illustrative model output was fabricated. All article evidence links point to GitHub and HF destinations.

Verification so far: 126 CPU tests passed in 115.43 seconds; source lint/format checks; exact original model-manifest and weight hashes; credential-pattern scan over working source and all existing Git history; dataset SHA/LFS metadata and four streaming configs; mobile/desktop article controls checked against saved outputs. During transfer, one read connection interrupted and was inspected before resuming. A later private-storage-limit rejection prevented another private model upload. Verified releases are being made public one at a time to free staging quota; no paid upgrade. Receipt-based publication resumes only owned repositories and preserves all failures locally.

No training, GPU job or personal-site deployment is authorized by this release. Final HF pins, clean-checkout checks, anonymous access, Git commits/pushes and visibility checks will be recorded separately after completion.

### 2026-10-15T09:54:50.643386+00:00: Premise public release verified

Published and verified the renamed repository `adc77/premise` on public `main`, eight selected V7/V8 model bundles under the Premise prefix, and `axondendriteplus/premise-worlds` with all four exact frozen configurations. Public packaging preserves original weights, tokenizer and inference code; new cards/manifests have new identities and pin mappings. Historical private artifacts, source archives, and original study results remain unchanged. Public pins are in `release/manifest.json` and `docs/MODELS.md`.

All nine Hub repositories were verified anonymously, including remote model-weight SHA-256 metadata. All eight fresh model downloads passed their eight FP32 references (64 inputs). The release CPU suite passed 126 tests in 115.43 seconds. All four dataset configs loaded; 20 article GitHub targets returned 200 anonymously. A clean checkout rebuilt identical article assets without private run directories. The documented download/checksum/load/predict command worked with implicit authentication disabled; the actual Farah case remained TRUE before and after premise removal under CPU FP32, matching the recorded failure direction. Its probabilities differ slightly from original BF16 and are labeled separately in `release/example-cpu.json`.

A symlink issue in the standard snapshot-cache path was corrected by materializing the documented demo into an ignored local directory; original scientific inference code was unchanged. Private staging hit HF storage quota and a prior read connection interruption; inspected receipt-based resumption and immediate publication of each verified bundle completed all transfers without a paid upgrade. Temporary fresh-download weight copies were removed only after passing checks.

Article updates remove budget framing and the invented introductory world, explain world/question/paired/family units, retain the actual two-step Farah proof and full original inputs, label the new illustration as post hoc, and link public code, dataset and model cards. The final metadata/article changes and release tag are pushed after recording this entry; final remote ref checks live in the external publication receipt to avoid circular commit hashes. No new training, seeds, Modal calls, personal-site deployment, or changed historical outcome.
