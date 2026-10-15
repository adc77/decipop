# V8 execution record

The owner authorized V8 on 2026-10-14. Read the frozen [protocol](V8_PROTOCOL.md), [proposal](V8_PROPOSAL.md) and journal for scope. This document records execution status; it does not modify the scientific freeze.

## Preparation

New V8 source reuses the verified V7 execution pipeline with an explicit paired objective and identical examples in both arms. CPU tests exercise label alignment, identity pairs, extreme logits, finite gradients, wrong-but-consistent predictions, both-arm exact resume, fresh partitions, oracle labels, constant fact counts, blocked intervals and export/reload. The first direct `pytest` invocation failed at collection because top-level study packages were absent from that command's import path; subsequent `python -m pytest` runs use the project root and retain the initial failure log. This was a local invocation error, with no GPU call or scientific result.

`v8_deduplicate.py` verified all 72 owned historical safetensors paths, hardlinked 18 byte-identical copies and checked every preserved checksum afterward. Its receipt is `artifacts/v8-preparation/disk-deduplication.json`; the observed free-space change was 14,863,339,520 bytes. No unrelated paths, model bytes, reports, archives or historical receipts were deleted or rewritten.

## Commands and evidence

Run commands from the repository root with `uv run --extra train --extra cloud --extra analysis`. Preparation is `python v8_prepare.py`; it verifies the pinned private parent, current Modal rates/billing, generates and audits both fresh datasets and freezes identities. Before GPU submission, retain the original `before-execution` source snapshot and corrected `python v8_snapshot.py before-position-revalidation` and pass the complete CPU suite. `python v8_pipeline.py` sequentially runs pilots, bounded training, selection, final tests, raw-record retention, independent raw counts, analysis, scheduled-weight retention, private selected release checks, historical verification, cleanup and billing attribution. Each stage has its own log and status receipt under `runs/study-v8`; failed stages stop the controller and preserve their reservations.

`v8_app.py` submits only the frozen calls. The two original completed pilots remain in the ledger; corrected pilots consume the two reserved retry allocations, leaving no further recovery call. Each GPU worker verifies the scientific source, lockfile, protocol document, parent and data before work. Pilots verify eight parent references and time the longest complete-pair batches; discarded pilot weights never become candidates. Training order shuffles pairs, preserving both versions within every microbatch. Temporary saved optimizer/RNG state supports exact within-run recovery. All scheduled model weights remain private after temporary state is deleted.

The study freeze, per-repetition protocols, selection freezes, input registries, raw logits, trajectories, independent raw-count audits, final metric audits, pinned HF receipts and cleanup record are authoritative evidence. GPU BF16 aggregate scoring is locally recomputed from retained logits; fresh CPU checks cover eight FP32 reference inputs per selected bundle. Recomputing metrics is not rerunning every inference on CPU.

## Completion

Completion, actual failures, observed metering, model pins and cleanup will be recorded after execution. Source/report closure requires a separately verified archive and external closure receipt. No Git commit or push is authorized in this scope.

## Pretraining position correction

After the first freeze and two discarded pilots, inspection caught target slot equal to edit type for three-person worlds. Changing the generator caused the source guard to reject the next training stage before GPU submission. No full training, selection or final-test prediction occurred on the original preparation. The entire original run directory, protocols, datasets, pilot outputs, CPU logs and controller/preflight rejection are retained under `runs/study-v8-original`; the initial source archive remains unchanged. Every original parent/data checksum was verified before its owned Volume deletion, and all original app IDs were checked stopped with zero tasks.

The corrected generator rotates target position independently of edit type, validates every target slot within each edit type and limits per-slot imbalance to one three-world rotation block. Current families have a distinct balanced namespace, and all original prepared worlds are excluded. A new protocol/data/source freeze governs the corrected study. Its two revalidation pilots use the existing retry reservations; no further retry is available, no extra budget allocation is requested and no failed GPU call is invented. The exact correction and original resource IDs are retained in `artifacts/v8-preparation/position-correction.json` and copied into the corrected study.

The first corrected generation reused the original seeds while excluding their prepared worlds. Its deterministic replay exhausted the first-family attempt cap; this CPU-only failure and partial preparation remain under `runs/study-v8-position-seed-failure`. Corrected seeds 1601/1801 were then fixed before the new freeze, without looking at model results. The original pilots and old source/data identities remain unchanged. CPU test logs distinguish the initial 125-pass preparation, the position-corrected 126-pass check and the corrected-seed full run.

## Expired-JWT operational recovery

After update 756 was committed, the local client disconnected and its ephemeral r2 app stopped. A later refresh failed and the SDK attempted the next call with an expired cached JWT; `AttemptStart` rejected it before a final-chunk worker started. The failed reservation and original logs remain unchanged. Fresh profile authentication, owned-Volume identity, saved state/progress and checkpoint marker were checked read-only, with all six then-owned corrected apps stopped/zero tasks. See [the separately sealed amendment](V8_AUTH_RECOVERY_AMENDMENT.md). It permits one exact saved-state continuation, not a replay of completed training, with no scientific settings or original total budget change. New detached apps and a task-scoped idle-sleep inhibitor protect continuation; apps are stopped explicitly after receipts are collected. The first incorrectly named manifest read failed without mutations or GPU calls. The remaining pipeline then resumes selection/tests, retention, verification and closure.

## Selection cancellation and cached continuation

After exact training continuation passed, r2 selection received a cancellation signal. The cause is unconfirmed. All 30 planned candidate selection-fit/selection/development files were retained and validated (30,240 rows); there is no current r2 selection freeze or calibration output. Reuse completed outputs through the unchanged selector, score only the selected calibration partitions, and run final evaluation within the existing test reservation. [The continuation record](V8_SELECTION_CONTINUATION.md) explains cached-file identities, the r1 reconstruction check, unchanged science/budget and persisted asynchronous job collection. Do not resubmit a job because its collector disconnected.

## Completed execution and retention

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
