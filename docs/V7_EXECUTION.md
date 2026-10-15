# V7 execution and artifact boundaries

Read [the frozen protocol](V7_PROTOCOL.md) and [the research journal](../RESEARCH_LOG.md). The journal and receipts distinguish preparation, submission, completion, verification and publication. The master freeze links both repetition identities before any GPU call.

## Historical execution sequence

These commands document this study. Existing paths, stage logs, call names, predictions and archives are guarded against overwriting. A subsequent experiment needs new identities, seeds and owned resource names; do not rerun the opened V7 tests as fresh confirmatory evidence.

```bash
uv run --extra train --extra cloud --extra analysis python -m pytest -q
uv run --extra train --extra cloud --extra analysis python v7_prepare.py
uv run --extra train --extra cloud --extra analysis python v7_snapshot.py before-execution
uv run --extra train --extra cloud --extra analysis python v7_pipeline.py
```

The first r1 pilot failed before model scoring because the image omitted `uv.lock` and `docs/V7_PROTOCOL.md`, which the worker verifier requires. Its source, failed log and reservation remain preserved. The image correction adds those unchanged files; scientific source, data and protocol identities are unchanged. A separate source archive records the correction. The r1 reserved pilot rerun and r2 pilot passed all eight CUDA references and the timing gate. R1 has no further GPU retry allowance.

```bash
uv run --extra train --extra cloud --extra analysis python v7_snapshot.py before-pilot-recovery
uv run --extra train --extra cloud --extra analysis python v7_continue.py
```

`v7_continue.py` executes the recorded pilot recovery, r2 pilot, matched training, selection and tests, raw retention, analysis, reporting, private checkpoint storage, selected exports, audits, private publication/verification, historical checks, cleanup and billing. It stops on a failed stage and never commits or pushes Git changes. Exact commands/stdout are retained in separate stage logs. Later corrections must preserve the failed log and avoid resubmitting already completed GPU stages.

Once r1 scientific execution completed, `v7_retain_r1.py` started its private artifact retention while r2 trained. Separate `retention-r1-*.log` files and `r1/early-retention.json` preserve this operational overlap; it adds no GPU call or test inference. Reporting waits for that writer to finish, with explicit failure/timeout guards, before the main controller rechecks the idempotent artifact stages. Both writers must exit before final archival. Scientific source, datasets, selection and endpoints are unchanged.

Standalone stage commands follow this pattern; use the appropriate repetition and action rather than starting the whole controller over:

```bash
uv run --extra train --extra cloud --extra analysis modal run --profile amandogra2016 v7_app.py --repetition r1 --action train
uv run --extra train --extra cloud --extra analysis modal run --profile amandogra2016 v7_app.py --repetition r1 --action selection
uv run --extra train --extra cloud --extra analysis modal run --profile amandogra2016 v7_app.py --repetition r1 --action test
uv run --extra train --extra cloud --extra analysis python v7_artifacts.py r1 records
uv run --extra train --extra cloud --extra analysis python v7_ops.py analyse --repetition r1
```

Selection-temperature fitting, checkpoint selection and final calibration use distinct fresh partitions. Update zero remains eligible; development accuracy cannot select a checkpoint. Tests start only after the candidate/calibration prediction inventory is sealed. BF16 aggregate test inference and FP32 exported reference checks are kept distinct.

## Retained artifacts

- `runs/study-v7/`: master freeze, global bounded-call ledger, resource ownership, billing/cleanup and controller/failure receipts.
- `runs/study-v7/r1/` and `r2/`: independent frozen protocols, data, raw logits, trajectories, selection/calibration, model manifests and audit receipts.
- `artifacts/v7-preparation/`: CPU/style checks, original image-wrapper source, preparation logs and source-archive receipts.
- `artifacts/v7-source-snapshot/`: independently verified original pre-execution, corrected pre-recovery and final source archives.
- `artifacts/v7-releases/{r1,r2}/`: selected local inference bundles.
- `artifacts/v7-hub-verification/{r1,r2}/`: fresh pinned private-Hub verification bundles; identical owned weight copies are deduplicated only after successful checks.
- `reports/v7-evidence/`: compact metadata/log/source evidence with a checksum inventory; it does not contain all raw inputs, logits, optimizer states or weights.

Each repetition has a private HF checkpoint store containing eight additional scheduled weight files and a pinned inventory with both original-parent references. Intermediate weights are checked by pinned LFS SHA/size metadata, without claiming fresh inference verification. Each selected release is privately published, freshly downloaded at its exact commit and checked against eight FP32 references on CPU. A verified duplicate weight can then be atomically replaced by a hardlink to an identical retained local weight; the bundle remains complete and byte-identical. Disk-space checks bound staging, and interrupted downloads are preserved rather than silently overwritten.

Full optimizer/RNG recovery checkpoints are temporary within-run storage and are removed with the working Volume after completion. Persistent artifacts include every scheduled model weight, frozen inputs/settings/source and the complete logical training trajectory. A later experiment starts from retained weights with a fresh optimizer; historical progress pointers alone do not provide an optimizer-state restart after cleanup.

During execution, free laptop space fell to about 3.1 GiB. `v7_deduplicate_v6.py` reclaimed two previously verified V6 duplicate weight copies by atomically hardlinking them to their identical retained weights. It checked all members of both historical bundles before and after, plus the unchanged private-Hub verification receipt; every path and checksum remains intact. No historical result or receipt was rewritten, and no new inference/download is claimed. `runs/study-v7/disk-reclamation.json` records both pinned releases and the observed free-space change. Unrelated files and caches were untouched.

The numerical audit preserves worker metrics. Counts/types must match exactly, ordinary float differences stay within 1e-12 absolute error, and exceptional AURC differences require explicit extrema from sorting errors within confidence groups spanning at most 32 float64 epsilons. Original values and comparison evidence remain separate from any explanation.

`v7_raw_audit.py` separately reads serialized inputs and logits with the Python standard library, checking example/family/label identity, primary complete-world counts, old UNKNOWN contrasts and complete transformation families against the reports. Each repetition receives an immutable raw-count receipt with input/prediction hashes. Run it only after that repetition's records and analysis are complete; it does not score a model or change an endpoint.

```bash
uv run python v7_raw_audit.py r1
uv run python v7_raw_audit.py r2
```

Cleanup verifies retained predictions, all scheduled candidates and fresh private selected bundles before directly checking every owned app as stopped with zero tasks. Only the recorded V7 Volume can be deleted. Cost attribution uses the recorded owned resource IDs and an explicit complete-day interval, can lag and is not an invoice. Historical V1–V6 scientific sources/reports and frozen identities remain preserved.

After all controller and documentation writers complete, `v7_close.py` verifies closure requirements, exports compact evidence, builds the final archive, independently checks every member and writes an external closure receipt to avoid circular hashes. Full raw V7 records remain local and ignored by Git. Private model storage does not imply remote backup of the complete raw study or Git publication. A new robustness loss or architecture is a separate frozen experiment.
