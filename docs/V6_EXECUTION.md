# V6 execution and retained evidence

Read [the scientific protocol](V6_PROTOCOL.md), [the packaging amendment](V6_PACKAGING_AMENDMENT.md) and [the journal](../RESEARCH_LOG.md) together. The original protocol remains immutable; the amendment has a separate sealed identity. The journal distinguishes running, completed and verified stages.

## Execution sequence

These are historical execution commands, not instructions to rerun the opened experiment. Existing study paths, call names, predictions and archives are guarded against reuse. A replication needs new seeds, paths, resource ownership and a new protocol before submission.

```bash
uv run --extra train --extra cloud --extra analysis python -m pytest -q
uv run --extra train --extra cloud --extra analysis python v6_prepare.py
uv run --extra train --extra cloud --extra analysis python v6_snapshot.py before-inference
uv run --extra train --extra cloud --extra analysis modal run --profile amandogra2016 v6_app.py --action pilot
uv run --extra train --extra cloud --extra analysis modal run --profile amandogra2016 v6_app.py --action train
```

The first training call failed during candidate packaging after a complete update-250 checkpoint. It remains failed in the allocation ledger. Recovery verifies its saved weight, optimizer/RNG and progress hashes before resuming. The CPU regression compares interrupted/recovered final weights with uninterrupted training exactly. The operational amendment changes export packaging and provenance handling, leaving data and update math unchanged.

```bash
uv run --extra train --extra cloud --extra analysis python v6_snapshot.py before-recovery
uv run --extra train --extra cloud --extra analysis modal run --profile amandogra2016 v6_app.py --action recovery
uv run --extra train --extra cloud --extra analysis python v6_continue.py
```

`v6_continue.py` requires a successful reserved recovery, then runs the remaining training calls. `v6_finish.py` runs checkpoint selection, final tests, raw-record retention, analysis, reporting, candidate storage, selected exports, audits, private publication, cleanup and billing in sequence. A failed stage stops the controller; its log remains separate from any correction. It never commits or pushes Git changes.

Selection fits each candidate's temperature on selection_fit and selects calibrated NLL on selection. Final temperature fitting uses calibration. Development accuracy cannot select a model. Final inference starts only after the selection record is sealed. Both selected arms use the same fresh test inputs; the unchanged parent supplies the retention baseline.

## Artifact boundaries

- `runs/study-v6/`: frozen original/amended protocols, serialized data, raw logits, trajectories, call reservations, failures and timestamped receipts.
- `artifacts/v6-source-snapshot/`: independently checked source archives before inference, before recovery and after closure; the incomplete archive attempt is preserved separately and has no verified manifest.
- `artifacts/v6-releases/`: selected inference bundles linked to verified local weights.
- `artifacts/v6-hub-verification/`: fresh pinned private-Hub downloads checked on CPU against fixed FP32 references.
- `reports/v6-evidence/`: compact receipt/log/source evidence with its own checksum inventory; this does not contain all raw logits or optimizer states.

Eight additional scheduled weight files transfer one at a time to private Hugging Face checkpoint storage. Each is checked against pinned LFS size/SHA metadata before its owned staging copy is removed; this check does not claim intermediate inference verification. The two update-zero aliases reference the existing pinned parent. Selected bundles receive fresh private-Hub downloads and CPU inference checks. No Hugging Face write credential enters a Modal worker.

Cleanup requires verified retained predictions, candidate storage and selected releases, followed by direct stopped/zero-task checks for every owned app. Only the recorded V6 Volume can be deleted. Billing attribution uses the owned app/Volume IDs and a timestamped complete-day API interval; metering can lag and is not a final invoice.

Final closure checks the original/amended provenance chain, frozen data/build files, retained evidence inventory, private release verification, failed-call recovery and cleanup. Documentation is completed before the final source archive is built to avoid concurrent writer changes. Full raw run records remain local; Git publication is a separate owner request.

## Artifact-stage corrections

A transient Modal storage DNS failure interrupted the second-to-last weight download after six checkpoint uploads were verified. The failure log and 80 MiB partial file are preserved. `v6_finalize.py` resumes verified candidate storage and the remaining export/audit/private-verification/cleanup/billing stages; it never reruns training, selection or tests.

The original strict dictionary-equality audit missed expected cross-platform float roundoff. `v6_metric_audit.py` preserves exact counts/types, bounds ordinary float errors by an absolute 1e-12, and checks exceptional AURC differences against explicit near-tie ordering extrema. Confidence groups span at most 32 float64 epsilons. The original helper, all observed differences and validation receipts remain separate from the frozen results. New rejection tests verify that these bounds cannot admit changed counts, nonroundoff scores or an AURC outside its permissible tie-order interval.
