# Remote publication

Completed on 2026-10-13 at the owner's request. Both destinations are private.

## GitHub

[Implementation, reports, and journal](https://github.com/adc77/decision-model-lab/tree/feat/data-foundation) are on `feat/data-foundation`. The first implementation commit is `c22b5f005751c68b313472aa46855a4ef4b69f3a`, authored and committed by `adc77 <amandogra2016@gmail.com>`. A later publication update records the verified backup, operation helper, and next proposal. `main` still contains the original plan; the implementation branch has not been merged.

The repository includes V1–V4 source, tests, configs, protocols, reports, figures, compact evidence, and `RESEARCH_LOG.md`. Historical archived source bytes and frozen scientific outcomes remain preserved. Generated SVG path whitespace is excluded from Git whitespace warnings through `.gitattributes`, without rewriting figures.

## Hugging Face

All eleven selected private model repositories were rechecked at their recorded commits. Fresh manifest SHA-256 checks and pinned remote weight size/SHA-256 metadata checks passed. Their historical fresh-download inference checks remain recorded in the study release reports. No selected model required a new upload.

[Private research archive](https://huggingface.co/axondendriteplus/decision-lab-research-archive/tree/06c0fb2050f4727d84f6873445794df21f416134), pinned at `06c0fb2050f4727d84f6873445794df21f416134`:

- 1,457 non-cache raw files from `data`, `runs`, and `artifacts`, preserving original paths in `records.tar.gz` (115,953,564 bytes).
- Datasets, oracle records, raw predictions/logits, training records, operational failures, billing/cleanup receipts, and nine historical source archives.
- All 41 retained local weight paths represented through checksum mappings: reuse exact selected release revisions for duplicate weights; upload eight additional unique blobs, including six V4 intermediate checkpoints.
- `manifest.json` SHA-256: `87815f401d9a170bd7691c427f2b670f47ef78daeebc702d4ec862dca2a599e0`.

Verification at `2026-10-13T12:37:02.577139+00:00`: a fresh pinned records download passed its bundle hash and all 1,457 member checksums. Every additional weight blob passed pinned remote LFS size/SHA-256 checks. Intermediate checkpoints were not independently downloaded and inference-tested during backup. The selected inference bundles retain their earlier full verification.

The first transfer uploaded the data but its final commit request failed with `ConnectionError` / `BrokenPipeError`; remote inspection showed only the initial repository commit. Resuming from the ownership receipt reused uploaded content, reported zero new data bytes, and completed the commit and verification. Original operation logs remain local with hashes recorded in Git; the failure traceback is retained in compact evidence. No GPU calls occurred.

Excluded: SDK caches, compiled Python, and the publication operation's raw files. Its compact receipts are in Git. Previously discarded optimizer recovery states and V2 intermediate checkpoints cannot be recovered and are not claimed as retained. Earlier unrecorded conversation text is not reconstructed.

## Restore and verify

Clone the implementation branch, authenticate to the owner's private Hugging Face namespace, and download the research archive at the exact revision above. Verify `manifest.json` against the recorded hash, then verify the archive and its members before extracting into a new directory. Weight mappings in `manifest.json` identify either a file in this archive or an exact repository/revision/file tuple in a selected model release. Verify each SHA-256 before placing it at its recorded original path. Do not overwrite an existing study. Selected inference bundles include their matching code, configs, tokenizer, and entry point; bare intermediate weights do not constitute a standalone release.

For this publication's verification procedure, read [the utility](../publish_research_archive.py) and [evidence inventory](publication-evidence/archive-manifest.json). `prepare` refuses to overwrite its dated backup directory; `upload` resumes only from this publication's private ownership receipt. This is an operation for the dated snapshot, not a generic overwrite tool.

## Validation and next work

All 69 CPU tests passed in 26.99 seconds. Lint, formatting, lock, credential-pattern scanning, and staged whitespace checks passed. Checks against the V4 final archive manifest confirmed 106 scientific/build files unchanged. Historical archives were scanned before transfer and their member bytes are covered by the verified records backup. Exact receipts are in [publication evidence](publication-evidence/validation.json).

[Next proposal](../docs/NEXT_EXPERIMENT.md): fresh worlds with multiple independently answerable entities and different TRUE/FALSE/UNKNOWN answers in the same context. First audit the oracle and data, then freeze a diagnostic and evaluate retained V3/V4 models. Review the evidence before more training. No V5 freeze or GPU submission has occurred.
