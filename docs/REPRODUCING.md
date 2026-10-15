# Reproducing Premise

Use Python 3.12. Clone `https://github.com/adc77/premise`, then run `uv sync --extra train --extra analysis --extra baseline`. The historical package/import name remains `decision_lab`.

## Read or run the selected models

Choose a public checkpoint from [MODELS.md](MODELS.md). Run `uv run python examples/predict.py --model <repo> --revision <commit>` for the exact recorded prompts in `examples/prompts.json`. The loader verifies all files listed in `artifact-manifest.json`. Use the pinned commit rather than the moving `main` branch for a reproducible download.

The included model bundle supports standalone inference through its own `infer.py`, `code/`, tokenizer, calibration, and dependency requirements. It uses a custom three-class head, not a stock chat-model or sequence-classification pipeline. Inputs are full prompts following `prompt-template.txt`, not isolated questions. Maximum length is 512 tokens; overlong inputs fail.

For the eight fixed artifact checks, call `decision_lab.release.verify_release(Path(download_directory))` in the locked environment. Those checks use FP32 and stored numerical tolerances. Aggregate study scores were measured using BF16. Fresh CPU inference is not an exact reproduction of every BF16 aggregate logit.

## Restore the frozen data

The public dataset `axondendriteplus/premise-worlds` contains configs `v7-r1`, `v7-r2`, `v8-r1`, and `v8-r2`. Use `huggingface_hub.snapshot_download(..., repo_type="dataset", revision=<pin>)` to retain exact original JSONL files, rather than reserializing rows. Verify every digest in `sha256.json`.

`data/<config>/` contains the input splits. `provenance/<config>/` preserves the family registry and original manifest. Model training needs its matching arm's `train_control` or `train_treatment`; do not concatenate them. V8 intentionally uses the same training rows in both arms.

The study datasets exclude recorded earlier worlds. Regenerating a full historical dataset byte-for-byte also requires its recorded exclusion inventory; do not run a later generator without those records and claim it recreated the original partition. The public frozen files are the supported path to those exact inputs.

## Verify reported metrics

Public `reports/v7-evidence` and `reports/v8-evidence` contain the independent raw-count receipts, numerical checks, selection records, protocol seals, and inventories. The release dataset includes exact source inputs. Model cards include the original full evaluation JSON and fixed references. The article's `assets/results.json` is a compact presentation of those records, not another experiment.

Full original per-question logits remain in the owner's retained run directories and are not claimed as a public raw-logit archive. To independently reproduce aggregate scores, run the released model on its matching test inputs using the original BF16 precision and grouping rules, and recompute counts. `v7_raw_audit.py`, `v8_raw_audit.py` and the study analysis source specify those checks; their historical CLI wrappers expect the original `runs/study-v*/...` layout and refuse to overwrite receipts.

Bootstrap intervals resample three-world rotation blocks within declared strata, conditional on the fixed selected weights. They do not measure a training-seed population. The fresh-seed comparisons and all endpoint definitions are in the frozen protocols.

## Tests and local article

Run `uv run --extra train --extra analysis --extra baseline python -m pytest -q`. The suite tests the symbolic oracle, partition invariants, generated inputs, paired loss, selection, checkpointing, and reporting checks. It does not submit Modal jobs.

Run `python3 research-article/build.py`, then `python3 -m http.server 8765 --bind 127.0.0.1 --directory research-article/dist`. The builder reads the committed compact study receipts under `reports`, so rebuilding the presentation works in a fresh checkout without private run directories. Article controls display saved predictions, not live inference.

## New training

Training is separate from inference and CPU verification. Study scripts require Modal authentication, historical parents/exclusion inventories, source seals, a new run directory and an explicitly bounded allocation. They enforce source/data identities and do not silently launch a new study from a fresh checkout. Full optimizer/RNG recovery checkpoints were temporary. Released selected weights support a new fine-tuning run, not an exact continuation of deleted optimizer state.

This release does not add training, new seeds, an architecture experiment or coefficient sweeps.
