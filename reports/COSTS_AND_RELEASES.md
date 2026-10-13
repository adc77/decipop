# Costs, releases, and reproducibility

Observed at **2026-10-12T18:13:57.640278+00:00** in personal Modal workspace `amandogra2016`, billing cycle October 2026. All experiment jobs had stopped and the temporary Volume had been deleted before this snapshot.

The workspace summary reports **$2.77152523 metered usage and $0 billed**. Credits applied: $2.77000000; egress allowance applied: $0.00152523. The screenshot verified $30 starting credits. This snapshot leaves approximately $27.23 of those starting credits before later adjustments.

## Actual usage versus planning

| Phase | Observed raw app cost | Modal app |
| --- | ---: | --- |
| 40-step pilot | $0.02423425 | [ap-ZsCOtwEhIFCDjGlmzKPJ9n](https://modal.com/apps/amandogra2016/main/ap-ZsCOtwEhIFCDjGlmzKPJ9n) |
| 120-step pilot | $0.03597254 | [ap-doZuNoSWEqy31PxSHRdgk2](https://modal.com/apps/amandogra2016/main/ap-doZuNoSWEqy31PxSHRdgk2) |
| GPU wiring/resume gate | $0.06855279 | [ap-hZxc3SHRo73JmiRb9LUhkZ](https://modal.com/apps/amandogra2016/main/ap-hZxc3SHRo73JmiRb9LUhkZ) |
| Token training and interrupted head training | $1.69919858 | [ap-f0sFxRF8YIWtuWIl9sR1nc](https://modal.com/apps/amandogra2016/main/ap-f0sFxRF8YIWtuWIl9sR1nc) |
| Failed head recovery | $0.12590333 | [ap-nfkAoTno4vy2ILUaM2GQXx](https://modal.com/apps/amandogra2016/main/ap-nfkAoTno4vy2ILUaM2GQXx) |
| Completed head recovery and scratch training | $0.56933600 | [ap-aYbkj5IMOcP49SpMYIKZtn](https://modal.com/apps/amandogra2016/main/ap-aYbkj5IMOcP49SpMYIKZtn) |
| Calibration, evaluation, export verification | $0.20368433 | [ap-xr1bogIvVv11wSyPiMN51D](https://modal.com/apps/amandogra2016/main/ap-xr1bogIvVv11wSyPiMN51D) |

The hourly report totals **$2.72688182** across the seven owned apps. The workspace summary's ephemeral-app subtotal is $2.77067491; the $0.04379309 gap is retained as an unresolved difference between provider aggregation views. It has not been assigned to a model or hidden. The workspace metered total is the primary observed budget figure.

The invocation ledger allocated **$13.14567**, including a $1 pilot allowance. It is a conservative timeout-based reservation, not expenditure. Calls: {'gate': 1, 'token': 8, 'head': 10, 'scratch': 8, 'evaluation': 3}. The two failed head calls retained their allocations. Neither the $20 compute cap nor the $10 reserve changed.

Requested worker resources were one L4, four physical CPU cores, and 32 GiB RAM. At the frozen rates, $0.7992 + 4 × $0.04716 + 32 × $0.007992 = $1.243584 per hour. Warm inference unit costs in RESULTS.md use this requested-resource calculation; they exclude startup, tokenization, transfers, and idle service capacity. See [Modal pricing](https://modal.com/pricing).

Provider usage can arrive late; this is an observed reconciliation, not a finalized October invoice. Both raw provider views and their timestamps remain in `runs/study-v1/billing-final-*.json`. The granular billing CLI returned data for this Starter workspace although the documentation describes programmatic granular reports for Team/Enterprise. See [Modal billing](https://modal.com/docs/guide/billing).

## Cleanup

Deleted only the project-owned `decision-model-lab-runs` Volume (`vo-yxb8EBBPjdW7JfHKOzZsFB`) at 2026-10-12T18:13:17.486322+00:00. Its recorded live files totaled 31,755,930,407 bytes. Before deletion, all three private Hub round trips and the local frozen-input/weight/prediction checks passed; 164 small remote metadata files were archived locally with a checksum manifest. No research job, serving endpoint, or schedule remains running.

Optimizer recovery states and the public-model cache were temporary and were deleted. Final weights, selected weights, the interrupted-selection diagnostic, model bundles, data, source snapshots, logs, raw predictions, metrics, and manifests are retained locally. Deletion does not recreate an unfinished training checkpoint: continuing training would require a new declared study, while inference and analysis use the retained bundles.

The billing snapshot shows no Volume charge yet. Storage accounting may arrive later, and deleted snapshots can remain billable for up to four days. Do not interpret deletion or the $0 compute spend limit as an immediate guarantee of zero future storage charges. See [Modal Volume pricing](https://modal.com/docs/guide/volumes#pricing).

## Private model releases

| Model | Private repository | Pinned revision | Fresh download |
| --- | --- | --- | --- |
| token | [axondendriteplus/decision-lab-smollm2-360m-token](https://huggingface.co/axondendriteplus/decision-lab-smollm2-360m-token) | [`b5db649b84cef324ac59d5115ac5c730a919c0d7`](https://huggingface.co/axondendriteplus/decision-lab-smollm2-360m-token/tree/b5db649b84cef324ac59d5115ac5c730a919c0d7) | SHA-256 manifest checks; all 8 FP32 reference decisions/logits pass |
| head | [axondendriteplus/decision-lab-smollm2-360m-head](https://huggingface.co/axondendriteplus/decision-lab-smollm2-360m-head) | [`9bd0a31fa732ed2e746906638e7b0c349a9afeb2`](https://huggingface.co/axondendriteplus/decision-lab-smollm2-360m-head/tree/9bd0a31fa732ed2e746906638e7b0c349a9afeb2) | SHA-256 manifest checks; all 8 FP32 reference decisions/logits pass |
| scratch | [axondendriteplus/decision-lab-transformer-14m](https://huggingface.co/axondendriteplus/decision-lab-transformer-14m) | [`f09dbc99d615ba37e143e33ae34c6fea68d61c56`](https://huggingface.co/axondendriteplus/decision-lab-transformer-14m/tree/f09dbc99d615ba37e143e33ae34c6fea68d61c56) | SHA-256 manifest checks; all 8 FP32 reference decisions/logits pass |

Fresh download checks ran in separate CPU processes with bundled inference code, fixed prompt inputs, exact decisions, and logit tolerances of atol=0.001 and rtol=0.001. GPU reload checks had already passed for all three artifacts. The published cards include the post hoc counterfactual failure breakdown; this metadata addition did not change weights, code, tokenizer, or calibration.

Each bundle contains safetensors, tokenizer, custom inference code, model configuration, training and evaluation provenance, calibration/deferral settings, license attribution, fixed verification inputs, and a checksummed allowlist. Neither credentials nor optimizer states were uploaded. These use custom loaders rather than a stock causal-LM loading path.

## Retained provenance

- Training protocol: `b622cfa2d34eb2fb779a4692ceb69290e053d1dcd1b307da68d203bbd25f4d3e`.
- Calibration/checkpoint freeze: `b7104a02b532aae75ccb59ee882f46e5831449e86633638f7b33162bc9dd074f`.
- Evaluation code: `99a2d5d7df4d64dcdf2e58b2c97106b1946ff22846b66cede41496db1dc0703e`.
- Public base revision: `HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.
- Seed 17; three training passes; 60,003 questions and 1,878 optimizer updates per model.
- Local protocol/source/lock verification, 19 frozen input checks, three final-weight checks, three selected-weight checks, and 60 required prediction-file counts passed.
- Final CPU suite: **50 passed in 21.40 seconds**; Ruff checks, format checks, and locked dependency verification passed.

Phase timing separates dispatch-to-worker time from worker body time. Dispatch includes queueing and image/container startup; it does not isolate platform cold start.

| Phase | Dispatch to worker (s) | Worker body including Volume commit (s) | Client wall (s) |
| --- | ---: | ---: | ---: |
| calibrate | 416.51 | 83.54 | 500.17 |
| evaluate | 282.45 | 396.50 | 682.50 |
| verify | 9.94 | 35.92 | 45.96 |

Implementation and reports remain uncommitted on `feat/data-foundation`; GitHub main still contains the reviewed plan commit. The pre-run source archive is preserved alongside a final source/report archive and per-file hashes under `artifacts/source-snapshot`. Private model revisions pin the inference artifacts; the source archives pin the uncommitted implementation.

Commands and dependency setup: [SETUP.md](../docs/SETUP.md). Model/study diagrams: [ARCHITECTURE.md](../docs/ARCHITECTURE.md). Scientific results: [RESULTS.md](RESULTS.md). Central limitation: [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md). Learning walkthrough: [LEARNING.md](LEARNING.md).
