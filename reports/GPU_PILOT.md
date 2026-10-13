# First GPU pilots

Both pretrained decision approaches run on an L4 and memorize the fixed 32-example training set. The full study is not complete; these pilots do not measure generalization.

## Access and billing verification

On 2026-10-12, authenticated read-only workspace lookup returned `amandogra2016`. The user's Usage & billing screenshot showed the Starter plan, $30 credits, $0 usage, a $42.50 total usage limit, and a saved $0 spend limit for the October 1–31 cycle.

The spend limit applies to charges after credits. The dashboard explicitly warns that volume storage can continue accruing after workloads stop. The temporary pilot volume was deleted after verifying that it contained only these two pilots and the public model cache; both reports are retained locally. Modal notes that deleted storage can remain billable for up to four days. See [volume pricing](https://modal.com/docs/guide/volumes#pricing).

The authenticated billing summary still returned $0 after the first run. This does not establish that the runs were free or that $30 remains. [Billing data can lag](https://modal.com/docs/guide/billing#generating-billing-reports). Granular billing visibility is documented for Team and Enterprise; this workspace is Starter.

## Runs

The first run used 40 steps per approach; the second restarted both from the same pinned pretrained weights and used 120 steps. The examples, seed, optimizer settings, and input mapping were unchanged. These are separate runs, not resumed training.

| Run | Answer-token accuracy | Direct-head accuracy |
| --- | ---: | ---: |
| 40 steps | 81.25% | 90.625% |
| 120 steps | 100% | 100% |

Raw reports:

- [40-step report](../runs/pilot/report.json), run `pilot-dcffda825de74dafbefe8a56e897c793`.
- [120-step report](../runs/pilot-fit/report.json), run `pilot-e1981ca855ab4860baa5180fd42d446e`.

Modal app links:

- [First pilot](https://modal.com/apps/amandogra2016/main/ap-ZsCOtwEhIFCDjGlmzKPJ9n).
- [Extended fit pilot](https://modal.com/apps/amandogra2016/main/ap-doZuNoSWEqy31PxSHRdgk2).

The raw reports are local, gitignored artifacts. No implementation changes or reports have been committed or pushed.

## Extended fit measurements

| Measurement | Answer token | Direct head |
| --- | ---: | ---: |
| Trainable parameters | 361,821,120 | 361,824,003 |
| Peak allocated GPU memory | 6.821 GiB | 6.825 GiB |
| Mean training step after warmup | 0.358 seconds | 0.353 seconds |
| First eight steps: mean loss | 1.7203 | 1.1189 |
| Last eight steps: mean loss | 0.000410 | 0.000115 |

Losses use different objectives and should not be compared as a measure of quality. Tiny-set accuracies use the restricted three-class decision. Neither the small timing difference nor memorization identifies a better model.

Both runs used full fine-tuning, FP32 parameters and optimizer state, BF16 autocast, activation checkpointing, microbatch four, seed 17, and the same pinned SmolLM2-360M revision. There was no LoRA, quantization, or model substitution. Peak allocated memory is not total GPU reservation or host memory.

## Budget accounting

At the [published resource rates](https://modal.com/pricing), the requested L4 plus four physical CPU cores and 32 GiB RAM costs $1.243584/hour. The preflight estimates $0.270134 per run if the configured startup, execution, and teardown windows are fully consumed, excluding image-builder costs and resource overages. Each pilot has a separate $0.50 allocation.

The second preflight used $29.50 as a conservative planning balance after reserving the first pilot's entire $0.50 allocation. That number was not a new account-balance measurement. Actual invoiced usage remains unreconciled.

The temporary volume listed about 0.677 GiB, mostly the public checkpoint cache. At $0.09/GiB-month, its unadjusted storage estimate was about $0.061 for 30 days; this does not include credits or storage allowances. It has now been deleted.

Extrapolating the measured training steps to three passes over 20,001 examples gives approximately 1.5 GPU worker hours per pretrained approach, or roughly $1.9 each at requested rates. This is a preliminary training-only estimate. Varying input lengths, accumulation, evaluation, checkpointing, scratch training, recovery, and artifact verification still need measurement before the complete protocol is frozen.

## Next implementation

Build checkpoint saving and exact resume, measure checkpoint and development-evaluation overhead, and run a bounded scratch-model GPU pilot. Then freeze the shared exposure and cost forecast before confirmatory training. No test-split neural predictions have been examined. Hugging Face write credentials are needed only for the later private artifact upload.
