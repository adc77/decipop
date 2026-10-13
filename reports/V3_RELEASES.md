# V3: verified artifacts, execution, and cost

Four distinct selected candidates are published in new private Hugging Face repositories. V1/V2 repositories are preserved. Each bundle contains exact weights, the tokenizer, a custom inference loader, frozen protocol and selection/calibration records, fresh results, pinned requirements, license, file hashes, and eight FP32 reference inputs.

| Candidate | Private repository | Verified commit |
| --- | --- | --- |
| control-early | [axondendriteplus/decision-lab-v3-smollm2-360m-control-early](https://huggingface.co/axondendriteplus/decision-lab-v3-smollm2-360m-control-early) | `0af460093201087dae7b60ad16c78a994095f325` |
| control-final | [axondendriteplus/decision-lab-v3-smollm2-360m-control-final](https://huggingface.co/axondendriteplus/decision-lab-v3-smollm2-360m-control-final) | `96992f4ef0c3e30dafe02297b449a323fd3c6dcc` |
| treatment-early | [axondendriteplus/decision-lab-v3-smollm2-360m-treatment-early](https://huggingface.co/axondendriteplus/decision-lab-v3-smollm2-360m-treatment-early) | `fc2a810f0ead5039ae1364cde9667839b932aa0d` |
| treatment-final | [axondendriteplus/decision-lab-v3-smollm2-360m-treatment-final](https://huggingface.co/axondendriteplus/decision-lab-v3-smollm2-360m-treatment-final) | `d165bf46ed93deee9697143204f78bdccdddb17d` |

## Verification and retention

All four bundles passed eight-input GPU reload checks, then a pinned private Hub download with complete file-hash verification and an independent CPU reference-inference process. Published weights and temperatures match the frozen selected candidates. Reference checks use FP32; aggregate scoring uses GPU BF16 autocast.

All 59 CPU tests passed. Full-project lint and formatting checks, dependency-lock validation, and Git whitespace checks passed. The final audit checked every one of 54,400 raw prediction rows against its input identifier, family, label, and frozen file hash. This counts repeated scoring of 13,600 input rows across four candidates, not 54,400 independent questions. V1/V2 frozen scientific source hashes and the V3 predeclared protocol text remain unchanged.

Raw inputs, all four retained candidates, selection/calibration logits, test logits, complete results, and receipts remain locally under `runs/study-v3`. Self-contained published bundles and pinned verification downloads are under `artifacts/v3-releases` and `artifacts/v3-hub-verification`. The original V2 final weights are preserved.

The app stopped after two completed GPU calls; no failed calls or retries. The owned Volume `decision-model-lab-v3-runs` was deleted only after retention and private verification. Its absence and zero owned running containers were checked. No unrelated workspace resources were modified.

## Observed cost

The owned app/Volume resource report totals **$0.32294453** at `2026-10-13T05:45:50.592880+00:00`. The monthly workspace report shows **$0 billed**.

| Resource | Observed metered USD |
| --- | ---: |
| CPU | 0.04906931 |
| L4 | 0.20748123 |
| Memory | 0.06639399 |

The monthly workspace metered total increased by $0.33000000 between the saved snapshots. That aggregate delta and the owned resource subtotal are not identical and remain unreconciled; other workloads, delayed aggregation, or adjustments may affect the workspace view. We attribute only the currently listed owned resource rows, rather than treating the workspace delta as this project's exact cost.

No Volume charge rows appear in this snapshot. This is observed metering, not a final invoice; billing and deleted-storage accounting can update later. The owned working storage has been removed.

The predeclared compute allowance was $2 with a $1 storage/egress reserve. Maximum planned compute allocation including the unused recovery slot was $1.32856224; submitted ceilings for the two successful calls total $0.95479616. These are ceilings used by the submission ledger, not actual charges.

Rates were verified at [Modal pricing](https://modal.com/pricing): L4 $0.7992/hour, CPU $0.04716/physical-core-hour, RAM $0.007992/GiB-hour. Four requested physical cores plus 32 GiB and one L4 cost $1.243584/hour at those rates. No endpoint, schedule, or persistent GPU service was deployed.

## Source and evidence

Code and reports remain local and uncommitted on `feat/data-foundation`. GitHub main remains the previously authorized initial plan. This experiment does not commit or push implementation changes.

The exact source before inference is archived at `artifacts/v3-source-snapshot/source-before-inference.tar.gz` with SHA-256 `bd52bdc4dff1fbc8f60bbff7acfa7cd9e1ad87ec065cb006f4cddb7cd24146bb`. The final source and report archive is `artifacts/v3-source-snapshot/source-final-with-reports.tar.gz`; `manifest-final.json` beside it records every included file and the archive hash. Neither archive contains credentials or model weights.

Compact records and their complete SHA-256 inventory are in `reports/v3-evidence`. Full raw records and resource reports remain in the ignored local study directory. Read [V3_RESULTS.md](V3_RESULTS.md) for outcomes and limitations and [V3_LEARNING.md](V3_LEARNING.md) for numerical examples.
