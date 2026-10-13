# V2 costs, releases, and retention

Completed on 2026-10-13 local time. Billing snapshot: `2026-10-12T21:37:07.306268+00:00`.

## Observed cost

Owned v2 resources report **$2.13302126 metered usage**. Workspace billed charges are **$0**, with **$24.63000000 credits remaining** at this snapshot. Attribution includes only the recorded v2 app and Volume IDs; unrelated workspace apps are excluded. Workspace credit changes are not used as a project-cost estimate.

| Resource | Metered USD |
| --- | ---: |
| L4 | 1.37017609 |
| Network Egress | 0.00033769 |
| Memory | 0.43845635 |
| CPU | 0.32405113 |

The fixed compute limit was $5, with a separate $1 storage/accounting reserve. Planned worst-case allocations total $4.86863136; actual submitted-call ceilings total $3.48825312. All nine invocations completed: one pilot, three training chunks per arm, one evaluation, and one fresh GPU verification. No call failed and no retry was used. Allocation ceilings are not observed charges.

The requested L4, four physical CPU cores, and 32 GiB RAM rate was $1.243584/hour, from the provider rates checked before submission. See [Modal pricing](https://modal.com/pricing). Network egress allowances and credits are workspace-level adjustments; see [Modal billing](https://modal.com/docs/guide/billing). No owned-Volume rows appear in this billing report. The working Volume is deleted. Storage accounting may arrive later, so this is a snapshot rather than a final invoice; [Modal Volumes pricing](https://modal.com/docs/guide/volumes#pricing) describes storage allowances and deleted-storage reporting delay.

## Private selected-model releases

These are the preselected update-250 heads, each with 8,000 training presentations. They predict TRUE on every counterfactual test input. They are retained as negative research results; the later full-run weights are separate local artifacts.

| Arm | Private repository | Pinned revision | Fresh CPU inputs |
| --- | --- | --- | ---: |
| control | [axondendriteplus/decision-lab-v2-smollm2-360m-control](https://huggingface.co/axondendriteplus/decision-lab-v2-smollm2-360m-control) | `7cb185a64126d6812e3e96b2bb614cb20a33dba5` | 8 |
| treatment | [axondendriteplus/decision-lab-v2-smollm2-360m-counterfactual](https://huggingface.co/axondendriteplus/decision-lab-v2-smollm2-360m-counterfactual) | `bb6e562f66a3689da2a8b3a5ccd5d4135ff12471` | 8 |

Both repositories were confirmed private at the pinned revision. Every manifest file hash passed, and separate CPU processes loaded the bundled inference code to reproduce eight fixed FP32 decisions and logits per model within the recorded 0.001 absolute/relative tolerances. Fresh GPU reloads also passed before publication. Aggregate research evaluation uses BF16 autocast; artifact verification uses FP32.

Model cards and `posthoc-selection-analysis.json` state the constant-TRUE result and checkpoint-selection limitation. Those metadata additions occurred after GPU verification; weights, tokenizer, inference code, and calibration were unchanged. The final annotated manifests were checked again through the private Hub round trips.

## Retained local artifacts

- `data/v2` and `artifacts/v2-audit`: both training arms, shared development/calibration/tests, family registers, and tokenizer audits.
- `runs/study-v2`: frozen protocols, comparison, pilot, call ledger, trajectories, calibration, ten prediction files, statistics, publication receipts, billing attribution, and cleanup records.
- `artifacts/v2-releases` and `artifacts/v2-hub-verification`: annotated selected releases and independently verified private downloads.
- `artifacts/v2-final-weights/{control,treatment}`: full-run final weights and progress, checked against durable checkpoint hashes; optimizer/RNG state is not retained.
- `artifacts/v2-modal-records`: 64 small remote metadata records, with an inventory in `runs/study-v2/retention.json`.
- `reports/V2_RESULTS.md`, `V2_LEARNING.md`, and `v2-figures`: negative selected-model result, selection analysis, development-only worked example, and three scientific plots, each in SVG and PNG.

The fixed development example was chosen before test scoring. The final treatment answers TRUE/FALSE/UNKNOWN correctly on its three worlds, while both selected heads and the final control answer TRUE throughout. This is a descriptive FP32 development demonstration, not an additional final-model test result. Selected probabilities are calibrated; final probabilities are explicitly uncalibrated.

## Source and cleanup provenance

Frozen comparison identity: `be6db15a671430027698c88f6d48068c260285b6e0cec649b5f0d3c9656648bd`.

Pretraining source archive: `artifacts/v2-source-snapshot/source-before-training.tar.gz`, SHA-256 `beffbfa23880cc882d95b2d9ca4cc3173f8317a011e7de6dc9c8dceb92d61b81`. The archive includes the actual pretraining scientific code and operational controller, without credentials. The final source/report archive and its manifest are retained beside it.

All frozen input hashes and all prediction IDs, labels, counts, and finite logits were audited. Both owned v2 Modal apps are confirmed stopped. `decision-model-lab-v2-runs` was deleted after local retention and both Hub checks; absence from the Volume list was verified. Other workspace resources were not touched. V1 artifacts and model revisions remain preserved.

Implementation remains local and uncommitted on `feat/data-foundation`. No implementation GitHub commit or push was performed. Compact project-scoped evidence is in `reports/v2-evidence`; complete records remain in the ignored local run/artifact directories.
