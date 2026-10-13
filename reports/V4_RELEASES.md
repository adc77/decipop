# V4: verified releases, costs, and retained evidence

V4 is complete: controlled diagnostics, the matched additional-training comparison, frozen selection/calibration, fresh final tests, local candidate retention, private publication, independent CPU verification, and owned-resource cleanup. Implementation remains local on `feat/data-foundation`, uncommitted and unpushed. Read [V4_RESULTS.md](V4_RESULTS.md) for the positive targeted result and its transfer failures.

## Private pinned releases

| Arm | Private repository | Verified commit |
| --- | --- | --- |
| Control | [decision-lab-v4-smollm2-360m-control](https://huggingface.co/axondendriteplus/decision-lab-v4-smollm2-360m-control) | `f6fb417b7a7851fa3492412490ee7e39afa35815` |
| Mixed data | [decision-lab-v4-smollm2-360m-mixed-unknown](https://huggingface.co/axondendriteplus/decision-lab-v4-smollm2-360m-mixed-unknown) | `c4ee886cd3a718036297edb56b0202b590457cf2` |

Both selected update 1,878 after 60,003 presentations. Their final temperatures are 6.034315882573097 and 1.1300427719087858. Each bundle includes weights, tokenizer, configuration, calibration, inference code, reference predictions, study metadata, licenses, and a file inventory. Publication checked the private flag and pinned revision; a fresh download into a separate directory checked every declared file and all 8/8 FP32 reference predictions per model in a separate CPU process. GPU reload also passed 8/8 per model. Aggregate test predictions used GPU BF16 autocast; these FP32 artifact checks do not claim bit-identical BF16 probabilities on every test question.

Local releases: `artifacts/v4-releases/{control,treatment}`. Independent downloads: `artifacts/v4-hub-verification/{control,treatment}`. The publication receipt is [huggingface-release.json](v4-evidence/huggingface-release.json). No Hugging Face write credential entered Modal.

## Observed cost

Billing observation: **2026-10-13T10:54:33.272419+00:00**. The resource report starts at 2026-10-13 00:00 UTC, covers V4's execution day at hourly resolution, and filters only its five recorded app IDs and owned Volume ID.

| Owned resource category | Metered USD |
| --- | ---: |
| L4 GPU | 1.49386081 |
| CPU | 0.35330427 |
| Memory | 0.47803545 |
| Observed owned subtotal | **2.32520053** |

All five owned apps appear in the resource report. No additional owned storage or egress rows appear in this observation. This is an observed snapshot, not a final invoice or a claim that later accounting adjustments are impossible.

Workspace billed charges were **$0**. Workspace gross metered usage was $9.89950463, compared with $6.20106047 before V4: a $3.69844416 aggregate increase. That increase is not V4's exact cost; the report includes other workspace workloads. The conservative original-$30 margin calculated from workspace gross usage is $20.10049537, not an independently queried live credit-wallet balance. The summary reports $9.69 in applied credits and $0.20950463 in network-egress allowance.

Ten successful GPU calls reserved $3.5626555555555557 in compute ceilings; there was no failed GPU call or retry. The full worst-case compute allowance, including the unused recovery call, was $3.988790666666667 under the $4 compute cap, plus a $1 storage/egress reserve. Reservations are ceilings, not measured costs. [Billing attribution](v4-evidence/billing-attribution.json) preserves the scoped rows, snapshot, report range, and raw-report hash. Full raw report and summary remain in `runs/study-v4`.

## Verification and operational corrections

- All **69 CPU tests passed in 26.20 seconds**, including the three cleanup-verifier regression cases. The earlier 66-test result is preserved separately. The final suite used `python -m pytest`; a mistaken console-entry-point launch failed during collection, made no experiment call, and is retained in its own log.
- Lint, formatting for 108 Python files, dependency-lock verification, and Git whitespace checks passed. The [final-check receipt](v4-evidence/final-checks.json) contains exact commands and outputs.
- The [final audit](v4-evidence/final-audit.json) checked all **48,000 prediction rows**, protocol/selection identities, input alignment, finite logits, selected-weight hashes, the entire declared candidate pool, complete training exposure, and unchanged prior scientific source/protocol files. The ten post-test confusion matrices separately reconciled with the frozen aggregate accuracies.
- The initial private upload was progressing slowly. Its identified owned process was deliberately stopped; the controller consequently exited on that subprocess result. Publication resumed from its saved private creation receipt with four upload streams. Initial and resumed logs are retained. The [official Xet guide](https://huggingface.co/docs/hub/xet/using-xet-storage) documents fixed upload concurrency. This operational restart changed no weights, selection, calibration, tests, or GPU reservation.
- The initial cleanup deleted the owned Volume successfully, then its verifier incorrectly expected older stopped apps in Modal's recent-app list. The corrected helper queries each owned app's lifecycle and tasks directly. A read-only verification confirmed all five stopped, zero tasks, and the Volume absent at **2026-10-13T10:45:21.411650+00:00**. The initial source and failure log are retained; no earlier stop/delete stdout receipts are invented. The [cleanup receipt](v4-evidence/cleanup.json) distinguishes the initial actions from the successful read-only check.
- The first billing query requested month-to-date hourly rows and hit Modal's seven-day range limit. It wrote no successful billing observation. The corrected query starts on V4's preflight date; it uses daily resolution for ranges beyond seven days and refuses to overwrite retained observations. The initial helper and failure log are preserved.

No unrelated workspace resource was modified. Optimizer recovery files were intentionally discarded with the working Volume after retained weights and private releases were verified.

## Retention and provenance

The complete pool has ten candidate names but nine distinct weight sets: one shared parent and eight newly trained checkpoints. All are retained locally under `runs/study-v4/parent` and `runs/study-v4/candidates`; selected release weights are hardlinked to matching retained candidates. Intermediate weights, full synthetic inputs, raw logits, and operational logs remain local and ignored by Git. The two selected releases are backed by the private pinned Hub repositories above; the initial GitHub plan does not back up all bulk evidence.

| Record | Identity or location |
| --- | --- |
| Diagnostic protocol | `593025f62c7668bb77c4d89e1374c749d57d889a4e9e7792699df6b22465bfc3` |
| Additional-training protocol | `2ab1532953c1f8b4c2dfaab8d1b3a49ddc39a32cbeeff5e5413e6ed55a631650` |
| Selection/calibration freeze | `eacc80b38059a848a8f9565d688c43207bdd97a71c359cf35df062f5b694d050` |
| Frozen analysis script SHA-256 | `78387e3b292b23a65dc821c5cca651d191d6d1366d912ed602d7c6dcea1d7054` |
| Before-diagnostics archive SHA-256 | `40433308b02e8e6844d18ab07cb4b54213cdf000760daf3e99eb25dd93875713` |
| Before-training archive SHA-256 | `1299374ecf587f64c1b4838cab5706952038570658296288c7ba949d2c5f39c1` |
| Compact evidence | `reports/v4-evidence/inventory.json` |
| Final source/report archive | `artifacts/v4-source-snapshot/source-final-with-reports.tar.gz` |
| Final archive/member-hash receipt | `artifacts/v4-source-snapshot/manifest-final-with-reports.json` |

The pre-execution archives preserve their original members and hashes. Later operational/reporting helpers, regression tests, and post-test interpretation are included in the final archive, not claimed as members of the earlier archives. The final manifest is generated after closing the journal and reports to avoid a circular embedded archive hash. All archive members are independently hashed during creation. Existing V1–V3 outcomes, releases, and archives remain preserved.

The next proposed research step is a fresh diagnostic with multiple independently answerable entities and no distinguished-target cue, before another training or architecture change. It has not been submitted.
