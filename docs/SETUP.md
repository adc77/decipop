# Setup and study reproduction

CPU execution on this laptop is authorized for this personal project. Source publication is tracked in [the publication record](../reports/PUBLICATION.md). Bulk data, raw runs, and weights are excluded from Git and require the separately pinned Hugging Face backups documented there.

## Installed environment

The project uses Python 3.12 and `uv.lock`. The local environment includes the data, test, baseline, training, and Modal dependencies.

To reproduce it:

```sh
uv sync --frozen --extra baseline --extra train --extra cloud --extra analysis
uv run ruff check .
uv run --extra train --extra cloud python -m pytest -q
```

## Modal access

Authenticate in your own terminal from the project directory:

```sh
cd /Users/axon_dendrite/decision-model-lab
uv run --extra cloud modal setup
```

Choose the personal workspace intended for this experiment. Tell the agent the workspace name and remaining applicable credits shown in the billing dashboard. Verify the billing-cycle end date, existing usage, and a $0 net spend limit before launching the pilot. Do not paste token secrets into chat.

A $0 spend limit protects compute spending beyond applicable credits. The dashboard warns that Volume storage can continue accruing after workloads stop. Remove temporary pilot storage after preserving local reports; retain training checkpoints only while needed, then export verified artifacts. Deleted storage may remain billable for up to four days. See [Modal Volume pricing](https://modal.com/docs/guide/volumes#pricing).

Modal credentials stay in the SDK's local configuration or protected environment. `.env.example` contains empty placeholders for noninteractive configuration; it is not an authenticated setup.

## Hugging Face access

Tell the agent your Hugging Face username or namespace. Authenticate locally for later private model uploads:

```sh
uv run --extra train hf auth login
```

Use upload permissions for the intended model repositories. Training uses a public checkpoint and does not need your write token inside a GPU worker.

## Local data commands

Generation refuses to overwrite a dataset. Choose a new output directory for a revised generator or configuration.

```sh
uv run decision-lab generate --config configs/dataset.yaml --output data/new-pilot
uv run decision-lab validate --dataset data/new-pilot
uv run decision-lab baseline --dataset data/new-pilot --seed 17
uv run decision-lab audit-tokens --dataset data/new-pilot --output artifacts/new-audit --pilot-config configs/pilot.yaml --scratch-config configs/scratch.yaml
```

The unigram baseline uses training and development data only. Token auditing examines input lengths and does not evaluate neural predictions on final test labels. The original pilot was subsequently frozen as the confirmatory dataset for `study-v1`.

## GPU pilot, after access and billing verification

The next run measures memory, tiny-set learning, and throughput for both pretrained output approaches. It is not the full training experiment.

First run the local budget check with the actual verified remaining amount:

```sh
uv run decision-lab pilot-preflight --config configs/pilot.yaml --remaining-usd 30
```

Then, once the workspace and billing settings are verified:

```sh
uv run modal run modal_app.py --dataset data/pilot-v3 --token-audit artifacts/token-audit-v2/token-audit.json --remaining-usd 30 --output runs/pilot/report.json
```

Replace `30` with the verified remaining credits. The preflight is an estimate, not authenticated billing verification. The first image build and actual resource usage may add cost beyond the configured worker estimate.

The worker uses one L4, has explicit startup/execution timeouts, permits one container, and has no whole-job retry. It downloads the pinned public model on Modal, runs a bounded pilot, writes results to its working Volume, and returns a local report. It does not upload models to Hugging Face or create an inference endpoint.

Read the returned memory and throughput measurements before setting the common full-training exposure cap. If the tiny-set check has not fit yet, inspect its losses and wiring before scheduling a longer run.

## Frozen study

The initial pilots above are complete. The main controller runs one GPU job at a time and resumes through checksummed recovery checkpoints. These commands reproduce a new study after generating and validating its inputs; do not overwrite or refreeze the existing study after evaluation.

```sh
uv run modal run study_app.py --action gate
uv run modal run study_app.py --action freeze
uv run modal run --detach study_app.py --action train
```

Before final test scoring, train and calibrate the CPU baselines exactly once:

```sh
uv run python - <<'PYCODE'
from pathlib import Path
from decision_lab.evaluation.baselines import prepare_baselines
from decision_lab.evaluation.config import load_evaluation_config
prepare_baselines(Path('data/pilot-v3'), Path('runs/study-v1/baselines'),
                  ['TRUE', 'FALSE', 'UNKNOWN'], load_evaluation_config(Path('configs/evaluation.yaml')))
PYCODE
```

After all three training states are complete, run calibration, final evaluation, and fresh GPU artifact verification. The controller freezes all calibration temperatures and selected checkpoints before reading final test predictions.

```sh
uv run modal run --detach eval_app.py --action all
```

Then generate the local baseline evaluation and report:

```sh
uv run python - <<'PYCODE'
from pathlib import Path
from decision_lab.evaluation.baselines import evaluate_baselines
from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.evaluation.report import write_report
study = Path('runs/study-v1')
baselines = evaluate_baselines(Path('data/pilot-v3'), study/'baselines', study/'test-inputs',
                               load_evaluation_config(Path('configs/evaluation.yaml')))
write_report(study, Path('reports/RESULTS.md'), baselines)
PYCODE
```

Release operations require prior artifact verification. They upload only files listed in the checksummed model manifest and pin Hub commits. Fresh download verification uses a separate CPU process for each model.

```sh
uv run python release_models.py download --namespace axondendriteplus
uv run python release_models.py publish --namespace axondendriteplus
uv run python release_models.py verify --namespace axondendriteplus
```

Do not delete the working Volume until all release bundles, final and selected weights, logs, raw predictions, and manifests needed for this study are preserved locally and verified. No persistent inference service is part of this experiment.

## Completed v1 artifacts

The original `study-v1` is complete and its working Modal Volume has been deleted. The reproduction commands above would require a new working Volume and a new declared study; do not rerun them against the completed protocol after reading its test results. The inference bundles and retained local data remain available.

Read `reports/COSTS_AND_RELEASES.md` for the private repositories and exact verified Hub revisions. Compact protocol, evaluation, billing, release, and cleanup evidence is in `reports/evidence`; complete local logs and raw predictions remain in `runs/study-v1`.

To inspect the already downloaded head model using a JSON list of complete prompts:

```sh
uv run python artifacts/hub-verification/head/infer.py --input prompts.json
```

Each prompt must include the facts, rules, statement, and answer marker shown in `reports/LEARNING.md`. Replace `head` with `token` or `scratch` for the other verified bundles. This is FP32 inference with calibrated probabilities; it does not make the checkpoints reliable outside their evaluated distribution.

## V2 matched-data study

The new experiment is specified in [V2_PROTOCOL.md](V2_PROTOCOL.md). Its inputs are `data/v2/{control,treatment}`, tokenizer audits are `artifacts/v2-audit/{control,treatment}`, and records are `runs/study-v2`. Those paths are local and ignored by Git.

The generator reuses the same parent theories between training arms, keeps the development/calibration/tests identical, registers every changed world, and checks canonical overlap against v1. For exact regeneration, use the saved dataset manifest's generation settings; the frozen comparison and arm protocols separately record the operational settings used for training.

After generating, validating, auditing, and recording a fresh billing snapshot, the commands used for this study are:

```sh
uv run modal run --detach v2_app.py --action pilot
uv run modal run --detach v2_app.py --action all
```

`all` freezes the comparison, trains the two heads sequentially, freezes both calibrations before test scoring, exports the bundles, and verifies GPU reloads. The local invocation ledger retains failed-call allocations. The main controller checks runtime settings against the frozen protocols. A development lexical baseline is a separate CPU diagnostic; it never reads final test predictions.

This run froze the comparison locally before `all`, then explicitly uploaded that comparison, both arm protocols, and the test-input manifests to the owned Volume. When a local comparison already exists, the controller assumes those same frozen files have been uploaded. A fresh reproduction should let the controller create and upload its new comparison rather than copying this completed study's paths.

Generate the report and perform the private release after evaluation:

```sh
uv run python - <<'PYCODE'
from pathlib import Path
from decision_lab.v2.report import write_report
write_report(Path('runs/study-v2'), Path('reports/V2_RESULTS.md'))
PYCODE
uv run python v2_release.py download
uv run python v2_release.py publish
uv run python v2_release.py verify
uv run python v2_retain.py
uv run python v2_worked_example.py
uv run python v2_selection_plot.py
```

`v2_retain.py` requires both private Hub round trips, retains the full-run final weights separately from the selected releases, and records small remote metadata with checksums. It does not retain optimizer state or delete the working Volume. Cleanup follows only after all retained inputs, predictions, weights, and receipts have been checked.

`v2_worked_example.py` uses the development family chosen before test scoring. It loads each independently verified Hub bundle on CPU, then compares its selected weights with the retained final weights. Selected probabilities use the corresponding frozen calibration; final probabilities are explicitly uncalibrated. This FP32 demonstration is a descriptive example, separate from aggregate GPU BF16 evaluation.

Do not refreeze or retrain a completed study after reading its tests. Subsequent experiments need new reserved families and declared protocols. The two release repositories use new v2 names and preserve the v1 artifacts.

## V3 checkpoint selection and calibration

Read [V3_PROTOCOL.md](V3_PROTOCOL.md) for the frozen design. V3 uses only the retained step-250 and step-1878 weights from each V2 arm. It performs no training. Its raw inputs, candidate copies, predictions, and execution records live in the ignored `runs/study-v3` directory. The original final weights remain in `artifacts/v2-final-weights`.

The commands used for the study are:

```sh
uv run --extra train --extra cloud pytest -q
uv run --extra train --extra cloud python v3_prepare.py
uv run --extra train --extra cloud modal run v3_app.py --action all
uv run --extra train --extra cloud python v3_release.py download
uv run --extra train --extra analysis python v3_report.py
uv run --extra analysis python v3_plot.py
uv run --extra train --extra cloud python v3_worked_example.py
uv run --extra train --extra cloud python v3_release.py publish
uv run --extra train --extra cloud python v3_release.py verify
uv run --extra train --extra cloud python v3_audit.py
```

`v3_prepare.py` verifies V2 provenance, assembles the four retained candidates, excludes prior worlds, validates all formal answers and renderings, audits token lengths, and freezes inputs, source hashes, and call allowances. It refuses to overwrite an existing study directory.

`v3_app.py` uploads the frozen candidates and inputs to the owned temporary Volume. The selection worker fits temperatures on selection-fit families and chooses checkpoints on separate selection families. It then fits final temperatures on the independent calibration partition and durably seals those records. The test worker refuses to start without that freeze, scores every fixed candidate once, exports each distinct selected candidate, and verifies GPU reloads. The local allocation ledger retains submissions and failures; automatic retries are disabled.

`v3_release.py download` checks exported file hashes and retains every remote raw prediction and small execution record. Exported weight files are linked from their already retained local candidates only when their hashes match the GPU export manifest. Publication creates new private repositories and pins their commits. Verification downloads each pinned bundle into a fresh directory and checks all files and FP32 reference predictions in a separate CPU process. No write credential enters Modal.

`v3_worked_example.py` reads recorded selection logits without additional inference. It chooses the first sorted matched selection family and the first sorted original-style UNKNOWN query; both are explicitly post hoc descriptive examples. It generates [V3_LEARNING.md](../reports/V3_LEARNING.md). `v3_audit.py` requires all private download checks, validates every prediction against its formal input and frozen checksum, checks selected weights and calibration, and verifies that the prior studies' frozen sources remain unchanged. Cleanup follows this audit.

Read [the results](../reports/V3_RESULTS.md) and [release evidence](../reports/V3_RELEASES.md) after completion. Compact evidence is under `reports/v3-evidence`; complete raw records stay under `runs/study-v3`. Aggregate study predictions use GPU BF16 autocast; artifact reference checks use FP32. Temperatures fitted on BF16 logits need not produce bit-identical FP32 probabilities.

A completed study's Volume is deleted after local retention and private verification. Do not rerun or refreeze these commands against seen test families. A subsequent study needs its own input seed, study paths, Volume, protocol, and publication names. Do not overwrite the V1/V2/V3 evidence or use their test results to choose checkpoints.
## V4: controlled diagnostics and mixed UNKNOWN training

The running record is [RESEARCH_LOG.md](../RESEARCH_LOG.md). Read [V4_PROTOCOL.md](V4_PROTOCOL.md) for the frozen design and [V4_DIAGNOSTICS.md](../reports/V4_DIAGNOSTICS.md) for the controlled gap that motivated training.

These commands document this study's execution. Preparation refuses an existing parent/data root, submission refuses reused call names, and inference refuses overwriting predictions. A new experiment needs a new study identity and paths; do not rerun these commands over the frozen V4 records.

```bash
uv run --extra train --extra cloud python -m pytest -q
uv run --extra train python v4_prepare.py diagnostic
uv run --extra train --extra cloud modal run v4_app.py --action diagnostic
uv run --extra train --extra cloud modal run v4_app.py --action pilot
uv run --extra train python v4_report.py diagnostics
uv run --extra train python v4_prepare.py training
uv run python v4_snapshot.py before-training
uv run --extra train --extra cloud modal run v4_app.py --action train
```

After both arms finish, `v4_finish.py` executes selection, tests, retention, analysis, artifact audits, and private publication in order. It stops on a failed call or command. Its wait/poll limits are explicit CLI arguments. Cleanup and the final source/report archive follow verified retention.

```bash
uv run --extra train --extra cloud python v4_finish.py
```

Standalone post-training commands, used by that controller:

```bash
uv run --extra train --extra cloud modal run v4_app.py --action selection
uv run --extra train --extra cloud modal run v4_app.py --action test
uv run --extra train --extra cloud python v4_retain.py records
uv run --extra train --extra cloud python v4_retain.py candidates
uv run --extra train --extra cloud python v4_retain.py releases
uv run --extra train --extra analysis python v4_report.py results
uv run --extra train python v4_learning.py
uv run --extra train python v4_audit.py
uv run --extra train python v4_release.py
```

Scientific code is in the separate `v4_lab` package, preserving the earlier `decision_lab` source hashes. Both arms start from retained V3 treatment-final weights with fresh optimizers. The shared parent is update zero in each five-candidate pool; eight additional checkpoints are retained. Model selection and final calibration use distinct families and are frozen before tests.

After the controller completes, the post-test interpretation records descriptive label counts without changing the frozen analysis. Cleanup checks retained hashes and private pinned downloads before stopping only the recorded owned apps and deleting the owned working Volume. Billing attribution uses only those recorded resource IDs; the workspace aggregate is reported separately. Once reports and the journal are updated, compact evidence and the final source archive are saved:

```bash
uv run --extra train python v4_interpret.py
uv run --extra train --extra cloud python v4_cleanup.py
uv run --extra train --extra cloud python v4_cost.py
uv run --extra train python v4_evidence.py
uv run --extra train python v4_snapshot.py final-with-reports
```

These commands refuse to overwrite the interpretation, cleanup receipt, compact evidence, or archive. `v4_cost.py` records an observed billing snapshot rather than an invoice; preserve that snapshot before making a later observation.

The initial cleanup verifier incorrectly required every stopped app to appear in Modal's recent-app list. The corrected verifier queries each owned app's lifecycle and tasks directly. `v4_cleanup.py --verify-only` checks an already completed deletion without issuing another stop/delete; this study preserves the initial helper and failure log alongside the successful read-only verification. Regression tests use fake RPC responses and make no GPU call.

Raw records and bounded-call reservations live in `runs/study-v4`; local releases and pinned private-Hub verification downloads live in `artifacts/v4-releases` and `artifacts/v4-hub-verification`. Source snapshots live in `artifacts/v4-source-snapshot`. The working Volume is owned by this study and is deleted only after retention and independent verification. Bulk records remain local and ignored by Git; a GitHub plan alone does not back them up.
