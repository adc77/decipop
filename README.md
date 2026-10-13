# Decision Model Lab

A research project comparing a fine-tuned small language model, direct decision prediction, and a tiny transformer trained from scratch.

**Status: V1–V4 are complete.** Implementation, reports, and the research journal are pushed to the private `feat/data-foundation` branch; see [the publication record](reports/PUBLICATION.md). Eleven selected research model bundles are stored in private Hugging Face repositories. Raw study records, historical source archives, and the entire retained V4 candidate pool also have a verified private Hugging Face backup. Owned temporary Modal storage has been deleted after verification.

Read [RESEARCH_LOG.md](RESEARCH_LOG.md) for the running decision history, completed experiments, failed results, costs, and artifact index. V4's [frozen protocol](docs/V4_PROTOCOL.md) tested a matched training mixture after [controlled diagnostics](reports/V4_DIAGNOSTICS.md) exposed a query-entity generalization gap.

Read [PLAN.md](PLAN.md) for the research questions, data specification, experiments, evaluation protocol, budget, implementation sequence, and credential requirements.

The initial study tests reasoning over supplied facts and rules. It measures generalization, confidence calibration, abstention, and inference cost. It does not establish unrestricted general-purpose decision ability.

The central result is a limitation: both fine-tuned models score above 96% on the original held-out distribution, yet get **0 of 142** fact-flip questions correct when the changed premise removes support for a derived answer. The scratch model stays at 33.33% accuracy. These are research checkpoints, not reliable general decision engines.

Start with [the results](reports/RESULTS.md), [the counterfactual failure analysis](reports/FAILURE_ANALYSIS.md), and [the numerical walkthrough](reports/LEARNING.md). See [costs and verified releases](reports/COSTS_AND_RELEASES.md) for billing and pinned model versions.

Use [docs/SETUP.md](docs/SETUP.md) for reproducibility commands and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model and experiment diagrams. [reports/LOCAL_PROGRESS.md](reports/LOCAL_PROGRESS.md) records the historical V1 scope and verification; the research journal tracks later studies. The earlier pilot measurements remain in [reports/GPU_PILOT.md](reports/GPU_PILOT.md).

V2 completed a matched two-arm data experiment. Both selected heads answer TRUE on every counterfactual test question, with 0% UNKNOWN and complete-family accuracy. The late treatment reached 83.26% development accuracy, but raw-NLL selection retained an early chance-level head. Read [the v2 results](reports/V2_RESULTS.md), [the development walkthrough](reports/V2_LEARNING.md), and [verified releases and costs](reports/V2_RELEASES.md). The [frozen protocol](docs/V2_PROTOCOL.md), v1 artifacts, and all model revisions remain preserved.

V3 compared raw-NLL, calibrated-NLL, and accuracy selection over the two retained checkpoints per V2 arm, using fresh, separate families for temperature fitting, selection, calibration, and tests. Calibrated-NLL and accuracy selection chose the final treatment head: changed-premise accuracy improved from 33.33% to **99.93%**, with **100% UNKNOWN accuracy** and **99.8% complete-family accuracy** on 1,000 fresh parents. Wording accuracy remains 43.4%, single-branch accuracy 51.0%, and original-style UNKNOWN accuracy 0%. Calibration also worsens confidence metrics on the already accurate counterfactual subdistribution. Read [the v3 results](reports/V3_RESULTS.md), [the numerical walkthrough](reports/V3_LEARNING.md), [verified artifacts and cost](reports/V3_RELEASES.md), and [the frozen protocol](docs/V3_PROTOCOL.md).

V4 completed matched additional training from the V3 treatment-final head. Mixed data improves original-query UNKNOWN from **0/1,000 to 815/1,000** while retaining **1,000/1,000** changed-premise UNKNOWN; both declared engineering gates pass. Overall original-query accuracy rises from 66.67% to **93.73%**, but depth drops from 92.33% to **80.00%**, wording remains **42.53%**, and single-branch supported statements score only **59/500**. Read [V4 results](reports/V4_RESULTS.md), [the preselected failure example](reports/V4_LEARNING.md), and [private verified releases, costs, and cleanup](reports/V4_RELEASES.md). All 69 final CPU tests passed; the observed owned cost is **$2.32520053**, with **$0 billed**. The next proposed diagnostic removes the generator's distinguished-target cue; no new GPU study is running.

Read [the next experiment proposal](docs/NEXT_EXPERIMENT.md): test whether the retained models answer about the right entity in worlds with independently supported TRUE, FALSE, and UNKNOWN statements. No new study has been frozen or submitted.
