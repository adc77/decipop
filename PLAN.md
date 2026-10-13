# Decision Model Lab: research and implementation plan

- **Owner:** Aman / adc77
- **Created:** 2026-10-12
- **Status:** v1 study completed; implementation and reports remain local and uncommitted
- **Budget:** $30 starting credits verified by the billing screenshot; $0 net spend limit saved
- **Sequence:** fine-tune first, then train a tiny transformer from scratch
- **Completion:** defined by the deliverables below, without a calendar deadline

## 1. Project objective

Build and study small models that make decisions from supplied facts and rules. Compare how a pretrained language model learns this task with a direct prediction model and a small model initialized from scratch.

The research question is:

> How do pretraining, output architecture, and confidence calibration affect correctness, generalization, and cost on decisions governed by explicit rules?

The initial task has three answers: **TRUE, FALSE, UNKNOWN**. Each example supplies its own rules; the model must apply those rules to the facts rather than remember a fixed policy.

This gives us a controlled way to understand decision models before attempting broader tasks. Generalization here means transferring across rule sets, vocabulary, wording, and reasoning depths. It does not mean the model can make arbitrary business decisions.

This is an independent personal research project. Its data, credentials, infrastructure, and repositories are separate from Sidecar.

## 2. What completion means

The required result is a reproducible study with three trained approaches, a consistent evaluation, and a candid account of their limitations. A model failing to generalize is a research result; a broken pipeline or an unrun experiment is unfinished work.

Required deliverables:

- A versioned generator, symbolic reference solver, and dataset manifest.
- A pretrained model fine-tuned to emit a decision token.
- The same pretrained backbone fine-tuned with a three-class decision head.
- A small transformer trained from random initialization on the decision task.
- Raw predictions, accuracy and uncertainty metrics, robustness checks, and measured inference performance.
- Saved model artifacts and a load-and-predict check for each required model.
- A final report with plots, actual costs, failed experiments, and supported conclusions.
- Hugging Face model repositories containing verified release artifacts after the implementation has been approved and completed.

The study must fit the available credits. If the throughput pilot shows that this scope cannot fit, stop before the expensive runs and document the specific scope or budget change needed. Do not quietly omit the scratch model, spend more money, or call the study complete.

## 3. Scope boundaries

### Included

- Supervised decision training on short English inputs.
- Exact labels computed from formal rules.
- Generative versus direct decision outputs.
- Confidence calibration and selective prediction.
- Tests on unseen rule combinations and deeper reasoning.
- A small transformer implemented and trained from scratch.
- Modal batch Functions for later compute; Hugging Face for later model storage.

### Excluded from v1

- General language-model pretraining at the scale of SmolLM2.
- Claims of a novel architecture or a universal decision engine.
- Reinforcement learning, reward-model training, or paid teacher-model labeling.
- A production API, web application, always-on endpoint, or automated business actions.
- Multilingual training, long-document retrieval, or a large hyperparameter search.
- Model explanations presented as faithful reasoning without verification.

The scratch model learns the bounded decision task directly. It is not a chatbot pretrained on a general web corpus. New architectural ideas can follow after these baselines work.

## 4. Research questions and hypotheses

| Question | Experiment | Interpretation |
| --- | --- | --- |
| Does an explicit decision head help? | Compare two models initialized from the same SmolLM2 checkpoint | Differences concern the output/training setup within this benchmark |
| How much does pretraining help? | Compare pretrained models with the tiny scratch transformer | An informative practical comparison, confounded by size, tokenizer, and architecture |
| Are the models applying rules? | Unseen rule structures, entity renaming, fact changes, deeper chains | Measures specific forms of generalization, not unrestricted reasoning |
| Can confidence be improved? | Raw probabilities versus held-out temperature scaling | Calibration can improve on one distribution and fail on another |
| Does deferral reduce errors? | Error rate as a function of retained prediction coverage | Measures the cost of abstaining, not just headline accuracy |
| Where is the cost advantage? | Controlled latency, throughput, memory, and billing measurements | Report measured tradeoffs rather than assuming the head is faster |

Working hypotheses: pretraining helps with wording variation; direct outputs simplify the result format; deeper chains expose weaknesses; confidence becomes less reliable under distribution shift. These are hypotheses to test, not promised outcomes.

A one-token generative answer may already be fast. Replacing its vocabulary projection with a three-class head does not remove the backbone computation. Any speed advantage must be measured.

## 5. Decision semantics

Use a finite, consistent rule language with positive and explicitly negative facts. Rules are implications with one or two conditions and one conclusion. Variables range over the finite entities supplied in the example. Compute consequences until no new facts can be derived.

For a queried statement `q`:

- **TRUE:** `q` is derivable.
- **FALSE:** the explicit negation of `q` is derivable.
- **UNKNOWN:** neither `q` nor its negation is derivable.

Reject a generated theory if both a fact and its explicit negation can be derived. Absence of a fact is not evidence of its negation. Do not add negation-as-failure, rule priorities, arithmetic comparisons, or exception handling to v1.

Example:

```text
Facts:
- Aman is an administrator.

Rules:
- If someone is an administrator, they are a verified user.
- If someone is a verified user, they can deploy.

Question:
- Can Aman deploy?

Answer: TRUE
```

Dataset rendering will consistently distinguish individual facts from universally quantified rules. Redundant facts are allowed and their frequency will be controlled.

For the same query, an explicit fact that Aman cannot deploy could establish FALSE in a different consistent theory. Removing all paths to either conclusion establishes UNKNOWN.

**UNKNOWN is a semantic label. Abstention is a confidence-based choice to defer prediction.** A model can be highly confident that UNKNOWN is correct, or uncertain between TRUE and FALSE.

## 6. Data strategy

### 6.1 Primary dataset: generated, exact-label problems

Generate formal facts, rules, and questions first. Compute labels with a reference solver, then render them as English. This avoids relying on a language model to invent the correct answer.

Use several controlled vocabularies, such as access permissions, software capabilities, and neutral attributes. The vocabularies change the wording, not the logical semantics. No domain is tied to a particular label.

Generation controls:

- Two to eight entities per theory.
- Four to sixteen initial facts and zero to eight rules; depth-zero cases can use facts alone.
- Unary properties and binary relations.
- One- and two-condition rules, with explicit negative facts/conclusions.
- Short inference chains, irrelevant facts, redundant facts, and alternative valid paths.
- Randomized statement ordering and answer-balanced sampling.
- Multiple English rendering templates, with reserved templates for later tests.

Verify labels before balancing. Record rejection counts and label frequencies before and after sampling. Artificially balanced results do not establish calibration under an unknown real-world class distribution.

### 6.2 Dataset size and splits

Initial targets, adjustable once at the throughput gate before training and final evaluation:

| Split | Target questions | Purpose |
| --- | ---: | --- |
| Train | 20,000 | Optimize model weights; reasoning depth 0–3 |
| Development | 2,000 | Select checkpoint and permitted training adjustment |
| Calibration | 2,000 | Fit temperature and select deferral thresholds |
| In-distribution test | 3,000 | Final evaluation on new theories, depth 0–3 |
| OOD depth test | 1,500 | Reasoning depth 4–5 |
| OOD wording test | 1,500 | Reserved English templates and entity vocabulary |
| OOD structure test | 1,500 | Reserved rule compositions at trained depths |

Targets count questions, not independent theories. Report both counts. Use at most three queries per theory so one large theory does not dominate the results.

For TRUE and FALSE, proof depth is the shortest derivation of the queried statement or its explicit negation; an initial fact has depth zero. UNKNOWN has no proof depth and stores `null`. Training theories have no derived fact requiring more than three steps. In the depth test, answerable questions require four or five steps; UNKNOWN questions come from those deeper theories and are reported separately, not assigned fictitious proofs.

Split at the theory-family level before rendering. All queries, paraphrases, reorderings, and counterfactual variants of the same underlying theory stay together. Canonicalize variable/entity names and statement order to detect equivalent theories across splits.

For the structure test, reserve identified graph motifs before generating training theories; verify their absence in training. Describe what is actually held out, rather than calling all new rule sets structurally novel.

Create immutable split manifests with hashes, generator revision, rendering revision, random seeds, label definitions, counts, and depth distributions. Tokens are never truncated silently: reject and regenerate an overlong primary example, recording the rejection rate.

### 6.3 Existing research datasets

[RuleTaker](https://github.com/allenai/ruletaker) provides generation and theorem-proving tools, plus a dataset download. [ProofWriter](https://arxiv.org/abs/2012.13048) is relevant prior work on implications and proofs over natural-language rule bases.

Use these to study established evaluation methods and, if budget permits, run a small external evaluation. They are not prerequisites for the primary dataset. Verify the exact dataset license, version, label semantics, and theory grouping before importing or redistributing an external subset. A code repository's license does not automatically establish a separate dataset's license.

Do not combine closed-world labels with the project's open-world UNKNOWN semantics. Public benchmark performance may also reflect pretrained exposure; fresh generated held-out theories are the primary evidence.

### 6.4 Dataset validation

Before GPU training:

- Validate the schema, balanced sampling, consistency, and derivation depths.
- Check exact and canonicalized overlap across every split.
- Independently check solver outcomes on an exhaustive small problem set.
- Review 100 rendered examples across all three labels for English/formal agreement.
- Measure token lengths for both tokenizers and enforce the shared input limit.
- Verify transformed examples preserve or change the label as intended.
- Run a lexical baseline to detect suspicious shortcuts.

Use a simple unigram TF-IDF plus logistic regression baseline on the rendered inputs. If it performs unexpectedly well, inspect and correct the generator before training. The symbolic solver is the label oracle on formal inputs; its access to structured facts/rules is different from the neural models' access to English.

## 7. Shared example and prediction formats

An example stores:

```json
{
  "example_id": "theory-000001-query-1",
  "family_id": "family-000001",
  "split": "train",
  "facts": ["Aman is an administrator."],
  "rules": ["If someone is an administrator, they can deploy."],
  "question": "Aman can deploy.",
  "label": "TRUE",
  "proof_depth": 1,
  "renderer_id": "canonical-v1",
  "theory_hash": "<canonical-content-hash>"
}
```

The formal theory and any verified derivation are kept as metadata. The neural model sees only the rendered facts, rules, question, task instruction, and answer mapping. It never receives the label, proof depth, oracle derivation, split name, or family ID as input.

Each prediction records the example ID, run ID, checkpoint revision, predicted class, three class probabilities, raw versus calibrated status, abstention flag, input length, and latency metadata. Invalid generative outputs remain visible rather than being silently repaired.

## 8. Models and experimental controls

### 8.1 Model A: pretrained LLM with decision-token fine-tuning

Start from [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), an Apache 2.0 pretrained decoder model. Pin the exact checkpoint revision before the first run. Its existing pretraining is what makes this affordable; we are not recreating it.

Render a stable answer mapping: `A = TRUE`, `B = FALSE`, `C = UNKNOWN`. Verify that the three answer continuations are each one token under the pinned tokenizer and exact prompt suffix. If this fails, select and document a different equal-length mapping before data freeze; do not change it mid-study.

Train with full-vocabulary cross-entropy on the answer token only. Prompt positions and padding are masked from the loss. Start with full fine-tuning, not quantization or LoRA, to keep the first comparison clear. Use bfloat16 where supported and gradient checkpointing if needed.

Evaluate both:

- Unrestricted next-token argmax, including invalid answers.
- Restricted three-choice prediction with probabilities normalized over the allowed answer tokens.

Record probability mass assigned outside the allowed answers. Restricted-choice probabilities are conditional on that answer set; they are not unrestricted vocabulary confidence.

### 8.2 Model B: same pretrained backbone with a decision head

Initialize from the exact same base checkpoint as A. Take the last non-padding prompt position's final hidden state after the answer marker. Apply a learned linear projection to three class logits, then softmax. All preceding facts and rules are visible through the decoder's causal attention.

Train the backbone and head with three-class cross-entropy. Model A and B use the same input content, tokenization, training examples, exposure cap, and checkpoint selection rule.

This isolates the output setup more closely than comparing unrelated pretrained models. The objectives still differ: A competes against its full vocabulary, B against three classes. Report this difference, trainable parameter counts, runtime, and actual optimization steps.

### 8.3 Model C: tiny transformer from scratch

Implement a standard encoder transformer in PyTorch to understand the architecture directly. Proposed starting configuration:

| Component | Initial choice |
| --- | --- |
| Layers | 6 |
| Hidden dimension | 384 |
| Attention heads | 6 |
| Feed-forward dimension | 1,536 |
| Vocabulary | 8,192-token tokenizer fitted on training text only |
| Maximum input length | 512 tokens |
| Positional representation | Learned position embeddings |
| Normalization | Pre-layer normalization |
| Dropout | 0.1 |
| Output | Pooled decision token plus three-class linear head |
| Size | Approximately 14M parameters; calculate exact count in implementation |

Build token embeddings, masked multi-head attention, residual connections, feed-forward blocks, normalization, and the prediction head. Use PyTorch's attention primitives and optimizers rather than writing custom GPU kernels.

All weights start randomly. The tokenizer learns only from training inputs. The encoder can attend bidirectionally. Train directly on decision labels; no broad language pretraining or hidden external supervision.

This is a learning baseline using established components, not an architectural novelty claim. Its different tokenizer, size, attention pattern, and initialization make C versus A/B a practical comparison rather than a clean causal estimate of pretraining alone.

### 8.4 Common training protocol

- All models use the same split membership and rendered input content.
- Maximum length is 512 tokens; include instruction and answer marker in the check.
- Use AdamW, gradient clipping at 1.0, 5% warmup, and linear learning-rate decay.
- Initial backbone learning rate: `2e-5`; new decision head: `1e-4`; scratch model: `3e-4`.
- Weight decay: 0.01; effective batch target: 32 examples.
- Start pretrained microbatch at 4 and scratch at 16, with explicit accumulation.
- Primary exposure cap: three training passes for each model.
- Development evaluation and checkpointing every 250 optimizer steps and at completion.
- Select the checkpoint by development three-class negative log-likelihood; use accuracy to break an exact tie.
- For A, checkpoint selection uses its restricted three-choice probabilities, while unrestricted invalid-answer rates remain a separate reported metric.
- Preserve a final checkpoint and the selected checkpoint; retain at most two recovery checkpoints per active run.

These are starting configurations, not claims of optimal settings. The throughput/fit pilot may set a smaller common dataset size or exposure cap before confirmatory training. Freeze the resulting protocol and record its rationale.

Use primary seed 17. A second matched seed, 29, is conditional on the budget forecast covering all required remaining work. Do not run extra seeds only for the best-looking model and present that as a balanced comparison.

Training exposure is the main control; compute will differ. Report examples seen, input tokens processed, epochs completed, optimizer steps, GPU time, trainable parameters, and peak memory. Do not imply equal data exposure means equal compute.

## 9. Calibration and abstention

Compute raw class probabilities for all models. Fit one positive scalar temperature per selected model on the calibration split by minimizing negative log-likelihood:

```text
calibrated_probabilities = softmax(class_logits / temperature)
```

Freeze that temperature before test evaluation. [Guo et al.](https://arxiv.org/abs/1706.04599) provide the starting methodology. Calibration on one distribution does not guarantee calibration on deeper or differently worded problems.

Deferral rule: abstain if the maximum calibrated class probability is below a threshold. On the calibration split, select thresholds targeting 50%, 75%, and 90% retained coverage, handling ties explicitly. Apply these unchanged to every test split and report the resulting coverage and error rates.

Also report threshold-free risk–coverage curves on final test predictions. These describe tradeoffs; they are not a mechanism for tuning a production threshold on test data.

The model predicts a class even when its answer is deferred; preserve that underlying result. Deferral is not a fourth training class and is never merged with UNKNOWN.

## 10. Evaluation protocol

### 10.1 Correctness

Report accuracy, macro F1, class confusion matrices, and results by label, proof depth, input length, vocabulary, and test split. Compare with majority-class, lexical, and unfine-tuned pretrained baselines.

UNKNOWN examples receive their own results rather than appearing in proof-depth averages. Report depth-test answerable accuracy alongside the overall three-class result.

For Model A, unrestricted invalid outputs count as errors. Present restricted-choice accuracy separately so format constraints do not hide failures.

### 10.2 Uncertainty

Report:

- Negative log-likelihood.
- Multiclass Brier score using the sum over three classes, explicitly stating the convention.
- Expected calibration error with 10 equal-width bins and bin counts.
- Reliability diagrams before and after calibration.
- Error rate versus coverage and area under the risk–coverage curve, stating the integration convention.
- High-confidence mistakes and calibration drift on each OOD split.

Accuracy alone cannot establish decision reliability. ECE depends on binning and sample size, so interpret it alongside the proper scoring metrics and diagrams.

### 10.3 Robustness and causal changes

Generate test-only variants from held-out theory families:

- Rename entities without changing logical relationships.
- Reorder facts and rules.
- Add irrelevant facts.
- Use independently reviewed paraphrase templates.
- Remove a necessary premise and recompute the label with the oracle.
- Change a relevant explicit fact and recompute the label.

Use 100 eligible families per transformation, recording exclusions. Do not assume every fact deletion changes an answer: alternative proof paths can preserve it. Report both prediction consistency for label-preserving changes and correctness for oracle-confirmed label changes.

### 10.4 Performance

Benchmark on the same Modal GPU, precision, token lengths, and software image. Separate model loading from steady-state inference.

- Batch size 1 latency: median and p95 over 500 requests after warmup.
- Batch size 32 throughput on a fixed-length bucketed set.
- GPU synchronization around timing, with tokenization reported separately.
- Peak GPU memory and actual parameter count.
- Cold-start/model-loading time measured separately.
- Measured cost per 1,000 decisions at the benchmarked batch settings.

Use each model's normal inference path and report differences in token counts. For A, benchmark one-token prediction; do not manufacture a long explanation baseline to inflate the decision head's advantage.

### 10.5 Statistical discipline

Use paired bootstrap intervals for model differences, resampling theory families rather than individual related questions. Predeclare the primary comparisons: A versus B on in-distribution accuracy and NLL, and all models' accuracy loss at deeper reasoning depths.

Freeze model checkpoints, calibration, thresholds, and test manifests before final scoring. Failed sanity checks can be fixed, but an inspected test set cannot become an untouched test again. A later training revision needs a freshly generated reserved test set and an explicit revised protocol.

One seed does not establish training stability. With one seed, describe intervals as test-family sampling uncertainty, not uncertainty across training runs. Secondary analyses remain descriptive.

## 11. Modal compute and budget

### 11.1 Initial worker

Use one L4 GPU with 4 physical CPU cores and 32 GiB host RAM for batch training/evaluation Functions. Host RAM is separate from GPU VRAM. Use normal scheduling without a fixed-region premium or non-preemptible upgrade.

Current listed base rates, checked 2026-10-12:

| Resource | Rate | Hourly subtotal |
| --- | ---: | ---: |
| L4 GPU | $0.000222/second | $0.799200 |
| 4 physical CPU cores | $0.0000131/core/second | $0.188640 |
| 32 GiB RAM | $0.00000222/GiB/second | $0.255744 |
| Combined | | **$1.243584/hour** |

This estimate assumes measured CPU/RAM stay within requests; billing uses the greater of requested and actual resource usage. Container loading and idle teardown time also count. Storage, transfers, builds, and other workspace use must be reconciled separately. Recheck rates before execution. [Modal pricing](https://modal.com/pricing)

### 11.2 Allocation, not a runtime promise

| Work | Compute allocation |
| --- | ---: |
| CPU data work, image preparation, GPU smoke and throughput pilots | $2 |
| Model A fine-tuning | $5 |
| Model B fine-tuning | $5 |
| Model C from-scratch training | $3 |
| Final evaluation, calibration, benchmarks, artifact round-trip | $3 |
| Controlled debugging allowance | $2 |
| Planned subtotal | **$20** |
| Unscheduled reserve | **$10** |

At the illustrative worker rate, $20 buys about 16 worker-hours before ancillary charges. Measure real tokens/second, peak memory, evaluation overhead, and remaining work before committing to full runs. An allocation is not evidence that training will finish within it.

Hold the reserve through the scratch-model gate. Use it only for a documented necessary run whose forecast stays within the verified project budget. It is not automatic permission for optional ablations.

### 11.3 Cost controls

- Authenticate to the intended personal workspace and verify applicable remaining credits, cycle end date, and current usage.
- Set a net spend limit of $0. Workspace usage budgets apply before credits; inspect existing usage before setting the correct cumulative limit. For an unused cycle with $30 applicable credits, a $30 total-usage limit is the outer cap. [Modal budgets](https://modal.com/docs/guide/budgets)
- Keep a project ledger with estimated and dashboard-observed cost. The ledger is not a billing API substitute.
- Permit one GPU job at a time; no persistent deployment, schedule, or warm container fleet.
- Bound jobs with explicit timeouts and optimizer-step caps derived from their allocation.
- Use short resumable jobs with checkpoints, rather than a single unattended long run.
- Disable automatic whole-job retries; review a failed run's checkpoint and cost before resuming.
- At every gate, reserve the forecast cost of all remaining required deliverables.
- Stop if the remaining forecast exceeds credits. Never increase paid spending without a new user instruction.

Reserve and usage must be tracked across billing cycles; do not assume unused monthly credits roll over or reset the project ledger when the month changes.

### 11.4 Memory or throughput failure

If the first full-fine-tuning pilot fails, explicitly investigate batch size, activation memory, and checkpointing. A smaller microbatch may preserve the effective batch through accumulation. Do not silently switch GPU, quantize, change model size, or change to LoRA.

If a smaller base model or adapter training is necessary, amend the protocol before full runs and apply the change to both A and B. Preserve the meaning of the comparison. If all three required models no longer fit, report that scope/budget constraint before proceeding.

## 12. Storage, reproducibility, and Hugging Face

### 12.1 Responsibilities

| Location | Contents |
| --- | --- |
| Local checkout | Source, configs, documentation, and downloaded reviewed results |
| GitHub | Versioned code, manifests, compact metrics, and final report |
| Modal Volume, later | Working datasets, model cache, recovery state, and run outputs |
| Hugging Face, later | Versioned final model artifacts, tokenizer, calibration, and model cards |

Weights and optimizer checkpoints never go into the GitHub repository. A [Modal Volume](https://modal.com/docs/guide/volumes) is a working checkpoint store, not the final model registry. Explicitly commit completed checkpoints and download important logs; do not depend on the dashboard's log retention.

Checkpoint state includes model, optimizer, scheduler, random number state, data-order state, step count, configuration, and data hashes. Resume in a new run directory linked to its parent. Write checkpoints atomically and mark completion so interrupted files are not mistaken for valid recovery points.

### 12.2 Model release bundle

Each required model has a separate private Hugging Face model repository in `axondendriteplus`. Pinned verified revisions are listed in `reports/COSTS_AND_RELEASES.md`:

- `axondendriteplus/decision-lab-smollm2-360m-token`
- `axondendriteplus/decision-lab-smollm2-360m-head`
- `axondendriteplus/decision-lab-transformer-14m`

Release bundle:

- `model.safetensors` and complete architecture configuration.
- Tokenizer files, label mapping, and exact prompt template.
- Calibration temperature and documented abstention thresholds.
- Training config, source commit, dataset manifest/hash, and checkpoint-selection method.
- Per-split evaluation summary and a model card with limitations and license provenance.
- Loading instructions and a small set of fixed verification inputs.

Model B requires its custom head to be restored along with the backbone. Model C requires its own tokenizer and architecture implementation. Record these dependencies; do not claim that either loads as a stock causal LM.

Use [Hugging Face Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload) with an explicit release allowlist. After upload, load the pinned remote revision in a fresh CPU or GPU worker and verify predictions against the local export. Preserve the returned Hub commit IDs in the report. No hosted Hugging Face inference service is needed.

Publish project-authored code under Apache 2.0 at implementation, retain applicable pretrained-model attribution, and separately record dataset rights. Public visibility changes require the user's publishing instruction; private artifact storage is sufficient to finish v1.

## 13. Credentials and prerequisites

No Modal or Hugging Face credentials are needed to review this plan or publish this documentation to GitHub. GitHub authentication is already available as adc77.

Before implementation execution, the user needs to provide access through local authentication or protected environment configuration:

| Requirement | Needed for | Setup and verification |
| --- | --- | --- |
| Modal account and correct personal workspace | GPU/CPU jobs and working Volume | Browser-based `modal setup` or a workspace API token; verify account/workspace |
| Modal token ID and token secret | Noninteractive client execution | `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or the active local profile |
| Applicable Modal credits and billing settings | Budget feasibility | Verify balance, cycle dates, usage limit, and $0 net spend limit |
| Hugging Face username/namespace | Artifact ownership | Verify the namespace using authenticated account metadata |
| Hugging Face write access | Private model-repository creation and uploads | Fine-grained token with the necessary destination permissions |
| Python 3.12 and locked dependencies | Reproducible control and worker environments | Pin compatible versions after implementation smoke checks |

Use [Modal's setup flow](https://modal.com/docs/guide/getting-started) or [token commands](https://modal.com/docs/cli/latest/token). These are API credentials for running jobs, not endpoint proxy credentials. Browser setup avoids placing token values in shell command arguments.

For Hugging Face, use `hf auth login` locally or a protected `HF_TOKEN`. [Hugging Face token documentation](https://huggingface.co/docs/hub/security-tokens) describes read, write, and fine-grained permissions. Public base-model downloads do not by themselves require upload credentials. Validate write access against the intended private destination before the release stage.

Keep the Hugging Face write token out of training workers; use it only for the explicit publishing operation. Local upload of downloaded exports is the default. If a remote publishing worker is later needed, inject its credential through a separate scoped [Modal Secret](https://modal.com/docs/guide/secrets).

Never put credential values in source, GitHub, manifests, model cards, logs, or notebooks. An example configuration contains empty placeholders only. This project does not use Azure Key Vault or company accounts.

### Current readiness

- The plan is on GitHub; implementation is local on `feat/data-foundation` and has not been committed or pushed.
- The public pretrained checkpoint and all dependency versions are pinned.
- Data generation, all three models, training, evaluation, and private model releases are implemented and executed.
- Modal authentication is verified for `amandogra2016`; Hugging Face write access and private releases are verified for `axondendriteplus`.
- The billing screenshot verified $30 starting credits and a saved $0 net spend limit. The final observed accounting is in `reports/COSTS_AND_RELEASES.md`.
- The project uses Python 3.12 and `uv.lock`; the user explicitly authorized CPU-only tests on this laptop.

Credentials remain in local SDK authentication stores. No write token was sent to a GPU worker or included in a release bundle.

## 14. Proposed repository structure

The original plan commit contains only `README.md`, `PLAN.md`, and `.gitignore`. Implementation and results are local. The tree below is the original proposed layout; `docs/SETUP.md` records the implemented commands and `reports/LOCAL_PROGRESS.md` records the retained artifacts.

```text
decision-model-lab/
  README.md
  PLAN.md
  LICENSE
  pyproject.toml
  uv.lock
  .env.example
  configs/
    dataset.yaml
    model_a.yaml
    model_b.yaml
    model_c.yaml
    evaluation.yaml
    budget.yaml
  src/decision_lab/
    config.py
    cli.py
    data/
      schema.py
      generate.py
      solve.py
      render.py
      split.py
      validate.py
    models/
      token_model.py
      decision_head.py
      scratch_transformer.py
    training/
      train.py
      checkpoint.py
    evaluation/
      predict.py
      metrics.py
      calibrate.py
      robustness.py
      benchmark.py
      report.py
    artifacts/
      export.py
      publish.py
    budget.py
  modal_app.py
  tests/
    test_solver.py
    test_dataset.py
    test_model_contracts.py
    test_metrics.py
    test_checkpoint_resume.py
    test_artifact_roundtrip.py
  manifests/
  reports/
    FINAL_REPORT.md
    metrics/
    figures/
```

Use one configuration contract for tunable values. Validate required settings at startup, fail clearly on missing credentials or incompatible artifacts, and keep comments/docstrings to one line. No silent model, device, dataset, or credential fallbacks.

## 15. Implementation sequence and gates

### Stage 0: plan review

Deliver this local documentation and the GitHub main branch. Do not install training dependencies, create Modal resources, download weights, start runs, or create Hugging Face repositories during this stage.

**Exit:** user has reviewed the plan and instructed implementation to start.

### Stage 1: source and data foundation

Implement configuration, CLI contracts, the formal language, solver, generator, renderer, dataset splits, and validation. Add meaningful tests for explicit negation, UNKNOWN, multi-condition rules, alternate proofs, and contradictions.

Implement the lexical baseline and dataset manifest. Prepare pinned environments and the Modal batch wrapper in source without launching remote work.

**Exit:** reviewed examples and deterministic data tests pass, splits are clean, and inputs satisfy both tokenizers' limits. The required compute access is verified before any execution that needs it.

### Stage 2: GPU smoke and throughput gate

Use a short bounded L4 run after account authentication and budget verification. For A and B, check finite loss, correct answer positions, padding masks, and gradients. Overfit a tiny fixed set of 32 examples as a wiring test, not a generalization result.

Measure sustained throughput after warmup on representative lengths, checkpoint time, and development-evaluation time. Forecast all required training, scratch work, final evaluation, and artifact verification. If needed, reduce the shared training count or common exposure cap before data/protocol freeze.

**Exit:** viable memory/throughput, bounded cost, and a frozen protocol with enough remaining budget for all required stages.

### Stage 3: Model A fine-tuning

Run the unfine-tuned baseline and primary decision-token fine-tuning. Inspect training/development curves, unrestricted invalid answers, and loss masking. Select and export the development-selected checkpoint.

**Exit:** trained A checkpoint reloads and produces valid recorded predictions. No final test tuning.

### Stage 4: Model B fine-tuning

Initialize from the same base revision, not from trained A. Train the head and backbone under the shared exposure protocol. Verify probability normalization, gradients, selected checkpoint, and reload behavior.

**Exit:** trained B checkpoint is comparable to A under the declared protocol. Record any required differences explicitly.

### Stage 5: Model C architecture and training from scratch

Implement the tiny encoder and train-only tokenizer. Compare custom attention/masking behavior with a trusted PyTorch reference on fixed tensors. Verify gradients and tiny-set fit. Train the primary scratch model on the frozen task data.

**Exit:** a trained C checkpoint reloads and can be scored. If it learns poorly, document the result; do not relabel an untrained artifact as a finished baseline.

### Stage 6: calibration and final evaluation

Freeze checkpoints and fit calibration on the calibration split. Run the untouched tests, robustness transformations, performance measurements, and theory-family bootstrap analysis. Produce plots and inspect representative errors.

**Exit:** all required metrics and raw prediction files exist, and no final test information was used to choose weights or thresholds.

### Stage 7: Hugging Face artifacts and final report

Export each model, validate a clean-environment reload, upload private model bundles to the chosen Hugging Face namespace, and verify a download from the pinned Hub revisions. Write the report with actual usage, scope changes, caveats, and links.

**Exit:** all required completion checks below pass. Public releases, production deployment, and broader research are separate future work.

Work progresses by these gates, not by elapsed weekends. The user may review intermediate findings, but routine implementation choices within an approved protocol do not require repeated permission.

### Learning checkpoints

At each implementation stage, retain a short worked explanation with the actual tensor shapes and one verified example:

| Concept | Exercise |
| --- | --- |
| Attention and pooling | Trace which input positions each model can attend to and how its decision representation is selected |
| Fine-tuning | Inspect which parameters receive gradients and compare a prediction before and after training |
| Decision outputs | Compute the three logits, softmax probabilities, and cross-entropy for one example |
| Calibration | Compare confidence with empirical correctness before and after temperature scaling |
| Generalization | Trace a shallow correct example and a deeper error using the symbolic oracle |
| From-scratch training | Explain what random initialization and train-only tokenization do and do not provide |

The learned head's probabilities are statistical estimates. Changing the output format does not make decisions correct, and supervised cross-entropy is already a valid starting objective for learning class probabilities. Reinforcement learning is not required to understand these fundamentals.

## 16. Tests and verification

Tests must check behavior that could invalidate the study:

- Solver correctness on exhaustive small theories and hand-checked examples.
- Open-world UNKNOWN versus explicit negation.
- Canonicalization and family-level split isolation.
- Renderer/formal agreement and counterfactual label changes.
- Token positions, padding masks, answer mapping, and loss masks.
- Finite gradients and attention behavior in the scratch transformer.
- Metric formulas against simple known cases.
- Calibration fitted only on its allowed split.
- Recovery from a saved checkpoint with preserved data ordering and run provenance.
- Export/reload and Hub download prediction equivalence within documented tolerances.
- Budget preflight rejecting a run whose forecast consumes required remaining funds.

During this plan-only stage, validate documentation links, file contents, Git identity, branch, and remote contents. Do not claim model tests have run.

For later execution, the local machine remains the source/control workspace. Tests and training run in the approved isolated execution environment, with GPU work on Modal after review. There is no need to provision a Sidecar service stack. Select the non-GPU execution location explicitly at implementation start; do not silently use shared company infrastructure.

## 17. Failure analysis and decision rules

| Finding | Response |
| --- | --- |
| Lexical baseline exploits label shortcuts | Fix generator/rendering, regenerate manifests before neural training |
| Tiny-set fit fails | Fix the loss, masking, optimizer, or architecture before full runs |
| High in-distribution accuracy, poor deeper-chain accuracy | Report the limitation and inspect dependency depth; do not claim generalized reasoning |
| Good accuracy, overconfident errors | Compare calibration and deferral, including OOD failure |
| Decision head barely improves latency | Report backbone-dominated cost honestly |
| Scratch model underperforms | Report size/tokenizer/pretraining differences; do not infer a single cause |
| External data semantics/license are unclear | Exclude that optional dataset with an explicit reason |
| GPU unavailable | Stop and document availability; change GPU only through an explicit revised cost forecast |
| Budget forecast fails | Stop before the run; propose a concrete scope/budget revision |
| Upload/reload differs from the local artifact | Fix serialization or loading and rerun verification before calling it released |

Optional ideas after the required study: a parameter-matched initialization control, shared-tokenizer comparisons, variable candidate scoring, multi-task training, recurrence, or training with probability targets. None are prerequisites for completing v1.

## 18. Final report contents

The report must answer:

1. What task, data distributions, and held-out changes were actually tested?
2. How did the pretrained token model and direct head compare?
3. What did the scratch transformer learn, and where did it fail?
4. How reliable were raw and calibrated confidence estimates?
5. What did abstention cost in retained coverage?
6. Which perturbations changed predictions incorrectly?
7. What were actual training/inference costs and resource usage?
8. Which conclusions are supported, and which experiments remain unrun?

Include accuracy-by-depth plots, confusion matrices, reliability diagrams, risk–coverage curves, learning curves, and latency/throughput comparisons. Record seeds, revisions, dataset hashes, exact run commands, actual exposure, and the cost ledger. Link each required model to its verified Hugging Face revision.

## 19. Completion checklist

- [x] Plan reviewed; implementation authorized.
- [x] Personal Modal workspace, credits, budgets, and Hugging Face namespace verified.
- [x] Dataset semantics, solver, rendering, and split-isolation checks pass.
- [x] Training protocol frozen after the cost/throughput pilot.
- [x] Model A trained, selected on development data, exported, and reloaded.
- [x] Model B trained under the shared protocol, exported, and reloaded.
- [x] Model C implemented, trained from random initialization, exported, and reloaded.
- [x] Calibration and deferral thresholds fitted independently of tests.
- [x] Required test, robustness, uncertainty, and efficiency results recorded.
- [x] Actual project cost reconciled within verified credits.
- [x] Three private Hugging Face model bundles uploaded and round-trip checked.
- [x] Final report contains limitations, failed runs, and reproducibility details.
- [x] Required artifacts are retained and no research job is left running.

Completion is about a valid, reproducible study. No accuracy threshold, positive hypothesis, publication acceptance, or commercial usefulness is promised.

## 20. Reading sequence and source notes

Read these alongside implementation, in this order:

1. [Transformers as Soft Reasoners over Language](https://arxiv.org/abs/2002.05867): established rule-reasoning formulation and depth generalization.
2. [RuleTaker repository](https://github.com/allenai/ruletaker): generation and reference reasoning tools.
3. [ProofWriter paper](https://arxiv.org/abs/2012.13048): implications, proofs, and the difference between predicting an answer and supporting it.
4. [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) and [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json): backbone, tokenizer, and pretrained limitations.
5. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599): probabilities, temperature scaling, and calibration evaluation.
6. [Modal GPU guide](https://modal.com/docs/guide/gpu), [Volumes](https://modal.com/docs/guide/volumes), [pricing](https://modal.com/pricing), and [budgets](https://modal.com/docs/guide/budgets): worker configuration, recovery, and spend control.
7. [Hugging Face tokens](https://huggingface.co/docs/hub/security-tokens) and [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload): artifact access and versioned storage.

The experiment design, allocations, hyperparameters, dataset targets, and gates in this plan are project choices. Source papers establish prior methods; they do not guarantee this project's results. Recheck provider documentation and package compatibility when execution starts.
