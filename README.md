# decipop

Can a small model change its answer when the evidence changes?

**[Read the research article](https://adc77.github.io/decipop/)** · [Models](docs/MODELS.md) · [Dataset](https://huggingface.co/datasets/axondendriteplus/decipop-worlds)

decipop studies three-way decisions over explicit facts and rules: TRUE, FALSE or
UNKNOWN. It includes a symbolic oracle, synthetic data, controlled training
comparisons and eight selected SmolLM2-360M research checkpoints.

## What we found

| Test | Run 1 | Run 2 |
| --- | ---: | ---: |
| Multi-entity curriculum: every person correct | **245/360**, control 25/360 | **245/360**, control 19/360 |
| Added consistency loss: every person correct before and after an edit | **208/378**, control 205/378 | **235/378**, control 221/378 |
| Consistency model: every answer correct across all nine variants | **50/378** | **46/378** |

The curriculum gain replicated. The added-loss advantage failed its declared
replication check: the first interval includes zero. Even the later models remain
fragile under evidence edits. These are distinct tests on fresh data, not one
leaderboard. [V7 results](reports/V7_RESULTS.md) · [V8 results](reports/V8_RESULTS.md).

## Run a recorded case

```sh
git clone https://github.com/adc77/decipop.git
cd decipop
uv sync --extra train --extra analysis --extra baseline
uv run python examples/predict.py
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. The command downloads
a custom research checkpoint and checks its file hashes before running two saved
prompts on CPU. Use `--revision <commit>` from the [model index](docs/MODELS.md) to
pin the download.

In this [actual held-out case](examples/recorded-support-removal.json), Farah's
trusted and verified facts imply authorization. Removing verification changes the
correct answer from TRUE to UNKNOWN. The model still answers TRUE. The illustration
was chosen after evaluation; it is not an additional confirmatory test.

## Use the data

```python
from datasets import load_dataset
rows = load_dataset("axondendriteplus/decipop-worlds", "v8-r2", split="test_paired")
```

Install `datasets` separately for this optional example. The four configs contain
190,688 serialized rows, including repeats and related edits. They are not
independent worlds. One entity name is consistently normalized to `EntityA`.
Labels and proof depths are checked with two oracles. **Reported model scores use
the original inputs; the normalized release has not been re-evaluated.** See the
[dataset card](https://huggingface.co/datasets/axondendriteplus/decipop-worlds).

## Repository map

| Path | Contents |
| --- | --- |
| [src/decision_lab/data](src/decision_lab/data) | Schemas, oracle, generation and partitioning |
| [src/decision_lab/models](src/decision_lab/models) | Decision heads, token model and scratch baseline |
| [src/decision_lab/training](src/decision_lab/training) | Training, checkpointing and call ledger |
| [src/decision_lab/evaluation](src/decision_lab/evaluation) | Calibration, metrics and exports |
| [src/decision_lab/studies](src/decision_lab/studies) | V1–V8 study modules and orchestration |
| [src/decision_lab/publishing](src/decision_lab/publishing) | Dataset normalization and release utilities |
| [configs](configs), [tests](tests), [examples](examples) | Settings, CPU tests and inference example |
| [site](site), [tools](tools) | Public article and its data builder |
| [docs](docs), [reports](reports), [release](release) | Methods, results and pinned publication records |

The Python import name `decision_lab` is retained for compatibility with model
bundles. [Development and reproduction](docs/REPRODUCING.md) explains the new module
commands and how to recover the original experiment source from `v0.1.0`.
[RESEARCH_LOG.md](RESEARCH_LOG.md) records decisions, failures and corrections.

## Scope and license

One pretrained family and narrow synthetic distributions. No claim of unrestricted
reasoning, architecture novelty or validated business decisions. UNKNOWN means
neither a statement nor its explicit negation follows; it is not a refusal.

Project code and generated data: [Apache 2.0](LICENSE); retain [NOTICE](NOTICE).
Pretrained-model and dependency attribution remain separate. Related work includes
RuleTaker, ProofWriter, temperature scaling and consistency regularization.
