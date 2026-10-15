# Premise

Can a small model follow changing evidence?

Premise is a reproducible study of three-way decisions over explicit facts and rules. It includes an exact symbolic oracle, synthetic data generators, matched training comparisons, strict intervention tests, and research checkpoints based on SmolLM2-360M.

[Dataset](https://huggingface.co/datasets/axondendriteplus/premise-worlds) · [Models](docs/MODELS.md) · [Research article](research-article/README.md) · [Results](reports/V8_RESULTS.md)

## What the experiments found

| Comparison | Run 1 | Run 2 |
| --- | --- | --- |
| Multi-entity curriculum vs. old curriculum, complete base worlds | **245/360 vs. 25/360** | **245/360 vs. 19/360** |
| Added consistency vs. paired cross-entropy, complete paired worlds | **208/378 vs. 205/378** | **235/378 vs. 221/378** |
| Latest consistency models, complete families across all nine variants | **50/378** | **46/378** |

The curriculum gain replicated. The added-loss advantage failed its declared replication check: its first 95% world-sampling interval includes zero. Transformation reliability remains weak. These are different endpoints on fresh datasets, not a single leaderboard. See [V7](reports/V7_RESULTS.md) and [V8](reports/V8_RESULTS.md) for counts, intervals, selection, deviations, and limitations.

A correct individual answer is a lower bar than getting every person right. A complete paired world additionally requires correctness after an assigned edit. A complete transformation family requires every included query and variant correct. Intervals condition on fixed trained models; they do not estimate a population of training seeds.

## Try an actual recorded case

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/adc77/premise.git
cd premise
uv sync --extra train --extra analysis --extra baseline
uv run python examples/predict.py --model axondendriteplus/premise-v8-r2-consistency
```

The example uses the exact held-out prompts in [examples/prompts.json](examples/prompts.json). Farah is trusted and verified; those facts imply active, which implies authorized. Removing her verification prevents that derivation. The correct answer becomes UNKNOWN. The recorded consistency model still answers TRUE. [Full case and original saved outputs](examples/recorded-support-removal.json).

This illustration was chosen after evaluation; the original preselected cases remain in the [V8 report](reports/V8_RESULTS.md). Names are symbols in generated data, not real people or operational access decisions. CPU FP32 inference can differ slightly from the original BF16 aggregate inference; fixed reference checks have recorded tolerances.

The weights require the included custom decision-head loader. They are not stock chat checkpoints. Pass `--revision <commit>` to pin a model. The example verifies the downloaded artifact's SHA-256 inventory before inference.

## Dataset

```python
from datasets import load_dataset

rows = load_dataset("axondendriteplus/premise-worlds", "v8-r2", split="test_paired")
print(rows[0]["prompt"], rows[0]["label"])
```

Install `datasets` separately to use this optional Hub example. Four configurations preserve exact frozen V7/V8 files, IDs, symbolic theories, prompts, labels, splits and manifests. The release has **190,688 serialized rows**; repeated training rows and related transformations are not independent worlds. Do not combine configurations into one train/test split. The [dataset card](https://huggingface.co/datasets/axondendriteplus/premise-worlds) explains composition, overlap, licensing, and evaluation units.

TRUE means the statement follows. FALSE means its explicit negation follows. UNKNOWN means neither follows. Missing evidence is not negative evidence; UNKNOWN is not confidence-based deferral.

## Inspect and reproduce

```sh
uv run --extra train --extra analysis --extra baseline python -m pytest -q
python3 research-article/build.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory research-article/dist
```

Open http://localhost:8765 for the research article. Its figures and examples read recorded results; no live model inference occurs in the page.

- [Reproduction guide](docs/REPRODUCING.md): fresh checkout, artifacts, inference, metrics, and training scope.
- [Model index](docs/MODELS.md): all eight public matched-comparison checkpoints and revision pins.
- [V7 methods](docs/V7_PROTOCOL.md) and [V8 methods](docs/V8_PROTOCOL.md): frozen scientific specifications.
- [Research journal](RESEARCH_LOG.md): successful results, negative results, corrections, and historical publication states.
- [V1–V8 reports](reports): original outcomes, figures and compact evidence.
- [Original architecture](docs/ARCHITECTURE.md): token head, classification head, and scratch baseline.

The final study passed **126 CPU tests**, two independent raw-count audits and **44 metric recomputations**, covering **165,344 prediction rows**. Public model bundles have separate publication verification. Historical private-release receipts and archived source hashes describe their original versions; subsequent naming and documentation changes do not rewrite them.

## Scope

This is controlled research on one pretrained model family and narrow synthetic distributions. Shared templates, vocabulary, and rule skeletons remain. The experiments do not establish unrestricted logical reasoning, architecture novelty, or reliable business decisions. The original scratch baseline differs in size, tokenizer and attention, so its chance-level result does not isolate the effect of pretraining.

The public code retains the historical `decision_lab` Python package and study file names to preserve reproducibility. The project was previously called Decision Model Lab. Selected V7/V8 bundles are public; earlier private archives and intermediate training checkpoints are not part of this public model release. Full optimizer/RNG recovery states were temporary and are not included.

## License and related work

Project-authored code and generated data: [Apache 2.0](LICENSE). Retain [NOTICE](NOTICE). Pretrained model and dependencies retain their own attribution. RuleTaker, ProofWriter, temperature scaling, and consistency regularization are prior work; this project does not claim to invent those methods. See the article references and frozen protocols.
