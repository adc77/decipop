# Develop and reproduce decipop

The [article is public](https://adc77.github.io/decipop/). Reading it requires no
installation or local server. This guide is for running code and checking results.

## Environment and checks

```sh
git clone https://github.com/adc77/decipop.git
cd decipop
uv sync --extra train --extra cloud --extra analysis --extra baseline
uv run pytest -q
uv run ruff check src tests tools examples
```

Use Python 3.12. CPU tests do not submit Modal jobs. The distribution is named
`decipop`; the import package remains `decision_lab` for saved-model compatibility.

## Models and precision

Choose a [pinned checkpoint](MODELS.md), then run:

```sh
uv run python examples/predict.py --model axondendriteplus/decipop-v8-r2-consistency --revision <commit>
```

The example verifies the bundle's SHA-256 inventory, restores its custom decision
head and saved temperature, and evaluates `examples/prompts.json`. These are full
prompts, not isolated questions. Maximum length is 512 tokens; longer inputs fail.
The weights are not stock chat checkpoints. Standalone inference uses the bundle's
`infer.py`, `code/`, tokenizer and requirements.

`decision_lab.release.verify_release(Path(bundle_directory))` checks eight stored
FP32 references. Aggregate reported results used BF16. Fresh FP32 predictions are
not exact reproductions of all BF16 logits. Rename-only releases preserve weight,
tokenizer and inference hashes; publication records identify the earlier revision
at which each full fresh-download CPU check ran.

## Data and historical results

The public `axondendriteplus/decipop-worlds` dataset has four configs: `v7-r1`,
`v7-r2`, `v8-r1`, `v8-r2`. Download a pinned snapshot with `repo_type="dataset"`.
`sha256.json` records each normalized file's digest and original source digest.
`normalization.json` describes semantic checks; `row-counts.json` has split counts.

The public release replaces one entity symbol consistently with `EntityA`.
**Logical labels are preserved, but model tokenization can change. Original scores
must not be presented as evaluations on the normalized inputs.** Original files
are retained privately. Exact replay of the reported aggregate scores requires
those original files; public normalized data support a new, explicitly named
re-evaluation instead. No such re-evaluation has been performed in this release.

Do not concatenate control and treatment training files. They are experimental
arms and intentionally identical in V8. Keep related questions and edits together
by family. Do not merge configs into a single fresh benchmark.

Public `reports/v7-evidence` and `reports/v8-evidence` contain compact raw-count
audits, numerical checks, selection records and protocol seals. Full per-question
logits remain in retained private run directories. They are not a public raw-logit
archive. World-sampling intervals condition on fixed selected models and resample
three-world rotation blocks; they do not estimate a population of training seeds.

## Code organization and frozen source

Shared implementation lives in `src/decision_lab/{data,models,training,evaluation}`.
Study-specific logic and former root scripts live in `src/decision_lab/studies/vN`.
For example, `v8_raw_audit.py` is now:

```sh
uv run python -m decision_lab.studies.v8.raw_audit --help
```

Module imports are supported; individual file execution is not required. The
[refactor map](../release/refactor-map.json) records every old path and module.
Some historical orchestration modules are intentionally stateful and expect the
owner's run directories. Do not invoke them merely to discover their usage.

The `v0.1.0` tag preserves original paths, historical manifests and publication
receipts. Frozen scientific documents describe that source. Moving modules and
changing the default entity vocabulary change source identities, so current code
will not pass old source seals. No historical hashes were rewritten. For an exact
historical implementation, use a separate checkout of `v0.1.0`; reuse of an old run
identity with new source should fail.

## Article development

See [ARTICLE_DEVELOPMENT.md](ARTICLE_DEVELOPMENT.md). The builder reads committed
receipts and works without private run directories. GitHub Pages publishes `site/`.

## New training

Training requires Modal authentication, historical parents/exclusions, a new run
identity and an explicitly bounded allocation. No new training is part of this
release. Selected weights support new fine-tuning; deleted temporary optimizer/RNG
states do not support exact continuation. Source seals deliberately prevent
silently resuming a frozen study with modified code.
