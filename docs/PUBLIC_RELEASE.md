# decipop public release

- [Research article](https://adc77.github.io/decipop/): hosted publicly; no installation required.
- [Source](https://github.com/adc77/decipop): modular Python implementation and V1–V8 records.
- [Models](MODELS.md): eight selected V7/V8 checkpoints with pinned cards and custom inference bundles.
- [Dataset](https://huggingface.co/datasets/axondendriteplus/decipop-worlds): four normalized configs, 190,688 serialized rows.

The dataset replaces one entity symbol consistently with EntityA. Labels and proof
depths were checked with two oracles. The original text is retained privately.
Historical model scores apply to the original inputs, not this normalized release;
renaming can change tokenization and predictions. Related rows are not independent
worlds. Model weights, tokenizer and inference bytes are unchanged.

[Release pins](../release/manifest.json) identify every public artifact.
[Verification](../release/verification.json) records the 128-pass CPU suite, 11
follow-up smoke-config tests, anonymous dataset loads, model checks, fresh-checkout
inference, deployed assets and public evidence links. [CPU example output](../release/example-cpu.json)
is a publication check, not a new aggregate evaluation.

The original scientific source and publication are preserved at `v0.1.0`. The
modular source release is tagged `v0.2.0`; later documentation cleanup on main
removes duplicate card copies in favor of their pinned Hugging Face originals.
The [journal](../RESEARCH_LOG.md) records the full sequence. No new training or GPU
job is part of this publication. Earlier private archives remain private.
