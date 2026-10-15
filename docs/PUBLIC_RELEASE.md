# Premise public release

The owner authorized the project rename, public GitHub repository, selected public models, dataset release, model cards, and article links on 2026-10-15.

## Scope

- Repository: [adc77/premise](https://github.com/adc77/premise), previously `decision-model-lab`.
- Dataset: [Premise Worlds](https://huggingface.co/datasets/axondendriteplus/premise-worlds), four exact frozen V7/V8 configurations, 190,688 serialized rows. Rows are not independent worlds.
- Models: all eight selected V7/V8 checkpoints, including both control/treatment arms and both fresh repetitions. See [MODELS.md](MODELS.md).
- Article: revised local preview in the owner's axondendrite.com style, with readable result definitions, a genuine held-out example, GitHub source links, and HF dataset/model links. No budget framing.

## State and verification

All eight selected models and the dataset are public, verified at pinned revisions, and readable anonymously. The representative source is on public GitHub `main`; all 20 GitHub links in the article returned 200 without authentication. The documented inference command also passed from a clean checkout with implicit HF authentication disabled. Machine-readable model/data pins are in [release/manifest.json](../release/manifest.json). Historical scientific studies, model weights, tokenizer, inference source, selection and evaluation outcomes remain unchanged. The public packaging adds new model cards and provenance manifests with their own identities.

The release re-ran all 126 CPU tests. The dataset's four configurations load through the Hugging Face datasets interface. Model releases are freshly downloaded and checked on their eight FP32 reference inputs before visibility changes. An initial read connection interruption and later private-storage-limit rejection are retained locally; verified releases are made public individually so no paid storage upgrade is needed.

The original 21 historical selected bundles, private archives and intermediate candidate storage remain separate from these eight public packaging copies. Temporary optimizer/RNG states are not released. Earlier historical publication notes describe their original state and are not rewritten to claim they were public then.

No new training, seeds, model architecture experiment, GPU jobs, or public deployment of the personal website is part of this publication.

## Release verification

[Machine-readable checks](../release/verification.json) record all nine anonymous Hub checks, 64 fresh reference predictions, the 126-pass CPU suite, four dataset configurations, the clean-checkout article rebuild and documented CPU inference. [The CPU example output](../release/example-cpu.json) is separately labeled: it is a publication smoke check, not a replacement for original BF16 aggregate scores.
