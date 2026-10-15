# decipop models

All eight selected checkpoints from the V7 curriculum and V8 paired-loss comparisons. These are full-fine-tuned SmolLM2-360M decision heads with three semantic labels, not chat models.

| Study | Seed | Arm | Selected update | Model card |
| --- | ---: | --- | ---: | --- |
| V7 | 731 | Old curriculum | 900 | [decipop-v7-r1-control](https://huggingface.co/axondendriteplus/decipop-v7-r1-control) |
| V7 | 731 | Multi-entity curriculum | 1134 | [decipop-v7-r1-multi-entity](https://huggingface.co/axondendriteplus/decipop-v7-r1-multi-entity) |
| V7 | 947 | Old curriculum | 900 | [decipop-v7-r2-control](https://huggingface.co/axondendriteplus/decipop-v7-r2-control) |
| V7 | 947 | Multi-entity curriculum | 900 | [decipop-v7-r2-multi-entity](https://huggingface.co/axondendriteplus/decipop-v7-r2-multi-entity) |
| V8 | 1601 | Paired cross-entropy | 1134 | [decipop-v8-r1-paired-ce](https://huggingface.co/axondendriteplus/decipop-v8-r1-paired-ce) |
| V8 | 1601 | Paired CE + consistency | 1134 | [decipop-v8-r1-consistency](https://huggingface.co/axondendriteplus/decipop-v8-r1-consistency) |
| V8 | 1801 | Paired cross-entropy | 1134 | [decipop-v8-r2-paired-ce](https://huggingface.co/axondendriteplus/decipop-v8-r2-paired-ce) |
| V8 | 1801 | Paired CE + consistency | 1134 | [decipop-v8-r2-consistency](https://huggingface.co/axondendriteplus/decipop-v8-r2-consistency) |

## Inference

Use the repository’s Python 3.12 locked environment:

```sh
uv sync --extra train --extra analysis --extra baseline
uv run python examples/predict.py --model axondendriteplus/decipop-v8-r2-consistency
```

Pass `--revision <commit>` from the publication receipt to pin the exact bundle. The loader checks the SHA-256 manifest, restores the custom classification head and saved temperature, and runs the exact held-out prompts in `examples/prompts.json`. No GPU service is needed.

Each card describes the matching dataset config, training exposure, checkpoint selection, original evaluation and limitations. The original aggregate scores use BF16; CPU verification uses FP32 on eight saved reference prompts.

## Provenance

Public packaging preserves original weights, tokenizer and inference code. README/card and provenance metadata have new hashes, so each public bundle has a new artifact manifest and HF commit. Historical private bundles and their pinned manifests remain unchanged. `public-release.json` inside each bundle identifies the original revision and exact weight SHA-256.

The linked [decipop worlds](https://huggingface.co/datasets/axondendriteplus/decipop-worlds) preserves config and split relationships but normalizes one entity name to EntityA. Models were trained and evaluated on the original text. Historical scores are not evaluations on this normalized derivative; tokenization and predictions can change. The original input data are retained privately. V7 complete base worlds and V8 complete paired worlds are distinct endpoints. The highest opened test score is not a new universal model-selection rule.

See [V7 results](../reports/V7_RESULTS.md), [V8 results](../reports/V8_RESULTS.md), and [reproduction](REPRODUCING.md).

## Pinned public revisions

| Repository | Public commit |
| --- | --- |
| [axondendriteplus/decipop-v7-r1-control](https://huggingface.co/axondendriteplus/decipop-v7-r1-control) | [`e37a423f76dd7b82c9a276f7fc21407f324e25ad`](https://huggingface.co/axondendriteplus/decipop-v7-r1-control/tree/e37a423f76dd7b82c9a276f7fc21407f324e25ad) |
| [axondendriteplus/decipop-v7-r1-multi-entity](https://huggingface.co/axondendriteplus/decipop-v7-r1-multi-entity) | [`88b2562ada13ce54a1e6fdf2ea33e72986d96c3a`](https://huggingface.co/axondendriteplus/decipop-v7-r1-multi-entity/tree/88b2562ada13ce54a1e6fdf2ea33e72986d96c3a) |
| [axondendriteplus/decipop-v7-r2-control](https://huggingface.co/axondendriteplus/decipop-v7-r2-control) | [`519d6cf07fd04c3c1423df107b6fee36ad6ba008`](https://huggingface.co/axondendriteplus/decipop-v7-r2-control/tree/519d6cf07fd04c3c1423df107b6fee36ad6ba008) |
| [axondendriteplus/decipop-v7-r2-multi-entity](https://huggingface.co/axondendriteplus/decipop-v7-r2-multi-entity) | [`1c90ddd9ac9fc001582edbcab5fe6c258908b797`](https://huggingface.co/axondendriteplus/decipop-v7-r2-multi-entity/tree/1c90ddd9ac9fc001582edbcab5fe6c258908b797) |
| [axondendriteplus/decipop-v8-r1-paired-ce](https://huggingface.co/axondendriteplus/decipop-v8-r1-paired-ce) | [`9193af7e2f9776671325fb48c8715625e352d085`](https://huggingface.co/axondendriteplus/decipop-v8-r1-paired-ce/tree/9193af7e2f9776671325fb48c8715625e352d085) |
| [axondendriteplus/decipop-v8-r1-consistency](https://huggingface.co/axondendriteplus/decipop-v8-r1-consistency) | [`5e6729066d10ce1d6d8d69ba8f41d751defe425f`](https://huggingface.co/axondendriteplus/decipop-v8-r1-consistency/tree/5e6729066d10ce1d6d8d69ba8f41d751defe425f) |
| [axondendriteplus/decipop-v8-r2-paired-ce](https://huggingface.co/axondendriteplus/decipop-v8-r2-paired-ce) | [`3440070b10e1fc57c74796c65c83c543220852ce`](https://huggingface.co/axondendriteplus/decipop-v8-r2-paired-ce/tree/3440070b10e1fc57c74796c65c83c543220852ce) |
| [axondendriteplus/decipop-v8-r2-consistency](https://huggingface.co/axondendriteplus/decipop-v8-r2-consistency) | [`11ffe22bed8459a3af19a0296a403dd0974648ab`](https://huggingface.co/axondendriteplus/decipop-v8-r2-consistency/tree/11ffe22bed8459a3af19a0296a403dd0974648ab) |

Dataset pin: `f59ba2a8e0e9822d76637d03bf0f781282b82ef1`. Machine-readable pins, unchanged-weight checks and earlier eight-input CPU verification receipts: [release/manifest.json](../release/manifest.json). All eight model repositories and the normalized dataset were checked anonymously at these revisions.
