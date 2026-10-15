# Premise models

All eight selected checkpoints from the V7 curriculum and V8 paired-loss comparisons. These are full-fine-tuned SmolLM2-360M decision heads with three semantic labels, not chat models.

| Study | Seed | Arm | Selected update | Model card |
| --- | ---: | --- | ---: | --- |
| V7 | 731 | Old curriculum | 900 | [premise-v7-r1-control](https://huggingface.co/axondendriteplus/premise-v7-r1-control) |
| V7 | 731 | Multi-entity curriculum | 1134 | [premise-v7-r1-multi-entity](https://huggingface.co/axondendriteplus/premise-v7-r1-multi-entity) |
| V7 | 947 | Old curriculum | 900 | [premise-v7-r2-control](https://huggingface.co/axondendriteplus/premise-v7-r2-control) |
| V7 | 947 | Multi-entity curriculum | 900 | [premise-v7-r2-multi-entity](https://huggingface.co/axondendriteplus/premise-v7-r2-multi-entity) |
| V8 | 1601 | Paired cross-entropy | 1134 | [premise-v8-r1-paired-ce](https://huggingface.co/axondendriteplus/premise-v8-r1-paired-ce) |
| V8 | 1601 | Paired CE + consistency | 1134 | [premise-v8-r1-consistency](https://huggingface.co/axondendriteplus/premise-v8-r1-consistency) |
| V8 | 1801 | Paired cross-entropy | 1134 | [premise-v8-r2-paired-ce](https://huggingface.co/axondendriteplus/premise-v8-r2-paired-ce) |
| V8 | 1801 | Paired CE + consistency | 1134 | [premise-v8-r2-consistency](https://huggingface.co/axondendriteplus/premise-v8-r2-consistency) |

## Inference

Use the repository’s Python 3.12 locked environment:

```sh
uv sync --extra train --extra analysis --extra baseline
uv run python examples/predict.py --model axondendriteplus/premise-v8-r2-consistency
```

Pass `--revision <commit>` from the publication receipt to pin the exact bundle. The loader checks the SHA-256 manifest, restores the custom classification head and saved temperature, and runs the exact held-out prompts in `examples/prompts.json`. No GPU service is needed.

Each card describes the matching dataset config, training exposure, checkpoint selection, original evaluation and limitations. The original aggregate scores use BF16; CPU verification uses FP32 on eight saved reference prompts.

## Provenance

Public packaging preserves original weights, tokenizer and inference code. README/card and provenance metadata have new hashes, so each public bundle has a new artifact manifest and HF commit. Historical private bundles and their pinned manifests remain unchanged. `public-release.json` inside each bundle identifies the original revision and exact weight SHA-256.

All V7/V8 models must be compared on the matching config of [Premise Worlds](https://huggingface.co/datasets/axondendriteplus/premise-worlds). V7 complete base worlds and V8 complete paired worlds are distinct endpoints. The highest opened test score is not a new universal model-selection rule.

See [V7 results](../reports/V7_RESULTS.md), [V8 results](../reports/V8_RESULTS.md), and [reproduction](REPRODUCING.md).

## Pinned public revisions

| Repository | Public commit |
| --- | --- |
| [axondendriteplus/premise-v7-r1-control](https://huggingface.co/axondendriteplus/premise-v7-r1-control) | [`157e587e557b1ce2b11ef7920ad24817b10c8f86`](https://huggingface.co/axondendriteplus/premise-v7-r1-control/tree/157e587e557b1ce2b11ef7920ad24817b10c8f86) |
| [axondendriteplus/premise-v7-r1-multi-entity](https://huggingface.co/axondendriteplus/premise-v7-r1-multi-entity) | [`42f6df7804d334757a81d39ac960acd3d022c5fc`](https://huggingface.co/axondendriteplus/premise-v7-r1-multi-entity/tree/42f6df7804d334757a81d39ac960acd3d022c5fc) |
| [axondendriteplus/premise-v7-r2-control](https://huggingface.co/axondendriteplus/premise-v7-r2-control) | [`e4deb9a42b91c369cdc27b911f7e16fa4bd32fcc`](https://huggingface.co/axondendriteplus/premise-v7-r2-control/tree/e4deb9a42b91c369cdc27b911f7e16fa4bd32fcc) |
| [axondendriteplus/premise-v7-r2-multi-entity](https://huggingface.co/axondendriteplus/premise-v7-r2-multi-entity) | [`94f8868b00aed4388243fe43b4531772939a4776`](https://huggingface.co/axondendriteplus/premise-v7-r2-multi-entity/tree/94f8868b00aed4388243fe43b4531772939a4776) |
| [axondendriteplus/premise-v8-r1-paired-ce](https://huggingface.co/axondendriteplus/premise-v8-r1-paired-ce) | [`e6577327150614975fd4dc5f6967234e2a1d254e`](https://huggingface.co/axondendriteplus/premise-v8-r1-paired-ce/tree/e6577327150614975fd4dc5f6967234e2a1d254e) |
| [axondendriteplus/premise-v8-r1-consistency](https://huggingface.co/axondendriteplus/premise-v8-r1-consistency) | [`61ad5248c6330d07a87b3e83b2abe4c38eb6c888`](https://huggingface.co/axondendriteplus/premise-v8-r1-consistency/tree/61ad5248c6330d07a87b3e83b2abe4c38eb6c888) |
| [axondendriteplus/premise-v8-r2-paired-ce](https://huggingface.co/axondendriteplus/premise-v8-r2-paired-ce) | [`441fe30d022e0b2ffa9c7ae32a7507369745db31`](https://huggingface.co/axondendriteplus/premise-v8-r2-paired-ce/tree/441fe30d022e0b2ffa9c7ae32a7507369745db31) |
| [axondendriteplus/premise-v8-r2-consistency](https://huggingface.co/axondendriteplus/premise-v8-r2-consistency) | [`fd52d3fc0bd81504b72cb174d36d6f8613de6ebb`](https://huggingface.co/axondendriteplus/premise-v8-r2-consistency/tree/fd52d3fc0bd81504b72cb174d36d6f8613de6ebb) |

Dataset pin: `f5d6c9458e0331dbe21a0a2888dc9aa39d5bd24b`. Machine-readable pins and eight-input CPU verification receipts: [release/manifest.json](../release/manifest.json). All nine Hub repositories were checked anonymously at these revisions.
