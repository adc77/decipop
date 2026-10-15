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
