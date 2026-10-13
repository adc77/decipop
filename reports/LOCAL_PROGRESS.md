# Local implementation checkpoint

Status on 2026-10-12: all three models completed training, selection, calibration, final evaluation, and GPU export verification. Implementation remains local and uncommitted on `feat/data-foundation`.

## Data and access

The personal Modal workspace is `amandogra2016`. The billing screenshot verified $30 starting credits and a saved $0 net spend limit. The three private research model repositories belong to `axondendriteplus`; pinned revisions and download verification are recorded in [COSTS_AND_RELEASES.md](COSTS_AND_RELEASES.md).

The frozen dataset is `data/pilot-v3`, with 31,503 questions in 10,501 theory families.

| Split | Questions | Families |
| --- | ---: | ---: |
| Train | 20,001 | 6,667 |
| Development | 2,001 | 667 |
| Calibration | 2,001 | 667 |
| ID test | 3,000 | 1,000 |
| Depth test | 1,500 | 500 |
| Wording test | 1,500 | 500 |
| Structure test | 1,500 | 500 |

The formal solver, renderer, balancing, canonical family isolation, and reserved conditions pass validation. Development unigram accuracy is 33.53%. Both tokenizers fit the 512-token limit: maximum pretrained length 334, scratch length 283. The scratch tokenizer learned 514 entries; its configured embedding has 8,192 slots.

## GPU gates and frozen training

Both pretrained approaches memorized 32 examples in the initial pilots. The subsequent gate also fit the scratch model to 100% and verified a bitwise-identical next CUDA update after checkpoint reload for all three architectures.

The gate measured approximately 1.16 seconds per pretrained optimizer update, 0.048 seconds for scratch, and about 7.4 GB peak allocated memory for pretrained training. Full development inference takes about 15 seconds per pretrained model.

Protocol identity: `b622cfa2d34eb2fb779a4692ceb69290e053d1dcd1b307da68d203bbd25f4d3e`. Each architecture has three passes over the same 20,001 training questions, an effective batch of 32, and 1,878 updates. Selection uses development restricted-class NLL. Calibration, robustness variants, and performance inputs are specified before final test scoring.

Modal calls use one L4, 4 physical CPU cores, 32 GiB RAM, explicit timeouts, one container, and no automatic whole-job retry. The local invocation ledger reserves each call before submission; failed calls retain their allocation. Temporary recovery files are pruned after committed saves. No serving endpoint is deployed.

## Evaluation and artifacts

Implemented: temperature scaling, calibration-selected abstention thresholds, accuracy/F1/NLL/Brier/ECE, confusion matrices, family bootstrap intervals, depth and wording tests, six oracle-checked robustness transformations, synchronized inference benchmarks, and result plots.

A local integration test runs calibration, evaluation, export, checksum verification, and standalone inference for all three tiny model configurations. The full-sized models were evaluated on Modal only after checkpoint selection and calibration were frozen. Raw predictions for all six models and ten held-out suites are retained locally.

Private release tooling downloads checksum-verified model bundles from Modal, uploads only their manifest-listed files, pins Hugging Face commits, and checks a fresh download in a separate CPU process. The bundles include custom loaders, tokenizer, FP32 weights, calibration, protocol, evaluation, attribution, and fixed inference references.

## Research result and retained records

The token model reaches 96.60% ID accuracy, the head 97.37%, and scratch 33.33%. Both fine-tuned models fail all 142 fact-flip questions whose correct answer becomes UNKNOWN when derivation support is lost. The head does not demonstrate faster inference. Read [RESULTS.md](RESULTS.md), [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md), and [LEARNING.md](LEARNING.md) before interpreting the checkpoints.

Local `runs/study-v1` retains the protocol, calibration freeze, selected checkpoint metadata, successful training chunk logs, raw predictions, bootstrap statistics, invocation ledger, interruption/replay evidence, billing snapshots, and release receipt. `artifacts` retains final and selected weights, exported bundles, fresh Hub downloads, archived Modal metadata, and source snapshots. Temporary optimizer recovery states are not retained after the completed study.

The final accounting snapshot and resource cleanup evidence are linked from [COSTS_AND_RELEASES.md](COSTS_AND_RELEASES.md). Storage accounting can lag and deleted snapshots may remain billable for up to four days. The proposed next counterfactual training experiment is documented but has not been run.
