# Decision Model Lab architecture

Every model receives the same English facts, rules, question, and answer mapping. Labels and oracle proofs are metadata used for training and evaluation; they are not embedded in the model input.

```mermaid
flowchart LR
    P[English prompt] --> T[Pinned SmolLM2 tokenizer]
    T --> A[Causal backbone A]
    T --> B[Causal backbone B]
    A --> L[Last nonpadding hidden state]
    L --> V[Tied vocabulary projection]
    V --> D1[A / B / C decision logits]
    B --> H[Last nonpadding hidden state]
    H --> C[New three-class head]
    C --> D2[TRUE / FALSE / UNKNOWN logits]
    P --> S[Train-only byte BPE]
    S --> E[Randomly initialized bidirectional encoder]
    E --> CLS[First CLS hidden state]
    CLS --> D3[Three-class head]
```

A and B start independently from the same pinned pretrained backbone. Both update its full weights. A trains with full-vocabulary cross-entropy and additionally reports unrestricted invalid answers. B trains with three-class cross-entropy. C uses established transformer components and learns this task from random initialization; it is not general language pretraining.

Temperature scaling operates on the three class logits after training. It changes confidence, not the selected class. A confidence threshold may defer any class, including UNKNOWN. UNKNOWN itself means neither the queried statement nor its explicit negation is derivable.

## Study sequence

```mermaid
flowchart TD
    G[Seeded formal generator] --> O[Symbolic oracle and English rendering]
    O --> I[Family isolation and tokenizer audit]
    I --> K[GPU fit, throughput and exact-resume gate]
    K --> F[Freeze data, exposure, settings and test variants]
    F --> A[Train token model]
    A --> B[Train direct head]
    B --> C[Train scratch encoder]
    C --> S[Freeze development-selected checkpoints]
    S --> CAL[Fit temperatures and thresholds on calibration data]
    CAL --> E[Score untouched tests and robustness variants]
    E --> R[Record metrics, timings, raw predictions and plots]
    R --> X[Export custom model bundles and verify GPU reload]
    X --> HF[Upload private Hugging Face revisions]
    HF --> V[Fresh download and separate-process CPU verification]
    V --> P[Preserve final weights and study records]
    P --> D[Remove project-owned temporary Modal storage]
```

Modal runs bounded batch Functions, one GPU call at a time. The laptop controls calls, keeps the invocation ledger, runs approved CPU tests, and uploads verified artifacts. The Hugging Face write token stays on the laptop.

## Recovery and selection

Recovery checkpoints store weights, Adam moments, scheduler state, data cursor, example counts, and Python/Torch RNG states. Files are checksummed before a checkpoint is marked complete. The controller links each chunk to its parent checkpoint and retains two recent recovery points.

Selection uses development three-class NLL, with accuracy breaking an exact tie. It does not use calibration or test results. Final weights are retained even when a previous development checkpoint is selected.

Once test evaluation starts, the training wrapper rejects further training for that study. Evaluation validates the frozen weights, calibration, baseline configuration, test inputs, and evaluation source before scoring. A later experiment needs a new declared protocol.

## Interpreting the comparison

A versus B compares output layers and loss objectives on the same pretrained backbone and exposure. C differs in initialization, parameter count, tokenizer, and attention pattern; its result cannot isolate pretraining alone. All conclusions are limited to a single seed and the generated rule distributions. See the result and learning reports for measured behavior.
