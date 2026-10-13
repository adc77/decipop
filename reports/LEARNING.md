# What we learned about decision models

## One numerical decision

```text
Decide using only these facts and rules. TRUE means the statement follows; FALSE means its explicit negation follows; UNKNOWN means neither follows. Absence is not negation. Answer A for TRUE, B for FALSE, or C for UNKNOWN.

Facts:
Farah is not kind.
Farah supports Chen.
Evan does not know Gabe.
Aman is calm.
Gabe is round.

Rules:
For every person mentioned: if person 1 is calm, then person 1 is large.

Statement: Aman is not large.

Answer:
```

Oracle answer: FALSE; shortest proof depth: 1.

- Given: Aman is calm.
- Derive: Aman is large.

| Model | TRUE logit | FALSE logit | UNKNOWN logit | Prediction | Cal confidence |
| --- | ---: | ---: | ---: | --- | ---: |
| pretrained | 20.2500 | 20.0000 | 19.2500 | TRUE | 0.3403 |
| token | 23.0000 | 35.0000 | 31.7500 | FALSE | 0.9443 |
| head | -3.5469 | 6.5938 | 1.4844 | FALSE | 0.9874 |
| scratch | -0.0586 | -0.0986 | -0.0718 | TRUE | 0.3336 |

For the head, raw softmax probabilities are [3.920579868770827e-05, 0.993957433780509, 0.006003360420803287]. Restricted cross-entropy is −log P(FALSE) = 0.006061. Temperature scaling divides the three logits by a positive fitted number; it changes confidence without changing the top class.

The token model trains against the full vocabulary. Its three displayed decision probabilities are renormalized over A/B/C. Its training loss additionally includes the mass outside those tokens; outside mass and invalid unrestricted answers are recorded separately in the evaluation.

## Attention and output shapes

With B prompts and padded length L, the pretrained backbone produces [B,L,960]. Causal attention allows each position to read its prefix. We gather the last nonpadding position, yielding [B,960]. The token projection produces [B,49152], then selects A/B/C. The decision head projects directly to [B,3]. Both approaches update the full backbone; this is full fine-tuning rather than LoRA.

The scratch encoder uses 6 layers, 6 attention heads and width 384. Each nonpadding position can attend to all supplied positions. The first [CLS] representation is projected to three logits. This changes attention, model size, tokenizer, and initialization together; the comparison cannot isolate pretraining alone.

The scratch tokenizer sees training prompts only and can encode unseen text through its byte alphabet. Learning this bounded task from random weights does not teach general conversational language understanding.

## Checkpoint recovery

Recovery stores model weights, Adam moments, scheduler state, example order/cursor, and Python/Torch CPU/CUDA RNG states. A checkpoint is complete only after its file checksums and protocol identity are written. The GPU gate verified a bitwise-identical next update after saving and reloading each architecture.

## A deeper held-out case

```text
Decide using only these facts and rules. TRUE means the statement follows; FALSE means its explicit negation follows; UNKNOWN means neither follows. Absence is not negation. Answer A for TRUE, B for FALSE, or C for UNKNOWN.

Facts:
Gabe does not support Farah.
Hana is blue.
Farah is large.
Gabe supports Chen.
Aman supports Gabe.
Chen knows Hana.
Chen is kind.
Gabe follows Farah.
Gabe is not calm.
Gabe does not support Aman.
Farah follows Chen.
Gabe does not follow Aman.

Rules:
Whenever person 1 supports person 2 and person 2 is bright, person 1 is round; this applies to all people mentioned.
Whenever person 1 follows person 2 and person 2 is not kind, person 1 is bright; this applies to all people mentioned.
Whenever person 1 follows person 2 and person 2 is large, person 1 is not kind; this applies to all people mentioned.
Whenever person 1 knows person 2 and person 2 is blue, person 1 is large; this applies to all people mentioned.

Statement: Aman is not round.

Answer:
```

Oracle: FALSE; shortest proof depth: 4.

- Given: Aman supports Gabe.
- Given: Gabe follows Farah.
- Given: Farah follows Chen.
- Given: Chen knows Hana.
- Given: Hana is blue.
- Derive: Chen is large.
- Derive: Farah is not kind.
- Derive: Gabe is bright.
- Derive: Aman is round.

- pretrained: TRUE, calibrated confidence 34.03%.
- token: UNKNOWN, calibrated confidence 85.02%.
- head: FALSE, calibrated confidence 82.70%.
- scratch: TRUE, calibrated confidence 33.36%.

This proof comes from the symbolic oracle. It is not a claim about the model's internal reasoning. An incorrect answer is compatible with exploiting shallow patterns; an individual correct answer cannot establish rule following.

## UNKNOWN versus abstention

UNKNOWN is a legitimate answer when neither side is derivable. Abstention means deferring a low-confidence prediction of any class. Confidence thresholds are chosen on calibration examples and then applied unchanged to held-out data. Their achieved coverage and error rates can shift on deeper or reworded inputs.
