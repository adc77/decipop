import torch
from torch import nn
from torch.nn import functional as F


def last_hidden(backbone, input_ids, attention_mask):
    output = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    positions = attention_mask.sum(dim=1) - 1
    if torch.any(positions < 0):
        raise ValueError("Each prompt requires at least one token")
    return output.last_hidden_state[
        torch.arange(input_ids.shape[0], device=input_ids.device), positions
    ]


class TokenDecisionModel(nn.Module):
    def __init__(self, causal_lm, answer_ids):
        super().__init__()
        self.backbone = causal_lm.model
        self.vocabulary_head = causal_lm.lm_head
        self.register_buffer("answer_ids", torch.tensor(answer_ids, dtype=torch.long))

    def forward(self, input_ids, attention_mask):
        hidden = last_hidden(self.backbone, input_ids, attention_mask)
        return self.vocabulary_head(hidden)

    def loss(self, logits, labels):
        return F.cross_entropy(logits.float(), self.answer_ids[labels])

    def class_logits(self, logits):
        return logits[:, self.answer_ids]

    def outside_probability(self, logits):
        return 1 - logits.float().softmax(-1)[:, self.answer_ids].sum(-1)
