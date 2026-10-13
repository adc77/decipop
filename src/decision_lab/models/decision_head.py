from torch import nn
from torch.nn import functional as F

from decision_lab.models.token_model import last_hidden


class HeadDecisionModel(nn.Module):
    def __init__(self, backbone, hidden_size, classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, classes)

    def forward(self, input_ids, attention_mask):
        return self.head(last_hidden(self.backbone, input_ids, attention_mask))

    def loss(self, logits, labels):
        return F.cross_entropy(logits.float(), labels)

    def class_logits(self, logits):
        return logits
