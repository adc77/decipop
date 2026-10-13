import torch
from torch import nn
from torch.nn import functional as F

from decision_lab.model_config import ScratchConfig


class Attention(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.heads = config.heads
        self.head_size = config.hidden_size // config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden, attention_mask):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_size)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None, None, :].bool(),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = Attention(config)
        self.feedforward_norm = nn.LayerNorm(config.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, attention_mask):
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), attention_mask))
        return hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))


class ScratchDecisionModel(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_id)
        self.positions = nn.Embedding(config.max_length, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.classes)
        self.apply(self.initialize)
        with torch.no_grad():
            self.tokens.weight[config.pad_id].zero_()

    def initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        length = input_ids.shape[1]
        if length > self.config.max_length:
            raise ValueError("Input exceeds scratch model's position limit")
        if not torch.all(attention_mask[:, 0]):
            raise ValueError("The first decision token cannot be padding")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.tokens(input_ids) + self.positions(positions)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.head(self.final_norm(hidden[:, 0]))

    def loss(self, logits, labels):
        return F.cross_entropy(logits.float(), labels)

    def class_logits(self, logits):
        return logits
