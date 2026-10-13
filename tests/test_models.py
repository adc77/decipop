import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from decision_lab.model_config import ScratchConfig
from decision_lab.models.decision_head import HeadDecisionModel
from decision_lab.models.scratch_transformer import Attention, ScratchDecisionModel
from decision_lab.models.token_model import TokenDecisionModel


def test_token_model_matches_full_causal_lm_answer_logits():
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    base = LlamaForCausalLM(config).eval()
    inputs = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    expected = base(inputs, attention_mask=mask).logits[torch.arange(2), torch.tensor([2, 3])]
    model = TokenDecisionModel(base, [7, 8, 9])
    actual = model(inputs, mask)
    torch.testing.assert_close(actual, expected)
    labels = torch.tensor([0, 2])
    torch.testing.assert_close(
        model.loss(actual, labels), F.cross_entropy(expected, torch.tensor([7, 9]))
    )
    assert model.class_logits(actual).shape == (2, 3)
    assert torch.all(model.outside_probability(actual) > 0)


def test_head_receives_gradients_and_ignores_right_padding():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    backbone = LlamaForCausalLM(config).model
    model = HeadDecisionModel(backbone, 16, 3)
    inputs = torch.tensor([[1, 2, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    logits = model(inputs, mask)
    model.loss(logits, torch.tensor([1])).backward()
    assert model.head.weight.grad.norm() > 0
    assert model.backbone.embed_tokens.weight.grad.norm() > 0
    model.eval()
    torch.testing.assert_close(model(inputs, mask), model(inputs[:, :3], mask[:, :3]))


def small_config():
    return ScratchConfig(
        vocab_size=32,
        hidden_size=16,
        heads=2,
        layers=1,
        intermediate_size=32,
        max_length=16,
        dropout=0.0,
        classes=3,
        pad_id=0,
        initializer_std=0.02,
    )


def test_scratch_attention_matches_explicit_attention_reference():
    attention = Attention(small_config()).eval()
    inputs = torch.randn(2, 4, 16)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    qkv = attention.qkv(inputs).view(2, 4, 3, 2, 8).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    scores = query @ key.transpose(-2, -1) / (8**0.5)
    scores = scores.masked_fill(~mask[:, None, None, :].bool(), float("-inf"))
    hidden = (scores.softmax(-1) @ value).transpose(1, 2).reshape(2, 4, 16)
    torch.testing.assert_close(attention(inputs, mask), attention.output(hidden))


def test_scratch_padding_invariance_and_training_gradients():
    model = ScratchDecisionModel(small_config()).eval()
    inputs = torch.tensor([[2, 7, 8, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    torch.testing.assert_close(model(inputs, mask), model(inputs[:, :3], mask[:, :3]))
    model.train()
    model.loss(model(inputs, mask), torch.tensor([2])).backward()
    assert model.head.weight.grad.norm() > 0
    assert model.blocks[0].attention.qkv.weight.grad.norm() > 0
    assert torch.isfinite(model.tokens.weight.grad).all()


def test_scratch_model_can_fit_a_tiny_fixed_set():
    torch.manual_seed(17)
    model = ScratchDecisionModel(small_config())
    inputs = torch.tensor([[2, 4, 7], [2, 5, 7], [2, 6, 7]])
    mask = torch.ones_like(inputs)
    labels = torch.tensor([0, 1, 2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(inputs, mask), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    assert torch.equal(model(inputs, mask).argmax(-1), labels)


def test_scratch_safetensors_reload_preserves_predictions(tmp_path):
    from safetensors.torch import load_model, save_model

    model = ScratchDecisionModel(small_config()).eval()
    inputs = torch.tensor([[2, 7, 8]])
    mask = torch.ones_like(inputs)
    expected = model(inputs, mask)
    artifact = tmp_path / "model.safetensors"
    save_model(model, artifact)
    loaded = ScratchDecisionModel(small_config()).eval()
    load_model(loaded, artifact)
    torch.testing.assert_close(loaded(inputs, mask), expected)
