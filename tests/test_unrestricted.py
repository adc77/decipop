from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from decision_lab.evaluation.config import load_evaluation_config
from decision_lab.evaluation.study import decision_predictions, scored
from decision_lab.models.token_model import TokenDecisionModel


def test_unrestricted_accuracy_uses_full_vocabulary_tie_order():
    class Backbone(nn.Module):
        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 1))

    class Vocabulary(nn.Module):
        def forward(self, hidden):
            logits = torch.zeros(len(hidden), 10)
            logits[:, 8:10] = 1
            return logits

    model = TokenDecisionModel(SimpleNamespace(model=Backbone(), lm_head=Vocabulary()), [7, 9, 8])
    predictions = decision_predictions(model, [[1, 2]], [2], 1, 0, "cpu")
    assert predictions["logits"].argmax(-1).item() == 1
    assert predictions["unrestricted_prediction"].item() == 2
    metrics = scored(
        predictions,
        {"temperature": 1, "thresholds": {}},
        load_evaluation_config(Path("configs/evaluation.yaml")),
    )
    assert metrics["raw"]["accuracy"] == 0
    assert metrics["unrestricted_accuracy"] == 1
    assert metrics["unrestricted_invalid_answer_rate"] == 0
