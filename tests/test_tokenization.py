from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import generate_dataset
from decision_lab.model_config import load_scratch_config
from decision_lab.tokenization import train_scratch_tokenizer


def test_train_only_tokenizer_preserves_unseen_entity_names(tmp_path):
    config = load_dataset_config(Path("configs/smoke.yaml"))
    config = config.model_copy(
        update={
            "splits": {
                name: spec.model_copy(update={"families": 2})
                for name, spec in config.splits.items()
            }
        }
    )
    directory = tmp_path / "data"
    generate_dataset(config, directory)
    scratch = load_scratch_config(Path("configs/scratch.yaml"))
    tokenizer = train_scratch_tokenizer(directory, tmp_path / "tokenizer", scratch)
    text = "Inez supports Jules."
    encoded = tokenizer.encode(text)
    assert tokenizer.unk_token_id not in encoded
    assert tokenizer.decode(encoded, skip_special_tokens=True) == text
    assert encoded[0] == tokenizer.cls_token_id
    assert tokenizer.pad_token_id == scratch.pad_id
