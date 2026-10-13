import hashlib
import json
from pathlib import Path

from decision_lab.data.validate import read_examples, validate_dataset
from decision_lab.model_config import PilotConfig, ScratchConfig


def load_tokenizer(config: PilotConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def answer_ids(tokenizer, config: PilotConfig) -> list[int]:
    prefix = "Answer:"
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    result = []
    for token in config.class_tokens:
        combined = tokenizer.encode(prefix + config.answer_prefix + token, add_special_tokens=False)
        if combined[: len(prompt_ids)] != prompt_ids or len(combined) != len(prompt_ids) + 1:
            raise ValueError(f"Answer continuation is not a stable single token: {token}")
        result.append(combined[-1])
    if len(set(result)) != len(result):
        raise ValueError("Answer token IDs must be distinct")
    return result


def train_scratch_tokenizer(directory: Path, output: Path, config: ScratchConfig):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite tokenizer: {output}")
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
    )
    train = read_examples(directory / "train.jsonl")
    backend.train_from_iterator((row.prompt for row in train), trainer=trainer)
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(output)
    return tokenizer


def audit_tokens(directory: Path, output: Path, pilot: PilotConfig, scratch: ScratchConfig):
    validate_dataset(directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite token audit: {output}")
    output.mkdir(parents=True)
    pretrained = load_tokenizer(pilot)
    mapping = answer_ids(pretrained, pilot)
    custom = train_scratch_tokenizer(directory, output / "scratch-tokenizer", scratch)
    summaries = {}
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, meta in manifest["splits"].items():
        rows = read_examples(directory / meta["file"])
        lengths = {}
        for tokenizer_name, tokenizer in (("pretrained", pretrained), ("scratch", custom)):
            counts = [len(tokenizer.encode(row.prompt, add_special_tokens=True)) for row in rows]
            lengths[tokenizer_name] = {
                "min": min(counts),
                "max": max(counts),
                "mean": sum(counts) / len(counts),
                "over_limit": sum(n > pilot.max_length for n in counts),
            }
        summaries[name] = lengths
    report = {
        "model_id": pilot.model_id,
        "model_revision": pilot.revision,
        "answer_ids": mapping,
        "class_labels": pilot.class_labels,
        "max_length": pilot.max_length,
        "dataset_manifest_sha256": hashlib.sha256(
            (directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "splits": summaries,
        "within_limit": all(
            stats["over_limit"] == 0 for split in summaries.values() for stats in split.values()
        ),
        "scratch_vocab_size": custom.vocab_size,
    }
    (output / "token-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["within_limit"]:
        raise ValueError(
            "Token audit found overlong inputs; revise generation before freezing data"
        )
    return report
