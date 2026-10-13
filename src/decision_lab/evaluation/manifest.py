import json
from pathlib import Path

from decision_lab.data.validate import read_examples
from decision_lab.evaluation.robustness import generate_variants
from decision_lab.tokenization import load_tokenizer
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.core import get_tokenizer


def prepare_test_inputs(dataset: Path, audit: Path, output: Path, pilot, evaluation):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reserved test inputs: {output}")
    rows = read_examples(dataset / "test_id.jsonl")
    variants, summary = generate_variants(rows, evaluation.robustness_families, evaluation.seed)
    pretrained = load_tokenizer(pilot)
    scratch = get_tokenizer("scratch", pilot, audit)
    output.mkdir(parents=True)
    inventory = {}
    for name, group in variants.items():
        prompts = [item["example"].prompt for item in group]
        lengths = {}
        for tokenizer_name, tokenizer in (("pretrained", pretrained), ("scratch", scratch)):
            counts = [len(ids) for ids in tokenizer(prompts, truncation=False)["input_ids"]]
            if max(counts) > pilot.max_length:
                raise ValueError(f"Overlong reserved {name} input for {tokenizer_name}")
            lengths[tokenizer_name] = {"min": min(counts), "max": max(counts)}
        path = output / f"{name}.jsonl"
        with path.open("w") as stream:
            for item in group:
                payload = {
                    "example": item["example"].model_dump(mode="json"),
                    "base_example_id": item["base_example_id"],
                    "label_changed": item["label_changed"],
                }
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        inventory[path.name] = file_hash(path)
        summary[name]["token_lengths"] = lengths
    encoded = pretrained([row.prompt for row in rows], truncation=False)["input_ids"]
    order = sorted(range(len(rows)), key=lambda index: len(encoded[index]))
    start = len(order) // 2 - evaluation.performance_requests // 2
    indices = order[start : start + evaluation.performance_requests]
    manifest = {
        "seed": evaluation.seed,
        "files": inventory,
        "transformations": summary,
        "performance_test_id_indices": indices,
        "performance_example_ids": [rows[index].example_id for index in indices],
        "performance_selection": "central length bucket; identical semantic inputs across models",
        "paraphrase_review": "oracle checked; no independent human linguistic review",
        "test_predictions_inspected": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
