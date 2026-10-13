import json
import os
import shutil
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from decision_lab.v2.protocol import verify_comparison
from decision_lab.v3.data import generate, prior_worlds
from decision_lab.v3.protocol import freeze, inventory, settings


def main():
    root = Path("runs/study-v3")
    if root.exists():
        raise FileExistsError(root)
    comparison = verify_comparison(Path("runs/study-v2"))
    retention = json.loads(Path("runs/study-v2/retention.json").read_text())
    spec = settings()
    root.mkdir(parents=True)
    candidates = {}
    for arm in ("control", "treatment"):
        source = Path("artifacts/v2-releases") / arm
        original = release_inventory(source)
        for stage, step, presentations in (("early", 250, 8000), ("final", 1878, 60003)):
            name = f"{arm}-{stage}"
            directory = root / "candidates" / name
            directory.mkdir(parents=True)
            for filename in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ):
                shutil.copyfile(source / filename, directory / filename)
            weights = source / "model.safetensors"
            digest = original["files"]["model.safetensors"]
            if stage == "final":
                weights = Path("artifacts/v2-final-weights") / arm / "model.safetensors"
                digest = retention["final_weights"][arm]["weights_sha256"]
            if file_hash(weights) != digest:
                raise ValueError("Retained candidate checksum mismatch")
            os.link(weights, directory / "model.safetensors")
            candidates[name] = {
                "arm": arm,
                "step": step,
                "presentations": presentations,
                "parent_protocol_identity": comparison["arms"][arm],
                "files": inventory(directory),
            }
    excluded, exclusion_files = prior_worlds()
    audit = generate(root / "data", spec, excluded, exclusion_files)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root / "candidates/control-early")
    lengths = {}
    for path in sorted((root / "data").glob("*.jsonl")):
        if path.name == "families.jsonl":
            continue
        prompts = [json.loads(line)["prompt"] for line in path.read_text().splitlines()]
        token_counts = [len(tokens) for tokens in tokenizer(prompts, truncation=False)["input_ids"]]
        if max(token_counts) > spec["max_length"]:
            raise ValueError(f"Token limit exceeded: {path.name}")
        lengths[path.stem] = {"maximum_tokens": max(token_counts), "questions": len(prompts)}
    audit |= {"token_audit": lengths, "prior_worlds_excluded": len(excluded)}
    atomic_json(root / "data-audit.json", audit)
    frozen = freeze(root, candidates, comparison["identity"], audit)
    print(
        json.dumps({"frozen_v3": frozen["identity"], "budget": frozen["budget"], "audit": audit}),
        flush=True,
    )


if __name__ == "__main__":
    main()
