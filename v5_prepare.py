import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerFast

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import inventory
from v5_lab.data import exclusions, generate, validate_dataset
from v5_lab.protocol import budget, freeze, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path("runs/study-v5")
    spec = settings()
    if (root / "protocol.json").exists():
        raise FileExistsError("Study is already frozen")
    if args.resume and (not (root / "models").exists() or not (root / "data").exists()):
        raise ValueError("Resume requires the retained models and generated data")
    if not args.resume and ((root / "models").exists() or (root / "data").exists()):
        raise FileExistsError("Preparation already started; preserve this attempt")
    before = json.loads((root / "billing-before.json").read_text())
    rates = json.loads((root / "billing-rates.json").read_text())
    remaining = 30 - float(before["metered_cost"])
    if float(before["billed_cost"]) != 0 or remaining < 2:
        raise ValueError("Billing snapshot does not support the allocation")
    for key, maximum in [
        ("gpu_hour_cost_l4", spec["gpu_hourly_usd"]),
        ("cpu_hour_cost", spec["cpu_core_hourly_usd"]),
        ("mem_gib_hour_cost", spec["memory_gib_hourly_usd"]),
    ]:
        if float(rates[key]) > maximum:
            raise ValueError("Configured rates understate current API rates")
    api, candidates = HfApi(), {}
    for name, config in spec["models"].items():
        source = Path(config["directory"])
        manifest = release_inventory(source)
        info = api.model_info(config["repo_id"], revision=config["commit"])
        if not info.private or info.sha != config["commit"]:
            raise ValueError("Pinned private model mismatch")
        destination = root / "models" / name
        destination.mkdir(parents=True, exist_ok=args.resume)
        for relative in [*manifest["files"], "artifact-manifest.json"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.resume:
                if file_hash(target) != file_hash(source / relative):
                    raise ValueError("Retained candidate differs on resume")
            else:
                os.link(source / relative, target)
        metadata = json.loads((destination / "config.json").read_text())
        if metadata["architecture"] != "head" or metadata["class_labels"] != spec["class_labels"]:
            raise ValueError("Model architecture/class order mismatch")
        candidates[name] = {
            "repo_id": config["repo_id"],
            "commit": info.sha,
            "files": inventory(destination),
            "temperature": json.loads((destination / "calibration.json").read_text())[
                "temperature"
            ],
        }
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root / "models/v4_mixed")
    excluded, exclusion_files = exclusions()
    audit = (
        json.loads((root / "data/audit.json").read_text())
        if args.resume
        else generate(root / "data", spec, excluded, exclusion_files, tokenizer)
    )
    if audit["exclusion_files"] != exclusion_files or audit["excluded_worlds"] != len(excluded):
        raise ValueError("Prior-world exclusions changed")
    audit["independent_serialized_data_audit"] = validate_dataset(root / "data", excluded)
    lengths = {}
    for name in spec["models"]:
        current = PreTrainedTokenizerFast.from_pretrained(root / "models" / name)
        for partition in ("pilot", "diagnostic"):
            rows = [
                json.loads(line)
                for line in (root / "data" / f"{partition}.jsonl").read_text().splitlines()
            ]
            counts = [
                len(ids)
                for ids in current([r["prompt"] for r in rows], truncation=False)["input_ids"]
            ]
            if max(counts) > spec["max_length"]:
                raise ValueError("Input exceeds a retained model context limit")
            lengths[f"{name}/{partition}"] = {"maximum_tokens": max(counts), "rows": len(rows)}
    audit["token_audit"] = lengths
    atomic_json(root / "data-audit.json", audit)
    atomic_json(
        root / "preflight.json",
        {
            "snapshot_utc": datetime.now(UTC).isoformat(),
            "conservative_original_budget_remaining_usd": remaining,
            "workspace_billed_usd": before["billed_cost"],
            "budget": budget(spec),
            "rates_source": "Modal billing API",
            "pricing_url": "https://modal.com/pricing",
        },
    )
    protocol = freeze(root, candidates, audit)
    print(
        json.dumps(
            {
                "identity": protocol["identity"],
                "partitions": audit["partitions"],
                "excluded_worlds": audit["excluded_worlds"],
                "token_audit": lengths,
                "budget": protocol["budget"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
