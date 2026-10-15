import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import PreTrainedTokenizerFast

from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from v6_lab.data import exclusions, generate
from v6_lab.protocol import budget, freeze, settings


def main():
    root = Path("runs/study-v6")
    if (root / "parent").exists() or (root / "data").exists():
        raise FileExistsError("Preparation already started; preserve and audit this attempt")
    spec = settings()
    summary = json.loads((root / "billing-before.json").read_text())
    rates = json.loads((root / "billing-rates.json").read_text())
    remaining = 30 - float(summary["metered_cost"])
    if (
        float(summary["billed_cost"]) != 0
        or remaining < spec["compute_allocation_usd"] + spec["storage_reserve_usd"]
    ):
        raise ValueError("Current billing does not support the allocation")
    for key, maximum in (
        ("gpu_hour_cost_l4", spec["gpu_hourly_usd"]),
        ("cpu_hour_cost", spec["cpu_core_hourly_usd"]),
        ("mem_gib_hour_cost", spec["memory_gib_hourly_usd"]),
    ):
        if float(rates[key]) > maximum:
            raise ValueError("Configured rates understate API rates")
    source = Path(spec["parent_directory"])
    manifest = release_inventory(source)
    api = HfApi()
    info = api.model_info(spec["parent_repo"], revision=spec["parent_commit"], files_metadata=True)
    if not info.private or info.sha != spec["parent_commit"]:
        raise ValueError("Pinned private parent revision differs")
    pinned = Path(
        hf_hub_download(
            spec["parent_repo"], "artifact-manifest.json", revision=spec["parent_commit"]
        )
    )
    if file_hash(pinned) != file_hash(source / "artifact-manifest.json"):
        raise ValueError("Pinned remote manifest differs")
    weights = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    if weights.lfs.sha256 != manifest["files"]["model.safetensors"]:
        raise ValueError("Pinned remote weight checksum differs")
    destination = root / "parent"
    destination.mkdir(parents=True)
    for name in (*manifest["files"], "artifact-manifest.json"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / name, target)
    atomic_json(
        root / "parent-remote-verification.json",
        {
            "repo_id": spec["parent_repo"],
            "commit": info.sha,
            "private": info.private,
            "manifest_sha256": file_hash(pinned),
            "weights_sha256": weights.lfs.sha256,
        },
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(destination)
    excluded, files = exclusions()
    audit = generate(root / "data", spec, excluded, files)
    tokens = {}
    for path in sorted((root / "data").glob("*.jsonl")):
        if path.name == "families.jsonl":
            continue
        rows = read_examples(path)
        lengths = [
            len(ids) for ids in tokenizer([r.prompt for r in rows], truncation=False)["input_ids"]
        ]
        if not lengths or max(lengths) > spec["max_length"]:
            raise ValueError(f"Empty or overlength partition: {path}")
        tokens[path.stem] = {
            "questions": len(rows),
            "maximum_tokens": max(lengths),
            "total_input_tokens": sum(lengths),
        }
    audit |= {
        "token_audit": tokens,
        "exclusion_files": files,
        "preselected_family": "v6-test_multi-multi-3-0000-0",
    }
    atomic_json(root / "data-audit.json", audit)
    atomic_json(
        root / "preflight.json",
        {
            "snapshot_utc": datetime.now(UTC).isoformat(),
            "conservative_original_budget_remaining_usd": remaining,
            "workspace_billed_usd": summary["billed_cost"],
            "budget": budget(spec),
            "rates_source": "Modal billing API",
            "pricing_url": "https://modal.com/pricing",
        },
    )
    record = freeze(root, audit)
    print(
        json.dumps({"identity": record["identity"], "audit": audit, "budget": record["budget"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
