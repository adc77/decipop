import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import PreTrainedTokenizerFast

from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from v8_lab.data import exclusions, generate
from v8_lab.protocol import ROOT, budget, freeze, freeze_study, settings


def billing(command):
    result = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "billing",
            command,
            "--profile",
            "amandogra2016",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def prepare_parent(root, spec):
    source = Path(spec["parent_directory"])
    manifest = release_inventory(source)
    api = HfApi()
    if api.whoami()["name"] != spec["hf_namespace"]:
        raise ValueError("Hugging Face identity differs")
    info = api.model_info(spec["parent_repo"], revision=spec["parent_commit"], files_metadata=True)
    if not info.private or info.sha != spec["parent_commit"]:
        raise ValueError("Pinned private parent differs")
    pinned = Path(hf_hub_download(spec["parent_repo"], "artifact-manifest.json", revision=info.sha))
    weights = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    if (
        file_hash(pinned) != file_hash(source / "artifact-manifest.json")
        or weights.lfs.sha256 != manifest["files"]["model.safetensors"]
    ):
        raise ValueError("Pinned parent manifest/weight hashes differ")
    destination = root / "parent"
    destination.mkdir()
    for name in (*manifest["files"], "artifact-manifest.json"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / name, target)
    atomic_json(
        root / "parent-remote-verification.json",
        {
            "repo_id": spec["parent_repo"],
            "commit": info.sha,
            "private": True,
            "manifest_sha256": file_hash(pinned),
            "weights_sha256": weights.lfs.sha256,
        },
    )


def main():
    if ROOT.exists():
        raise FileExistsError("Preserve and inspect the existing preparation attempt")
    ROOT.mkdir(parents=True)
    summary, rates = billing("summary"), billing("rates")
    atomic_json(ROOT / "billing-before.json", summary)
    atomic_json(ROOT / "billing-rates.json", rates)
    remaining = 30 - float(summary["metered_cost"])
    if float(summary["billed_cost"]) != 0 or remaining < 9:
        raise ValueError("Billing does not support both repetitions and reserves")
    for key, maximum in (
        ("gpu_hour_cost_l4", 0.8),
        ("cpu_hour_cost", 0.0473),
        ("mem_gib_hour_cost", 0.008),
    ):
        if float(rates[key]) > maximum:
            raise ValueError("Configured costs understate current API rates")
    atomic_json(
        ROOT / "preflight.json",
        {
            "snapshot_utc": datetime.now(UTC).isoformat(),
            "conservative_original_budget_remaining_usd": remaining,
            "workspace_billed_usd": summary["billed_cost"],
            "joint_compute_allocation_usd": 8,
            "joint_storage_reserve_usd": 1,
            "pricing_url": "https://modal.com/pricing",
        },
    )
    completed = []
    for repetition in ("r1", "r2"):
        root, spec = ROOT / repetition, settings(repetition)
        root.mkdir()
        prepare_parent(root, spec)
        excluded, files = exclusions(tuple(completed))
        audit = generate(root / "data", spec, excluded, files)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(root / "parent")
        tokens = {}
        for path in sorted((root / "data").glob("*.jsonl")):
            if path.stem == "families":
                continue
            rows = read_examples(path)
            lengths = [
                len(ids)
                for ids in tokenizer([r.prompt for r in rows], truncation=False)["input_ids"]
            ]
            if not lengths or max(lengths) > spec["max_length"]:
                raise ValueError(f"Invalid token lengths: {path}")
            tokens[path.stem] = {
                "questions": len(rows),
                "maximum_tokens": max(lengths),
                "total_input_tokens": sum(lengths),
            }
        audit |= {
            "token_audit": tokens,
            "exclusion_files": files,
            "preselected_family": f"{spec['study']}-test_multi-multi-3-0000-0",
        }
        atomic_json(root / "data-audit.json", audit)
        protocol = freeze(root, repetition, audit)
        completed.append(root / "data/families.jsonl")
        print(
            json.dumps(
                {
                    "repetition": repetition,
                    "identity": protocol["identity"],
                    "audit": audit,
                    "budget": budget(spec),
                }
            ),
            flush=True,
        )
    print(json.dumps({"study_freeze": freeze_study()}), flush=True)


if __name__ == "__main__":
    main()
