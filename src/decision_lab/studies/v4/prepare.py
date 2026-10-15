import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerFast

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.studies.v4.data import exclusions, generate
from decision_lab.studies.v4.protocol import budget, freeze, settings
from decision_lab.training.checkpoint import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("diagnostic", "training"))
    args = parser.parse_args()
    root = Path("runs/study-v4")
    spec = settings()
    if args.phase == "diagnostic":
        if (root / "parent").exists():
            raise FileExistsError("Parent already prepared")
        source = Path(spec["parent_directory"])
        manifest = release_inventory(source)
        api = HfApi()
        if api.whoami()["name"] != spec["hf_namespace"]:
            raise ValueError("HF account mismatch")
        info = api.model_info(spec["parent_repo"], revision=spec["parent_commit"])
        if not info.private or info.sha != spec["parent_commit"]:
            raise ValueError("Parent privacy/revision mismatch")
        parent = root / "parent"
        parent.mkdir(parents=True)
        for name in (
            "config.json",
            "model.safetensors",
            "calibration.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            if file_hash(source / name) != manifest["files"][name]:
                raise ValueError("Parent file hash mismatch")
            os.link(source / name, parent / name)
        summary = json.loads((root / "billing-before.json").read_text())
        remainder = 30 - float(summary["metered_cost"])
        if (
            float(summary["billed_cost"]) != 0
            or remainder < spec["compute_allocation_usd"] + spec["storage_reserve_usd"]
        ):
            raise ValueError("Current billing snapshot does not support the V4 allocation")
        rates = json.loads((root / "billing-rates.json").read_text())
        if (
            float(rates["gpu_hour_cost_l4"]) > spec["gpu_hourly_usd"]
            or float(rates["cpu_hour_cost"]) > spec["cpu_core_hourly_usd"]
            or float(rates["mem_gib_hour_cost"]) > spec["memory_gib_hourly_usd"]
        ):
            raise ValueError("Frozen rates understate current API rates")
        atomic_json(
            root / "preflight.json",
            {
                "snapshot_utc": datetime.now(UTC).isoformat(),
                "conservative_original_budget_remaining_usd": remainder,
                "workspace_billed_usd": summary["billed_cost"],
                "rates_source": "Modal billing API",
                "published_rates_url": "https://modal.com/pricing",
                "budget": budget(spec),
                "parent_repo": spec["parent_repo"],
                "parent_commit": info.sha,
            },
        )
    else:
        diagnostic = json.loads((root / "diagnostics.json").read_text())
        pilot = json.loads((root / "pilot.json").read_text())
        projected = pilot["seconds_per_update"] * spec["chunk_steps"] + 120
        if projected >= spec["training_timeout_seconds"] - 30:
            raise ValueError(
                f"Pilot projects {projected:.1f}s per chunk; revise scope before freeze"
            )
        atomic_json(
            root / "training-preflight.json",
            {
                "diagnostic_protocol_identity": diagnostic["protocol_identity"],
                "pilot_sha256": file_hash(root / "pilot.json"),
                "projected_chunk_seconds": projected,
                "diagnostic_interpretation": (
                    "Controlled diagnostics motivate the preregistered mixed-data comparison"
                ),
                "heldout_tests_inspected": False,
            },
        )
    excluded, files = exclusions(include_diagnostics=args.phase == "training")
    audit = generate(
        root / args.phase, spec, excluded, files, diagnostic=args.phase == "diagnostic"
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root / "parent")
    lengths = {}
    for path in sorted((root / args.phase).glob("*.jsonl")):
        if path.name == "families.jsonl":
            continue
        prompts = [json.loads(line)["prompt"] for line in path.read_text().splitlines()]
        counts = [len(ids) for ids in tokenizer(prompts, truncation=False)["input_ids"]]
        if max(counts) > spec["max_length"]:
            raise ValueError(f"Token limit exceeded: {path}")
        lengths[path.stem] = {"maximum_tokens": max(counts), "questions": len(prompts)}
    audit["token_audit"] = lengths
    if args.phase == "training":
        audit["analysis_script_sha256"] = file_hash(Path("src/decision_lab/studies/v4/report.py"))
    atomic_json(root / f"{args.phase}-audit.json", audit)
    protocol = freeze(root, args.phase, audit)
    print(
        json.dumps({"frozen": protocol["identity"], "audit": audit, "budget": protocol["budget"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
