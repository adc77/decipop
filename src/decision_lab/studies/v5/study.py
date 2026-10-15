import gc
import json
import time
from pathlib import Path

import torch

from decision_lab.publication import atomic_json
from decision_lab.release import load_release, verify_release
from decision_lab.studies.v3.protocol import inventory
from decision_lab.studies.v5.protocol import verify
from decision_lab.training.checkpoint import file_hash
from decision_lab.training.core import predict, seed_everything, synchronize


def run(root, phase, commit, config_directory=Path("configs")):
    protocol = verify(root, config_directory)
    spec = protocol["settings"]
    torch.set_num_threads(spec["cpu_threads"])
    seed_everything(spec["model_seed"], True)
    if phase not in ("pilot", "diagnostic"):
        raise ValueError("Invalid inference phase")
    rows = [
        json.loads(line) for line in (root / "data" / f"{phase}.jsonl").read_text().splitlines()
    ]
    summaries = {}
    for name in spec["models"]:
        path = root / "predictions" / f"{name}.jsonl"
        if phase == "diagnostic" and path.exists():
            raise FileExistsError("Refusing to overwrite recorded logits")
        directory = root / "models" / name
        reference = verify_release(directory, "cuda") if phase == "pilot" else None
        start = time.perf_counter()
        model, tokenizer, _ = load_release(directory, "cuda")
        loaded_seconds = time.perf_counter() - start
        encoded = tokenizer([r["prompt"] for r in rows], truncation=False, padding=False)[
            "input_ids"
        ]
        if any(len(ids) > spec["max_length"] or not ids for ids in encoded):
            raise ValueError("Input exceeds context limit")
        labels = [spec["class_labels"].index(r["label"]) for r in rows]
        synchronize("cuda")
        start = time.perf_counter()
        result = predict(
            model, encoded, labels, spec["evaluation_batch"], tokenizer.pad_token_id, "cuda"
        )
        synchronize("cuda")
        elapsed = time.perf_counter() - start
        if phase == "diagnostic":
            path.parent.mkdir(exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "family_id": row["family_id"],
                            "label": row["label"],
                            "logits": logits,
                        }
                    )
                    + "\n"
                    for row, logits in zip(rows, result["logits"].tolist(), strict=True)
                )
            )
        if inventory(directory) != protocol["models"][name]["files"]:
            raise ValueError("Inference changed a model file")
        summaries[name] = {
            "rows": len(rows),
            "inference_seconds": elapsed,
            "load_seconds": loaded_seconds,
            "rows_per_second": len(rows) / elapsed,
            "fixed_reference_verification": reference,
            "precision": "BF16 CUDA autocast for scoring; FP32 reference checks",
            "predictions_sha256": file_hash(path) if phase == "diagnostic" else None,
        }
        del model, result
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"model": name, "phase": phase, "timing": summaries[name]}), flush=True)
    record = {
        "protocol_identity": protocol["identity"],
        "models": summaries,
        "pilot_logits_retained": False,
        "final_predictions_inspected": phase != "pilot",
    }
    atomic_json(root / f"{phase}-execution.json", record)
    commit()
    return record
