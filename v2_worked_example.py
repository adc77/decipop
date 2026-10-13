import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json, release_inventory
from decision_lab.release import load_release
from decision_lab.training.checkpoint import file_hash
from decision_lab.v2.protocol import verify_comparison


def main():
    torch.set_num_threads(2)
    root = Path("runs/study-v2")
    comparison = verify_comparison(root)
    selection = json.loads((root / "worked-example-selection.json").read_text())
    retention = json.loads((root / "retention.json").read_text())
    receipts = json.loads((root / "huggingface-release.json").read_text())
    protocol = json.loads((root / "treatment/protocol.json").read_text())
    data = Path("data/v2/treatment/development.jsonl")
    if file_hash(data) != protocol["input_files"]["data/development.jsonl"]:
        raise ValueError("Worked example inputs differ from frozen development data")
    rows = [r for r in read_examples(data) if r.family_id == selection["family_id"]]
    labels = ["TRUE", "FALSE", "UNKNOWN"]
    rows.sort(key=lambda row: labels.index(row.label))
    if [r.label.value for r in rows] != labels or len({r.question for r in rows}) != 1:
        raise ValueError("Worked example must hold one question fixed across three worlds")
    result = {
        "comparison_identity": comparison["identity"],
        "selection": selection,
        "analysis": "descriptive development example; not an additional test estimate",
        "precision": "CPU FP32; aggregate study scoring uses GPU BF16 autocast",
        "oracle_labels": labels,
        "example_ids": [r.example_id for r in rows],
        "arms": {},
    }
    for arm in ("control", "treatment"):
        receipt = receipts[arm]
        if not receipt["remote_download_verification"]["verified"]:
            raise ValueError("Fresh private Hub verification is required")
        release = Path(receipt["verified_download_directory"])
        release_inventory(release)
        model, tokenizer, metadata = load_release(release)
        if metadata["class_labels"] != labels:
            raise ValueError("Release class ordering differs from the worked example")
        encoded = tokenizer([r.prompt for r in rows], padding=True, return_tensors="pt")
        if encoded.input_ids.shape[1] > metadata["max_length"]:
            raise ValueError("Worked example exceeds the token limit")
        temperature = json.loads((release / "calibration.json").read_text())["temperature"]
        stages = {}
        for stage in ("selected", "final"):
            weights = release / "model.safetensors"
            if stage == "final":
                weights = Path("artifacts/v2-final-weights") / arm / "model.safetensors"
                if file_hash(weights) != retention["final_weights"][arm]["weights_sha256"]:
                    raise ValueError("Final weights differ from the retained checkpoint")
                load_model(model, weights, strict=True)
            with torch.inference_mode():
                logits = model.class_logits(
                    model(encoded.input_ids, encoded.attention_mask)
                ).float()
            scale = temperature if stage == "selected" else 1.0
            stages[stage] = {
                "weights_sha256": file_hash(weights),
                "logits": logits.tolist(),
                "decisions": [labels[i] for i in logits.argmax(-1).tolist()],
                "probabilities": (logits / scale).softmax(-1).tolist(),
                "probabilities_calibrated": stage == "selected",
                "temperature": scale,
            }
        result["arms"][arm] = stages
        del model
        print(
            json.dumps(
                {
                    "worked_example_completed": arm,
                    "decisions": {stage: value["decisions"] for stage, value in stages.items()},
                }
            ),
            flush=True,
        )
    atomic_json(root / "worked-example-predictions.json", result)


if __name__ == "__main__":
    main()
