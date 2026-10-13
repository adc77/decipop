import importlib.metadata
import json
import shutil
from pathlib import Path

import torch

from decision_lab.data.render import build_prompt
from decision_lab.tokenization import answer_ids
from decision_lab.training.checkpoint import file_hash


def export_release(
    root,
    architecture,
    model,
    tokenizer,
    rows,
    pilot,
    scratch,
    protocol,
    calibration,
    device="cuda",
    license_directory=Path("/root"),
):
    directory = root / "releases" / architecture
    directory.mkdir(parents=True, exist_ok=False)
    selected = Path(calibration["weights"])
    shutil.copyfile(selected, directory / "model.safetensors")
    tokenizer.save_pretrained(directory)
    metadata = {
        "architecture": architecture,
        "class_labels": pilot.class_labels,
        "answer_ids": answer_ids(tokenizer, pilot) if architecture == "token" else None,
        "max_length": pilot.max_length,
        "scratch": scratch.model_dump() if architecture == "scratch" else None,
        "backbone": model.backbone.config.to_dict() if architecture != "scratch" else None,
        "protocol_identity": protocol["identity"],
        "requires_custom_loader": True,
    }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    amendment = root / "operational-invocation-amendment.json"
    if amendment.exists():
        shutil.copyfile(amendment, directory / amendment.name)
    (directory / "prompt-template.txt").write_text(
        build_prompt(["{facts}"], ["{rules}"], "{question}") + "\n"
    )
    fixed = [row.prompt for row in rows[:8]]
    encoded = tokenizer(fixed, padding=True, truncation=False, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        logits = model.class_logits(model(encoded.input_ids, encoded.attention_mask)).float().cpu()
    reference = {
        "prompts": fixed,
        "logits": logits.tolist(),
        "decisions": [pilot.class_labels[index] for index in logits.argmax(-1).tolist()],
        "precision": "FP32; separate from BF16 study scoring",
        "logit_atol": 0.001,
        "logit_rtol": 0.001,
    }
    (directory / "verification.json").write_text(json.dumps(reference, indent=2) + "\n")
    source = Path(__file__).parent.parent
    files = [
        "__init__.py",
        "config.py",
        "model_config.py",
        "release.py",
        "data/__init__.py",
        "data/render.py",
        "data/schema.py",
        "training/__init__.py",
        "training/checkpoint.py",
        *[str(path.relative_to(source)) for path in sorted((source / "models").glob("*.py"))],
    ]
    for name in files:
        destination = directory / "code/decision_lab" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    requirements = ["torch", "transformers", "tokenizers", "safetensors", "pydantic", "PyYAML"]
    (directory / "requirements.txt").write_text(
        "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in requirements) + "\n"
    )
    (directory / "infer.py").write_text(
        "import argparse\nimport json\nimport sys\nfrom pathlib import Path\n"
        "def main():\n"
        "    sys.path.insert(0, str(Path(__file__).parent / 'code'))\n"
        "    from decision_lab.release import predict_release\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--input', type=Path, required=True)\n"
        "    args = parser.parse_args()\n"
        "    prompts = json.loads(args.input.read_text())\n"
        "    print(json.dumps(predict_release(Path(__file__).parent, prompts)))\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    shutil.copyfile(license_directory / "LICENSE", directory / "LICENSE")
    shutil.copyfile(license_directory / "NOTICE", directory / "NOTICE")
    base = (
        f"Base: HuggingFaceTB/SmolLM2-360M, revision `{pilot.revision}`. "
        if architecture != "scratch"
        else "Base: random initialization; no pretrained weights. "
    )
    (directory / "README.md").write_text(
        "---\nlicense: apache-2.0\ntags:\n- research\n- text-classification\n- custom_code\n---\n\n"
        f"# Decision Model Lab: {architecture}\n\n"
        "Three-class decisions over supplied facts and explicit rules: TRUE, FALSE, UNKNOWN. "
        "UNKNOWN means neither a proposition nor its explicit negation is derivable.\n\n"
        f"{base}"
        "Code and any pretrained base weights use Apache 2.0. "
        "The project generated the dataset.\n\n"
        "This model requires the included loader. "
        "It is not a stock causal-LM or sequence-classification pipeline. "
        "Use Python 3.12, install requirements.txt, and run `python infer.py --input prompts.json` "
        "with a JSON list of prompts matching prompt-template.txt.\n\n"
        "Weights retain FP32 precision. The study used BF16 autocast; "
        "fixed artifact checks use FP32. "
        "See config.json, protocol.json, calibration.json, and evaluation.json "
        "for exact settings.\n\n"
        "This is a single-seed synthetic reasoning experiment. "
        "It does not establish general decision ability or external-domain calibration. "
        "Paraphrases were mechanically checked against the formal source "
        "without independent human review.\n\n"
        "Source implementation remains uncommitted locally; "
        "protocol.json records its training-code hash. "
        "The included inference source is versioned with this artifact.\n"
        "Any operational-invocation-amendment.json records recovery-call allowances; "
        "it does not change the frozen model hyperparameters or training exposure.\n"
    )
    return directory


def finalize_release(directory: Path, protocol_identity: str, evaluation: dict):
    (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    card = [
        "",
        "## Recorded held-out results",
        "",
        "| Split | Accuracy | Calibrated NLL | Calibrated ECE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, row in evaluation["splits"].items():
        card.append(
            f"| {split} | {row['raw']['accuracy']:.2%} | "
            f"{row['calibrated']['nll']:.4f} | {row['calibrated']['ece']:.4f} |"
        )
    if "unrestricted_accuracy" in evaluation["splits"]["test_id"]:
        row = evaluation["splits"]["test_id"]
        card += [
            "",
            f"ID unrestricted vocabulary accuracy: {row['unrestricted_accuracy']:.2%}. "
            f"Invalid unrestricted answers: {row['unrestricted_invalid_answer_rate']:.2%}. "
            "The main accuracy table restricts answers to A/B/C.",
        ]
    metadata = json.loads((directory / "config.json").read_text())
    if (
        metadata["architecture"] == "scratch"
        and evaluation["splits"]["test_id"]["raw"]["accuracy"] <= 0.4
    ):
        card += [
            "",
            "This scratch run remained near the balanced three-class chance level. "
            "It is a trained checkpoint retained as a negative research result, "
            "not evidence of useful decision accuracy.",
        ]
    (directory / "README.md").write_text(
        (directory / "README.md").read_text() + "\n".join(card) + "\n"
    )
    inventory = {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    (directory / "artifact-manifest.json").write_text(
        json.dumps({"protocol_identity": protocol_identity, "files": inventory}, indent=2) + "\n"
    )
