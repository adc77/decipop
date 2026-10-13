import json
import math
from itertools import product
from pathlib import Path

import torch

from decision_lab.data.render import render_atom
from decision_lab.data.schema import Label
from decision_lab.data.solve import solve
from decision_lab.data.validate import read_examples


def oracle_proof(row):
    resolution = solve(row.theory)
    actual, depth = resolution.answer(row.query)
    if actual != row.label or depth != row.proof_depth:
        raise ValueError("Worked example does not agree with the oracle")
    if actual == Label.UNKNOWN:
        return ["Neither the query nor its explicit negation occurs in the derived closure."]
    target = row.query if actual == Label.TRUE else row.query.opposite()
    lines = []
    visited = set()

    def visit(atom):
        if atom in visited:
            return
        if resolution.depths[atom] == 0:
            lines.append("Given: " + render_atom(atom, {}) + ".")
            visited.add(atom)
            return
        for rule in row.theory.rules:
            variables = sorted(
                {
                    term
                    for condition in rule.conditions
                    for term in condition.arguments
                    if term.startswith("?")
                }
            )
            for entities in product(row.theory.entities, repeat=len(variables)):
                binding = dict(zip(variables, entities, strict=True))
                conditions = [condition.ground(binding) for condition in rule.conditions]
                if rule.conclusion.ground(binding) != atom:
                    continue
                if not all(condition in resolution.depths for condition in conditions):
                    continue
                if (
                    1 + max(resolution.depths[condition] for condition in conditions)
                    != resolution.depths[atom]
                ):
                    continue
                for condition in conditions:
                    visit(condition)
                lines.append("Derive: " + render_atom(atom, {}) + ".")
                visited.add(atom)
                return
        raise RuntimeError("Unable to reconstruct a shortest oracle proof")

    visit(target)
    return lines


def write_learning(study: Path, releases: Path, dataset: Path, output: Path):
    models = ("pretrained", "token", "head", "scratch")
    worked = {
        name: [
            json.loads(line)
            for line in (study / "predictions" / name / "worked_development.jsonl")
            .read_text()
            .splitlines()
        ]
        for name in models
    }
    development = read_examples(dataset / "development.jsonl")[:8]
    first = development[0]
    class_names = ["TRUE", "FALSE", "UNKNOWN"]
    lines = [
        "# What we learned about decision models",
        "",
        "## One numerical decision",
        "",
        "```text",
        first.prompt,
        "```",
        "",
        f"Oracle answer: {first.label.value}; shortest proof depth: {first.proof_depth}.",
        "",
        *["- " + line for line in oracle_proof(first)],
        "",
        "| Model | TRUE logit | FALSE logit | UNKNOWN logit | Prediction | Cal confidence |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for name in models:
        value = worked[name][0]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{number:.4f}" for number in value["logits"])
            + f" | {class_names[value['prediction']]} | {max(value['probabilities']):.4f} |"
        )
    logits = torch.tensor(worked["head"][0]["logits"], dtype=torch.float64)
    probabilities = logits.softmax(-1)
    target = class_names.index(first.label.value)
    loss = -math.log(probabilities[target].item())
    lines += [
        "",
        f"For the head, raw softmax probabilities are {probabilities.tolist()}. "
        f"Restricted cross-entropy is −log P({first.label.value}) = {loss:.6f}. "
        "Temperature scaling divides the three logits by a positive fitted number; "
        "it changes confidence without changing the top class.",
        "",
        "The token model trains against the full vocabulary. Its three displayed decision "
        "probabilities are renormalized over A/B/C. Its training loss additionally includes "
        "the mass outside those tokens; outside mass and invalid unrestricted answers are "
        "recorded separately in the evaluation.",
        "",
        "## Attention and output shapes",
        "",
    ]
    token = json.loads((releases / "token/config.json").read_text())
    scratch = json.loads((releases / "scratch/config.json").read_text())["scratch"]
    hidden, vocab = token["backbone"]["hidden_size"], token["backbone"]["vocab_size"]
    lines += [
        f"With B prompts and padded length L, the pretrained backbone produces [B,L,{hidden}]. "
        "Causal attention allows each position to read its prefix. We gather the last nonpadding "
        f"position, yielding [B,{hidden}]. The token projection produces [B,{vocab}], "
        "then selects A/B/C. The decision head projects directly to [B,3]. "
        "Both approaches update the full backbone; this is full fine-tuning rather than LoRA.",
        "",
        f"The scratch encoder uses {scratch['layers']} layers, {scratch['heads']} attention heads "
        f"and width {scratch['hidden_size']}. Each nonpadding position can attend to all supplied "
        "positions. The first [CLS] representation is projected to three logits. "
        "This changes attention, model size, tokenizer, and initialization together; "
        "the comparison cannot isolate pretraining alone.",
        "",
        "The scratch tokenizer sees training prompts only and can encode unseen text through "
        "its byte alphabet. Learning this bounded task from random weights does not teach "
        "general conversational language understanding.",
        "",
        "## Checkpoint recovery",
        "",
        "Recovery stores model weights, Adam moments, scheduler state, example order/cursor, "
        "and Python/Torch CPU/CUDA RNG states. A checkpoint is complete only after its file "
        "checksums and protocol identity are written. The GPU gate verified a bitwise-identical "
        "next update after saving and reloading each architecture.",
        "",
        "## A deeper held-out case",
        "",
    ]
    depth_rows = read_examples(dataset / "test_depth.jsonl")
    predictions = {
        name: [
            json.loads(line)
            for line in (study / "predictions" / name / "test_depth.jsonl").read_text().splitlines()
        ]
        for name in models
    }
    index = next(
        (
            index
            for index, row in enumerate(depth_rows)
            if row.label != Label.UNKNOWN
            and predictions["token"][index]["prediction"] != class_names.index(row.label.value)
        ),
        0,
    )
    row = depth_rows[index]
    lines += [
        "```text",
        row.prompt,
        "```",
        "",
        f"Oracle: {row.label.value}; shortest proof depth: {row.proof_depth}.",
        "",
        *["- " + line for line in oracle_proof(row)],
        "",
    ]
    for name in models:
        value = predictions[name][index]
        lines.append(
            f"- {name}: {class_names[value['prediction']]}, "
            f"calibrated confidence {max(value['probabilities']):.2%}."
        )
    lines += [
        "",
        "This proof comes from the symbolic oracle. It is not a claim about the "
        "model's internal reasoning. An incorrect answer is compatible with exploiting "
        "shallow patterns; an individual correct answer cannot establish rule following.",
        "",
        "## UNKNOWN versus abstention",
        "",
        "UNKNOWN is a legitimate answer when neither side is derivable. Abstention means "
        "deferring a low-confidence prediction of any class. Confidence thresholds are "
        "chosen on calibration examples and then applied unchanged to held-out data. "
        "Their achieved coverage and error rates can shift on deeper or reworded inputs.",
    ]
    output.write_text("\n".join(lines) + "\n")
    return output
