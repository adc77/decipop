import json
from pathlib import Path

import torch

from decision_lab.data.validate import read_examples
from decision_lab.publication import atomic_json
from decision_lab.studies.v3.protocol import check_seal, verify
from decision_lab.training.checkpoint import file_hash


def main():
    root = Path("runs/study-v3")
    spec = verify(root)["settings"]
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    torch.set_num_threads(spec["cpu_threads"])
    labels = spec["class_labels"]
    rows = read_examples(root / "data/selection.jsonl")
    families = {}
    for row in rows:
        families.setdefault(row.family_id, []).append(row)
    matched = next(
        name for name in sorted(families) if len({row.query for row in families[name]}) == 1
    )
    original = next(
        row
        for row in sorted(rows, key=lambda row: row.example_id)
        if len({member.query for member in families[row.family_id]}) > 1 and row.label == "UNKNOWN"
    )
    chosen = sorted(families[matched], key=lambda row: labels.index(row.label.value)) + [original]
    records = {}
    for name in ("treatment-early", "treatment-final"):
        path = root / "predictions" / name / "selection.jsonl"
        if file_hash(path) != frozen["prediction_files"][name]["selection"]:
            raise ValueError("Worked example predictions differ from the frozen records")
        records[name] = {
            row["example_id"]: row
            for row in [json.loads(line) for line in path.read_text().splitlines()]
        }
    examples = []
    for row in chosen:
        result = {
            "example_id": row.example_id,
            "oracle_label": row.label.value,
            "facts": row.facts,
            "rules": row.rules,
            "question": row.question,
            "models": {},
        }
        for name, mapping in records.items():
            logits = torch.tensor(mapping[row.example_id]["logits"], dtype=torch.float64)
            temperature = frozen["final_calibrations"][name]["temperature"]
            result["models"][name] = {
                "logits": logits.tolist(),
                "decision": labels[int(logits.argmax())],
                "raw_probabilities": logits.softmax(-1).tolist(),
                "calibrated_probabilities": (logits / temperature).softmax(-1).tolist(),
                "temperature": temperature,
            }
        examples.append(result)
    output = {
        "selection_identity": frozen["identity"],
        "selection_rule": "first sorted matched selection family, plus first sorted "
        "original-style selection UNKNOWN; post hoc descriptive examples, not test estimates",
        "precision": "recorded GPU BF16 logits, FP64 probability calculation",
        "examples": examples,
    }
    atomic_json(root / "worked-example.json", output)
    lines = [
        "# V3: what selection and calibration actually changed",
        "",
        "These are descriptive examples from the selection partition, chosen after the study "
        "using the first sorted matched family and first sorted original-style UNKNOWN. "
        "They are not additional test estimates. Numbers use recorded GPU BF16 logits; "
        "calibrated probabilities use the separate final-calibration temperature.",
        "",
        "## One statement, three worlds",
        "",
        f"Statement: **{examples[0]['question']}**",
        "",
        "Original facts:",
        "",
    ]
    lines.extend(f"- {fact}" for fact in examples[0]["facts"])
    lines += ["", "Rules:", ""]
    lines.extend(f"- {rule}" for rule in examples[0]["rules"])
    lines += [
        "",
        "| World | Oracle | Early decision | Final decision | Final raw confidence "
        "| Final calibrated confidence |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for example in examples[:3]:
        early = example["models"]["treatment-early"]
        final = example["models"]["treatment-final"]
        lines.append(
            f"| {example['oracle_label']} | {example['oracle_label']} | {early['decision']} | "
            f"{final['decision']} | {max(final['raw_probabilities']):.4%} | "
            f"{max(final['calibrated_probabilities']):.2%} |"
        )
    before = set(examples[0]["facts"])
    for example in examples[1:3]:
        after = set(example["facts"])
        lines += [
            "",
            f"{example['oracle_label']} world: replace **{next(iter(before - after))}** "
            f"with **{next(iter(after - before))}**.",
        ]
    lines += [
        "",
        "UNKNOWN means neither the statement nor its explicit negation can be derived. "
        "It is a distinct label. The early head gives the same answer across these interventions; "
        "the final head's decisions follow this family's oracle.",
        "",
        "## Calibration cannot repair a wrong decision",
        "",
        f"Statement: **{original.question}**",
        "",
        "Facts:",
        "",
    ]
    lines.extend(f"- {fact}" for fact in original.facts)
    lines += ["", "Rules:", ""]
    lines.extend(f"- {rule}" for rule in original.rules)
    final = examples[3]["models"]["treatment-final"]
    temperature = final["temperature"]
    lines += [
        "",
        f"The oracle says **UNKNOWN**. The final model says **{final['decision']}** "
        f"with raw confidence **{max(final['raw_probabilities']):.4%}**. Temperature scaling "
        f"reduces that confidence to **{max(final['calibrated_probabilities']):.2%}**, "
        "but leaves the wrong decision unchanged.",
        "",
        f"The temperature is {temperature:.5f}. Compute probabilities as "
        "`softmax(logits / temperature)`; the largest logit remains the largest. "
        "This example has a proof about a different person from the statement. "
        "An entity-binding failure is a hypothesis to investigate, not a mechanism established "
        "by one example.",
        "",
        "## Why the old checkpoint selection failed",
        "",
        "An almost uniform early head pays about `log(3) = 1.099` NLL per question. "
        "The final head makes many more correct decisions, but confident mistakes on "
        "unsupported original-style queries can push its average raw NLL above that value. "
        "Fitting a temperature on selection-fit and evaluating it on separate selection families "
        "makes the final checkpoint competitive without changing its decisions. "
        "The checkpoint choice produces the accuracy gain; calibration adjusts its confidence.",
        "",
        "Read [V3_RESULTS.md](V3_RESULTS.md) for all fresh-test results, including wording "
        "and single-branch failures and the confidence degradation on the counterfactual suite.",
        "",
    ]
    Path("reports/V3_LEARNING.md").write_text("\n".join(lines))
    print(json.dumps({"worked_family": matched, "original_unknown": original.example_id}))


if __name__ == "__main__":
    main()
