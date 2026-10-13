import json
from pathlib import Path

import torch

from decision_lab.publication import atomic_json


def main():
    torch.set_num_threads(2)
    root = Path("runs/study-v4")
    selection = json.loads((root / "worked-example-selection.json").read_text())
    family = selection["family_id"]
    frozen = json.loads((root / "selection-freeze.json").read_text())
    rows, predictions = {}, {}
    for split in ("test_original", "test_counterfactual"):
        rows[split] = sorted(
            (
                json.loads(line)
                for line in (root / "training" / f"{split}.jsonl").read_text().splitlines()
                if json.loads(line)["family_id"] == family
            ),
            key=lambda r: r["label"],
        )
        predictions[split] = {}
        for arm in ("control", "treatment"):
            predictions[split][arm] = {
                r["example_id"]: r
                for r in (
                    json.loads(line)
                    for line in (root / "predictions" / arm / f"{split}.jsonl")
                    .read_text()
                    .splitlines()
                )
                if r["family_id"] == family
            }
    lines = [
        "# V4: one family, two forms of UNKNOWN",
        "",
        f"Family `{family}` was chosen by lexicographic order before final-test inference. "
        "This is an illustrative example, not an additional estimate of test accuracy.",
        "",
    ]
    retained = {"selection": selection, "suites": {}}
    labels = ["TRUE", "FALSE", "UNKNOWN"]
    for split, examples in rows.items():
        lines += [f"## {split}", ""]
        suite = []
        for row in examples:
            lines += [
                f"Statement: **{row['question']}**; oracle: **{row['label']}**.",
                "",
                "Facts:",
                "",
                *[f"- {f}" for f in row["facts"]],
                "",
                "Rules:",
                "",
                *[f"- {r}" for r in row["rules"]],
                "",
                "| Arm | Prediction | Raw confidence | Calibrated confidence "
                "| TRUE/FALSE/UNKNOWN logits |",
                "| --- | --- | ---: | ---: | --- |",
            ]
            record = {"example": row, "arms": {}}
            for arm in ("control", "treatment"):
                scored = predictions[split][arm][row["example_id"]]
                logits = torch.tensor(scored["logits"], dtype=torch.float64)
                name = frozen["choices"][arm]["calibrated_nll"]
                temperature = frozen["calibrations"][name]["temperature"]
                decision = labels[logits.argmax().item()]
                raw = logits.softmax(-1).max().item()
                calibrated = (logits / temperature).softmax(-1).max().item()
                lines.append(
                    f"| {arm} | {decision} | {raw:.2%} | {calibrated:.2%} | "
                    + ", ".join(f"{v:.4f}" for v in scored["logits"])
                    + " |"
                )
                record["arms"][arm] = {
                    "decision": decision,
                    "logits": scored["logits"],
                    "raw_confidence": raw,
                    "calibrated_confidence": calibrated,
                    "temperature": temperature,
                }
            lines.append("")
            suite.append(record)
        retained["suites"][split] = suite
    lines += [
        "The original-query suite keeps facts and rules fixed and changes the statement. "
        "Its UNKNOWN statement concerns an unsupported entity. The changed-premise suite "
        "keeps the statement fixed and changes facts. Its UNKNOWN world removes a necessary "
        "derivation by flipping an antecedent. Both are the same semantic label, but they "
        "require transfer across different input changes.",
        "",
        "Training uses three-class cross-entropy: minus the log probability of the oracle "
        "label. Both arms update the same backbone and head from the same starting weights. "
        "The treatment's data requires handling both types of input change. Calibration "
        "divides logits by a positive temperature; it adjusts confidence without changing "
        "the winning label. These are observed outputs, not proof of internal reasoning.",
        "",
        "Read [V4_RESULTS.md](V4_RESULTS.md) for the complete held-out results and limitations.",
    ]
    atomic_json(root / "worked-example.json", retained)
    Path("reports/V4_LEARNING.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
