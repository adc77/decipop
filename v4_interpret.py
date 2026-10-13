import json
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash


def main():
    root = Path("runs/study-v4")
    report = Path("reports/V4_RESULTS.md")
    text = report.read_text()
    marker = "## Post-test interpretation"
    if marker in text:
        raise ValueError("Post-test interpretation already recorded")
    output = {
        "scope": "Post-test descriptive label counts; no selection or protocol changes.",
        "label_order": ["TRUE", "FALSE", "UNKNOWN"],
        "matrix_axes": "rows: oracle label; columns: predicted label",
        "script_sha256": file_hash(Path(__file__)),
        "arms": {},
    }
    for arm in ("control", "treatment"):
        suites = {}
        for path in sorted((root / "predictions" / arm).glob("test_*.jsonl")):
            matrix = [[0] * 3 for _ in range(3)]
            for line in path.read_text().splitlines():
                row = json.loads(line)
                prediction = max(range(3), key=lambda index: row["logits"][index])
                matrix[row["label"]][prediction] += 1
            suites[path.stem] = {
                "prediction_sha256": file_hash(path),
                "confusion_matrix": matrix,
                "correct_by_label": [matrix[index][index] for index in range(3)],
                "questions_by_label": [sum(row) for row in matrix],
            }
        if len(suites) != 5:
            raise ValueError("Incomplete final-test predictions")
        output["arms"][arm] = suites
    atomic_json(root / "posthoc-error-breakdown.json", output)
    lines = [
        "",
        marker,
        "",
        "This interpretation and label breakdown were added after the frozen analysis. "
        "They do not change the selected models, primary endpoints, or engineering gates.",
        "",
        "The mixture closes much of the targeted gap: original-query UNKNOWN improves "
        "from 0/1,000 to 815/1,000. Original-query overall accuracy improves from "
        "2,000/3,000 to 2,812/3,000; 814/1,000 complete families are correct. "
        "Changed-premise UNKNOWN remains 1,000/1,000 for both arms. Treatment gets "
        "2,997/3,000 changed-premise questions and 997/1,000 complete families correct, "
        "versus 3,000/3,000 and 1,000/1,000 for control.",
        "",
        "The targeted improvement comes with transfer limits. Depth accuracy falls "
        "from 1,385/1,500 to 1,200/1,500, even though depth UNKNOWN accuracy improves "
        "from 463/500 to 474/500. Wording stays near chance: 634/1,500 for control "
        "and 638/1,500 for treatment; neither answers any complete wording family "
        "correctly. Single-branch accuracy falls from 526/1,000 to 516/1,000. "
        "Its label counts explain why improved UNKNOWN alone is insufficient:",
        "",
        "| Single-branch subset | Control correct | Mixed-data correct |",
        "| --- | ---: | ---: |",
        "| Supported TRUE or FALSE | 240 / 500 | 59 / 500 |",
        "| UNKNOWN | 286 / 500 | 457 / 500 |",
        "| All questions | 526 / 1,000 | 516 / 1,000 |",
        "",
        "The mixed model predicts UNKNOWN on 849/1,000 single-branch inputs, where "
        "only 500 oracle labels are UNKNOWN. This is observed behavior, not an "
        "identified internal mechanism. After calibration it still makes 745 "
        "wording errors and 460 single-branch errors with at least 90% confidence. "
        "Calibration on the training-style mixture does not make shifted inputs safe.",
        "",
        "A shortcut concern was logged before testing: the generator assigns a unique "
        "selector predicate to one target entity and excludes that predicate from noise. "
        "Matching the queried entity to this distinguished target could solve many "
        "original-query cases without general proof binding. The next justified study "
        "would remove that cue and give multiple independently answerable entities "
        "different labels within the same world. V4 does not establish a general "
        "decision engine, nor does it justify changing architecture before improving "
        "that diagnostic. No further GPU study has been submitted.",
        "",
        "Label counts and prediction hashes are retained in "
        "`runs/study-v4/posthoc-error-breakdown.json`; all raw logits remain available.",
    ]
    report.write_text(text + "\n".join(lines) + "\n")
    print(json.dumps({"post_test_interpretation_saved": True, "confusion_matrices": 10}))


if __name__ == "__main__":
    main()
