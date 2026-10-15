import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def counts(root, arm, split):
    data_path = root / "data" / f"{split}.jsonl"
    prediction_path = root / "predictions" / arm / f"{split}.jsonl"
    inputs = [json.loads(line) for line in data_path.open()]
    predictions = [json.loads(line) for line in prediction_path.open()]
    if len(inputs) != len(predictions):
        raise ValueError("Raw input/prediction counts differ")
    families = defaultdict(list)
    unknown_correct, unknown_total = 0, 0
    for source, row in zip(inputs, predictions, strict=True):
        label = ("TRUE", "FALSE", "UNKNOWN").index(source["label"])
        if (source["example_id"], source["family_id"], label) != (
            row["example_id"],
            row["family_id"],
            row["label"],
        ):
            raise ValueError("Raw prediction identity or label differs")
        logits = row["logits"]
        if len(logits) != 3 or not all(math.isfinite(v) for v in logits):
            raise ValueError("Raw logits are invalid")
        correct = max(range(3), key=lambda i: logits[i]) == label
        families[source["family_id"]].append(correct)
        if label == 2:
            unknown_correct += correct
            unknown_total += 1
    return {
        "questions": len(inputs),
        "correct_questions": sum(sum(v) for v in families.values()),
        "families": len(families),
        "complete_families": sum(all(v) for v in families.values()),
        "unknown_correct": unknown_correct,
        "unknown_questions": unknown_total,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repetition", choices=("r1", "r2"))
    root = Path("runs/study-v7") / parser.parse_args().repetition
    destination = root / "independent-raw-counts.json"
    if destination.exists():
        raise FileExistsError(destination)
    analysis = json.loads((root / "analysis.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    splits = ("test_multi", "test_original", "test_counterfactual")
    result = {
        arm: {split: counts(root, arm, split) for split in splits}
        for arm in ("parent", "control", "treatment")
    }
    for arm in ("control", "treatment"):
        result[arm]["test_multi_transformations"] = counts(root, arm, "test_multi_transformations")
        expected_worlds = sum(
            v["correct"] for v in analysis["complete_worlds_by_size"][arm].values()
        )
        if result[arm]["test_multi"]["complete_families"] != expected_worlds:
            raise ValueError("Primary complete-world count differs")
        for split, record in result[arm].items():
            expected = evaluation["arms"][arm]["splits"][split]
            pairs = (
                (record["correct_questions"] / record["questions"], expected["raw"]["accuracy"]),
                (
                    record["complete_families"] / record["families"],
                    expected["all_questions_in_family_correct"],
                ),
            )
            if any(abs(a - b) > 1e-12 for a, b in pairs):
                raise ValueError("Reported raw correctness differs")
    gain = (
        result["treatment"]["test_multi"]["complete_families"]
        - result["control"]["test_multi"]["complete_families"]
    ) / 360
    if abs(gain - analysis["primary"]["estimate"]) > 1e-12:
        raise ValueError("Primary contrast differs")
    for split in splits[1:]:
        parent, treatment = result["parent"][split], result["treatment"][split]
        difference = treatment["unknown_correct"] / treatment["unknown_questions"] - (
            parent["unknown_correct"] / parent["unknown_questions"]
        )
        if (
            abs(difference - analysis["retention_against_parent"][split]["left_minus_right"])
            > 1e-12
        ):
            raise ValueError("Retention contrast differs")
    with destination.open("x") as output:
        json.dump({"valid": True, "counts": result}, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"repetition": root.name, "valid": True, "primary_gain": gain}))


if __name__ == "__main__":
    main()
