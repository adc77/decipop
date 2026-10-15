import json
from collections import defaultdict
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash

ROOT = Path("runs/study-v6")


def main():
    destination = ROOT / "descriptive-edit-slices.json"
    if destination.exists():
        raise FileExistsError(destination)
    analysis = json.loads((ROOT / "analysis.json").read_text())
    records = {
        row["family_id"]: row
        for row in map(json.loads, (ROOT / "data/families.jsonl").read_text().splitlines())
    }
    labels = ("TRUE", "FALSE", "UNKNOWN")
    result = {
        "status": "Post hoc descriptive slicing of existing logits; no new inference or selection",
        "analysis_sha256": file_hash(ROOT / "analysis.json"),
        "arms": {},
    }
    for arm in ("control", "treatment"):
        path = ROOT / "predictions" / arm / "test_multi_transformations.jsonl"
        grouped = defaultdict(dict)
        for row in map(json.loads, path.read_text().splitlines()):
            variant, slot = row["example_id"].rsplit("-", 2)[-2:]
            grouped[row["family_id"]][variant, int(slot)] = row
        counts = defaultdict(
            lambda: {
                "pairs": 0,
                "base_correct": 0,
                "edited_correct": 0,
                "both_correct": 0,
                "expected_response": 0,
            }
        )
        for family, rows in grouped.items():
            slot = records[family]["target_slot"]
            base = rows["base", slot]
            before = max(range(3), key=lambda i: base["logits"][i])
            for variant in ("support_edit", "support_edit_inverse"):
                changed = rows[variant, slot]
                after = max(range(3), key=lambda i: changed["logits"][i])
                key = f"{variant}/{labels[base['label']]}->{labels[changed['label']]}"
                expected = ([1, 2, 0] if variant.endswith("inverse") else [2, 0, 1])[before]
                count = counts[key]
                count["pairs"] += 1
                count["base_correct"] += int(before == base["label"])
                count["edited_correct"] += int(after == changed["label"])
                count["both_correct"] += int(before == base["label"] and after == changed["label"])
                count["expected_response"] += int(after == expected)
        for variant in ("support_edit", "support_edit_inverse"):
            for field in ("pairs", "both_correct", "expected_response"):
                total = sum(v[field] for k, v in counts.items() if k.startswith(variant + "/"))
                if total != analysis["entity_checks"][arm][variant + "/target"][field]:
                    raise ValueError("Descriptive edit counts differ from frozen aggregates")
        result["arms"][arm] = dict(sorted(counts.items()))
    atomic_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
