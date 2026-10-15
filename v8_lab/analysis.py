import json
from collections import defaultdict

import numpy as np
import torch

from decision_lab.data.validate import read_examples
from decision_lab.evaluation.metrics import paired_family_interval
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal
from v8_lab.data import TESTS
from v8_lab.protocol import verify


def predictions(root, name, split):
    rows = read_examples(root / "data" / f"{split}.jsonl")
    saved = [
        json.loads(line)
        for line in (root / "predictions" / name / f"{split}.jsonl").read_text().splitlines()
    ]
    if [
        (r.example_id, r.family_id, list(("TRUE", "FALSE", "UNKNOWN")).index(r.label.value))
        for r in rows
    ] != [(r["example_id"], r["family_id"], r["label"]) for r in saved]:
        raise ValueError("Predictions differ from frozen input identities/labels")
    logits = torch.tensor([r["logits"] for r in saved], dtype=torch.float64)
    if logits.shape != (len(rows), 3) or not torch.isfinite(logits).all():
        raise ValueError("Invalid raw logits")
    return rows, logits, torch.tensor([r["label"] for r in saved])


def complete_worlds(rows, logits, labels):
    grouped = defaultdict(list)
    for row, correct in zip(rows, (logits.argmax(-1) == labels).tolist(), strict=True):
        grouped[row.family_id].append(correct)
    return {family: int(all(values)) for family, values in grouped.items()}


def world_contrast(left, right, records, spec):
    if set(left) != set(right):
        raise ValueError("World contrasts are not paired")
    blocks = defaultdict(list)
    for family in sorted(left):
        record = records[family]
        blocks[(len(record["entities"]), record["pair_variant"], record["sampling_block"])].append(
            left[family] - right[family]
        )
    if any(len(values) != 3 for values in blocks.values()):
        raise ValueError("A label-rotation block is incomplete")
    strata = defaultdict(list)
    for (size, edit, _), values in blocks.items():
        strata[size, edit].append(float(np.mean(values)))
    rng = np.random.default_rng(spec["bootstrap_seed"])
    estimates = np.zeros(spec["bootstrap_repetitions"])
    for values in strata.values():
        values = np.asarray(values)
        estimates += values[rng.integers(len(values), size=(len(estimates), len(values)))].sum(1)
    estimates /= len(blocks)
    tail = (1 - spec["bootstrap_confidence"]) / 2
    return {
        "estimate": float(np.mean([left[f] - right[f] for f in left])),
        "interval": np.quantile(estimates, [tail, 1 - tail]).tolist(),
        "families": len(left),
        "blocks": len(blocks),
        "unit": "Three-world label-rotation block, stratified by entity count and edit type",
    }


def entity_checks(rows, logits, labels, records):
    grouped, output = (
        defaultdict(dict),
        defaultdict(lambda: {"pairs": 0, "expected_response": 0, "both_correct": 0}),
    )
    decisions = logits.argmax(-1).tolist()
    for index, row in enumerate(rows):
        variant, slot = row.example_id.rsplit("-", 2)[-2:]
        grouped[row.family_id][variant, int(slot)] = index
    for family, group in grouped.items():
        record = records[family]
        for slot in range(len(record["entities"])):
            base = group["base", slot]
            for variant in record["variants"]:
                if variant == "base":
                    continue
                changed = group[variant, slot]
                key = variant
                if variant in ("support_removal", "support_reversal", "irrelevant_edit"):
                    key += "/target" if slot == record["target_slot"] else "/unaffected"
                expected = decisions[base]
                if variant == "negated_query":
                    expected = [1, 0, 2][expected]
                elif variant == "support_removal" and slot == record["target_slot"]:
                    expected = 2
                elif variant == "support_reversal" and slot == record["target_slot"]:
                    expected = [1, 0, 2][expected]
                output[key]["pairs"] += 1
                output[key]["expected_response"] += int(decisions[changed] == expected)
                output[key]["both_correct"] += int(
                    decisions[base] == labels[base] and decisions[changed] == labels[changed]
                )
                if (
                    variant in ("support_removal", "support_reversal")
                    and slot == record["target_slot"]
                ):
                    subset = key + ("_unknown" if labels[base] == 2 else "_known")
                    output[subset]["pairs"] += 1
                    output[subset]["expected_response"] += int(decisions[changed] == expected)
                    output[subset]["both_correct"] += int(
                        decisions[base] == labels[base] and decisions[changed] == labels[changed]
                    )

            for other in range(slot + 1, len(record["entities"])):
                compared = group["base", other]
                if labels[base] != labels[compared]:
                    output["different_entity_different_label"]["pairs"] += 1
                    output["different_entity_different_label"]["expected_response"] += int(
                        decisions[base] != decisions[compared]
                    )
                    output["different_entity_different_label"]["both_correct"] += int(
                        decisions[base] == labels[base] and decisions[compared] == labels[compared]
                    )
    return dict(output)


def analyse(root):
    protocol = verify(root)
    spec = protocol["settings"]
    selection = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    if selection["protocol_identity"] != protocol["identity"]:
        raise ValueError("Selection protocol differs")
    records = {
        r["family_id"]: r
        for r in [
            json.loads(line) for line in (root / "data/families.jsonl").read_text().splitlines()
        ]
    }
    result = {
        "protocol_identity": protocol["identity"],
        "selection_identity": selection["identity"],
        "comparisons": {},
        "entity_checks": {},
        "complete_worlds_by_size": {},
    }
    for split in TESTS:
        rows, treatment, labels = predictions(root, "treatment", split)
        other_rows, control, other_labels = predictions(root, "control", split)
        if [r.example_id for r in rows] != [r.example_id for r in other_rows] or not torch.equal(
            labels, other_labels
        ):
            raise ValueError("Final test inputs are not paired")
        result["comparisons"][split] = {}
        for name, mask in (
            ("overall", torch.ones(len(labels), dtype=torch.bool)),
            ("unknown", labels == 2),
        ):
            result["comparisons"][split][name] = paired_family_interval(
                treatment[mask],
                control[mask],
                labels[mask],
                [r.family_id for r, keep in zip(rows, mask.tolist(), strict=True) if keep],
                spec["bootstrap_repetitions"],
                spec["bootstrap_confidence"],
                spec["bootstrap_seed"],
                "accuracy",
            )
        if split == "test_paired":
            left, right = (
                complete_worlds(rows, treatment, labels),
                complete_worlds(rows, control, labels),
            )
            result["primary"] = world_contrast(left, right, records, spec)
            for arm, complete in (("control", right), ("treatment", left)):
                result["complete_worlds_by_size"][arm] = {
                    str(size): {
                        "correct": sum(
                            value
                            for family, value in complete.items()
                            if len(records[family]["entities"]) == size
                        ),
                        "families": sum(
                            len(records[family]["entities"]) == size for family in complete
                        ),
                    }
                    for size in (3, 4, 5)
                }
        if split == "test_multi_transformations":
            for arm, logits in (("control", control), ("treatment", treatment)):
                result["entity_checks"][arm] = entity_checks(rows, logits, labels, records)
    retention = {}
    for split in ("test_original", "test_counterfactual"):
        rows, parent, labels = predictions(root, "parent", split)
        _, treatment, _ = predictions(root, "treatment", split)
        mask = labels == 2
        retention[split] = paired_family_interval(
            treatment[mask],
            parent[mask],
            labels[mask],
            [r.family_id for r, keep in zip(rows, mask.tolist(), strict=True) if keep],
            spec["bootstrap_repetitions"],
            spec["bootstrap_confidence"],
            spec["bootstrap_seed"],
            "accuracy",
        )
    rows, parent, labels = predictions(root, "parent", "test_multi")
    _, treatment, _ = predictions(root, "treatment", "test_multi")
    base = world_contrast(
        complete_worlds(rows, treatment, labels),
        complete_worlds(rows, parent, labels),
        records,
        spec,
    )
    pair_rows, pair_parent, pair_labels = predictions(root, "parent", "test_paired")
    result["arms_against_parent"] = {}
    for arm in ("control", "treatment"):
        _, logits, _ = predictions(root, arm, "test_paired")
        result["arms_against_parent"][arm] = world_contrast(
            complete_worlds(pair_rows, logits, pair_labels),
            complete_worlds(pair_rows, pair_parent, pair_labels),
            records,
            spec,
        )
    result["base_world_retention_against_parent"] = base
    result["retention_against_parent"] = retention
    result["engineering_gate"] = {
        "primary_gain_interval_excludes_zero": result["primary"]["interval"][0] > 0,
        "retention_unknown_margin": spec["noninferiority_margin"],
        "base_world_retention_within_margin": base["interval"][0]
        >= -spec["base_noninferiority_margin"],
        "both_parent_unknown_retention_intervals_within_margin": all(
            r["interval"][0] >= -spec["noninferiority_margin"] for r in retention.values()
        ),
    }
    result["prediction_inventory"] = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted((root / "predictions").rglob("*.jsonl"))
    }
    atomic_json(root / "analysis.json", result)
    return result
