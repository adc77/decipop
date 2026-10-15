import json
from collections import Counter, defaultdict

import numpy as np
import torch

from decision_lab.evaluation.metrics import scores
from decision_lab.publication import atomic_json
from decision_lab.studies.v5.data import VARIANTS
from decision_lab.studies.v5.protocol import verify
from decision_lab.training.checkpoint import file_hash


def interval(values, spec):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(spec["bootstrap_seed"])
    draws = values[rng.integers(0, len(values), (spec["bootstrap_repetitions"], len(values)))].mean(
        1
    )
    tail = (1 - spec["bootstrap_confidence"]) / 2
    return {
        "estimate": float(values.mean()),
        "interval": np.quantile(draws, [tail, 1 - tail]).tolist(),
        "families": len(values),
        "confidence": spec["bootstrap_confidence"],
        "uncertainty": "parent-family sampling for fixed models; no seed replication",
    }


def pair_metrics(rows, predictions, family_records):
    positions = defaultdict(dict)
    for i, row in enumerate(rows):
        positions[row["family_id"]][row["variant"], row["slot"]] = i
    labels = [row["label"] for row in rows]
    result = {}
    for variant in VARIANTS:
        indices = [[lookup[variant, slot] for slot in range(3)] for lookup in positions.values()]
        result[variant] = {
            "families": len(indices),
            "complete_world_correct": sum(
                all(predictions[i] == labels[i] for i in group) for group in indices
            ),
            "constant_prediction_worlds": sum(
                len({predictions[i] for i in group}) == 1 for group in indices
            ),
        }
    same, correct, different = Counter(), Counter(), Counter()
    for family, lookup in positions.items():
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a, b = lookup["base", left], lookup["base", right]
            different["query_change"] += predictions[a] != predictions[b]
            correct["query_change"] += predictions[a] == labels[a] and predictions[b] == labels[b]
            same["query_change"] += 1
        for variant in ("renamed", "reordered", "distractor", "wording", "negated_query"):
            for slot in range(3):
                a, b = lookup["base", slot], lookup[variant, slot]
                expected = predictions[a]
                if variant == "negated_query":
                    expected = {"TRUE": "FALSE", "FALSE": "TRUE", "UNKNOWN": "UNKNOWN"}[expected]
                different[variant] += predictions[b] == expected
                correct[variant] += predictions[a] == labels[a] and predictions[b] == labels[b]
                same[variant] += 1
        for variant in ("remove_support", "reverse_support", "repair_unknown"):
            target = family_records[family][
                "unknown_slot" if variant == "repair_unknown" else "target_slot"
            ]
            for slot in range(3):
                key = variant + ("/target" if slot == target else "/unaffected")
                a, b = lookup["base", slot], lookup[variant, slot]
                different[key] += (
                    predictions[a] != predictions[b]
                    if slot == target
                    else predictions[a] == predictions[b]
                )
                correct[key] += predictions[a] == labels[a] and predictions[b] == labels[b]
                same[key] += 1
    return {
        "worlds": result,
        "pairs": {
            key: {
                "pairs": int(count),
                "expected_change_or_invariance": int(different[key]),
                "both_correct": int(correct[key]),
            }
            for key, count in same.items()
        },
    }


def summarise(rows, records, families, temperature, spec):
    if len(rows) != len(records):
        raise ValueError("Prediction row count mismatch")
    for row, record in zip(rows, records, strict=True):
        if record["example_id"] != row["example_id"] or record["label"] != row["label"]:
            raise ValueError("Prediction identity or label mismatch")
    logits = torch.tensor([r["logits"] for r in records], dtype=torch.float64)
    labels = torch.tensor([spec["class_labels"].index(r["label"]) for r in rows])
    if logits.shape != (len(rows), 3) or not torch.isfinite(logits).all():
        raise ValueError("Invalid prediction logits")
    decisions = [spec["class_labels"][i] for i in logits.argmax(-1).tolist()]
    results = {}
    for suite in spec["families"]:
        selected = [i for i, r in enumerate(rows) if r["suite"] == suite]
        local_rows = [rows[i] for i in selected]
        local_predictions = [decisions[i] for i in selected]
        results[suite] = pair_metrics(local_rows, local_predictions, families)
        results[suite]["metrics"] = {}
        for variant in VARIANTS:
            indices = [i for i in selected if rows[i]["variant"] == variant]
            metrics = {}
            for name, scale in (("raw", 1.0), ("saved_temperature", temperature)):
                metrics[name] = scores(
                    logits[indices],
                    labels[indices],
                    scale,
                    spec["ece_bins"],
                    spec["high_confidence_threshold"],
                )
                del metrics[name]["risk_coverage"]
            results[suite]["metrics"][variant] = metrics
    return results, decisions, logits


def analyse(root):
    protocol = verify(root)
    spec = protocol["settings"]
    rows = [json.loads(line) for line in (root / "data/diagnostic.jsonl").read_text().splitlines()]
    families = {
        r["family_id"]: r
        for r in [
            json.loads(line) for line in (root / "data/families.jsonl").read_text().splitlines()
        ]
        if r["partition"] == "diagnostic"
    }
    results, vectors, examples = {}, {}, {}
    preselected = {f"v5-diagnostic-{suite}-000000" for suite in spec["families"]}
    for model, candidate in protocol["models"].items():
        path = root / "predictions" / f"{model}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        summary, decisions, logits = summarise(
            rows, records, families, candidate["temperature"], spec
        )
        results[model] = {
            "temperature": candidate["temperature"],
            "suites": summary,
            "predictions_sha256": file_hash(path),
        }
        lookup = {r["example_id"]: i for i, r in enumerate(rows)}
        vectors[model] = []
        for family in families.values():
            if family["suite"] == "conjunction":
                ids = [lookup[f"{family['family_id']}-base-{slot}"] for slot in range(3)]
                vectors[model].append(int(all(decisions[i] == rows[i]["label"] for i in ids)))
        probabilities = (logits / candidate["temperature"]).softmax(-1)
        examples[model] = [
            {
                "example_id": r["example_id"],
                "oracle": r["label"],
                "prediction": decisions[i],
                "saved_temperature_confidence": float(probabilities[i].max()),
                "question": r["question"],
                "facts": r["facts"],
                "rules": r["rules"],
            }
            for i, r in enumerate(rows)
            if r["family_id"] in preselected
        ]
    output = {
        "protocol_identity": protocol["identity"],
        "models": results,
        "primary": interval(vectors["v4_mixed"], spec),
        "descriptive_paired_contrasts": {
            name: interval(np.asarray(vectors["v4_mixed"]) - np.asarray(vectors[name]), spec)
            for name in ("v3_parent", "v4_control")
        },
        "prediction_rows": len(rows) * len(results),
        "preselected_examples": examples,
        "notes": (
            "No training, model selection, temperature fitting, "
            "or multiplicity-adjusted transfer claims."
        ),
    }
    atomic_json(root / "analysis.json", output)
    return output
