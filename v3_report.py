import json
from pathlib import Path

import numpy as np
import torch

from decision_lab.evaluation.metrics import paired_family_interval
from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v2.data import TESTS
from decision_lab.v3.protocol import check_seal, settings, verify


def read_predictions(root, name, split, expected):
    path = root / "predictions" / name / f"{split}.jsonl"
    if file_hash(path) != expected:
        raise ValueError("Analysis prediction checksum mismatch")
    return [json.loads(line) for line in path.read_text().splitlines()]


def family_interval(left, right, spec):
    family_left = {}
    family_right = {}
    for lrow, rrow in zip(left, right, strict=True):
        family_left.setdefault(lrow["family_id"], []).append(
            int(np.argmax(lrow["logits"])) == lrow["label"]
        )
        family_right.setdefault(rrow["family_id"], []).append(
            int(np.argmax(rrow["logits"])) == rrow["label"]
        )
    differences = np.array(
        [
            float(all(values)) - float(all(family_right[family]))
            for family, values in family_left.items()
        ]
    )
    rng = np.random.default_rng(spec["bootstrap_seed"])
    draws = np.array(
        [
            differences[rng.integers(0, len(differences), len(differences))].mean()
            for _ in range(spec["bootstrap_repetitions"])
        ]
    )
    tail = (1 - spec["bootstrap_confidence"]) / 2
    return {
        "left_minus_right": float(differences.mean()),
        "interval": np.quantile(draws, [tail, 1 - tail]).tolist(),
        "families": len(differences),
        "uncertainty": "test-family sampling only; one training seed",
    }


def compare(root, evaluation, frozen, left, right, split, spec):
    left_rows = read_predictions(
        root, left, split, evaluation["candidates"][left]["prediction_files"][split]
    )
    right_rows = read_predictions(
        root, right, split, evaluation["candidates"][right]["prediction_files"][split]
    )
    for lrow, rrow in zip(left_rows, right_rows, strict=True):
        if (lrow["example_id"], lrow["label"], lrow["family_id"]) != (
            rrow["example_id"],
            rrow["label"],
            rrow["family_id"],
        ):
            raise ValueError("Paired comparison rows differ")
    logits_left = torch.tensor([row["logits"] for row in left_rows], dtype=torch.float64)
    logits_right = torch.tensor([row["logits"] for row in right_rows], dtype=torch.float64)
    labels = torch.tensor([row["label"] for row in left_rows])
    families = [row["family_id"] for row in left_rows]
    unknown = [index for index, label in enumerate(labels) if label == 2]
    common = (spec["bootstrap_repetitions"], spec["bootstrap_confidence"], spec["bootstrap_seed"])
    temperature_left = frozen["final_calibrations"][left]["temperature"]
    temperature_right = frozen["final_calibrations"][right]["temperature"]
    return {
        "left": left,
        "right": right,
        "unknown_accuracy": paired_family_interval(
            logits_left[unknown],
            logits_right[unknown],
            labels[unknown],
            [families[index] for index in unknown],
            *common,
            "accuracy",
        ),
        "overall_accuracy": paired_family_interval(
            logits_left, logits_right, labels, families, *common, "accuracy"
        ),
        "calibrated_nll": paired_family_interval(
            logits_left / temperature_left,
            logits_right / temperature_right,
            labels,
            families,
            *common,
            "nll",
        ),
        "all_family_correct": family_interval(left_rows, right_rows, spec),
    }


def main():
    torch.set_num_threads(2)
    root = Path("runs/study-v3")
    protocol = verify(root)
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    evaluation = json.loads((root / "evaluation.json").read_text())
    if evaluation["selection_identity"] != frozen["identity"]:
        raise ValueError("Analysis selection identity differs")
    spec = settings()
    choices = frozen["choices"]
    comparisons = {}
    for arm in choices:
        for policy in ("calibrated_nll", "accuracy"):
            key = f"{arm}:{policy}-minus-raw_nll"
            comparisons[key] = {
                split: compare(
                    root,
                    evaluation,
                    frozen,
                    choices[arm][policy],
                    choices[arm]["raw_nll"],
                    split,
                    spec,
                )
                for split in TESTS
            }
    for policy in spec["policies"]:
        key = f"{policy}:treatment-minus-control"
        comparisons[key] = {
            split: compare(
                root,
                evaluation,
                frozen,
                choices["treatment"][policy],
                choices["control"][policy],
                split,
                spec,
            )
            for split in TESTS
        }
    primary = comparisons["treatment:calibrated_nll-minus-raw_nll"]["test_counterfactual"]
    atomic_json(
        root / "statistics.json",
        {
            "protocol_identity": protocol["identity"],
            "primary": primary,
            "comparisons": comparisons,
            "secondary_analyses": "descriptive; no multiplicity correction",
        },
    )
    lines = [
        "# V3 results: checkpoint selection and calibration",
        "",
        f"Frozen protocol: `{protocol['identity']}`.",
        "",
        "Four retained checkpoints, two per arm; one training seed; no additional training. "
        "Fresh families exclude every recorded V1/V2 canonical world. "
        "This compares the retained early/final pool, not the full trajectory.",
        "",
        "## Selection on fresh development families",
        "",
        "Temperatures in this table are fitted on selection-fit, "
        "then scored on distinct selection families.",
        "",
        "| Candidate | Step | Accuracy | Raw NLL | Selection-fit temperature "
        "| Calibrated selection NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in protocol["candidates"].items():
        score = frozen["selection_scores"][name]
        lines.append(
            f"| {name} | {candidate['step']} | {score['accuracy']:.2%} | "
            f"{score['raw_nll']:.4f} | "
            f"{frozen['selection_fit_temperatures'][name]['temperature']:.4f} | "
            f"{score['calibrated_nll']:.4f} |"
        )
    lines += [
        "",
        "| Arm | Raw-NLL choice | Calibrated-NLL choice | Accuracy choice |",
        "| --- | --- | --- | --- |",
    ]
    for arm, values in choices.items():
        lines.append(
            f"| {arm} | {values['raw_nll']} | {values['calibrated_nll']} | {values['accuracy']} |"
        )
    gain = primary["unknown_accuracy"]
    lines += [
        "",
        "## Primary result",
        "",
        "Treatment calibrated-NLL selection minus raw-NLL selection: "
        f"**{gain['left_minus_right'] * 100:.2f} percentage points** in UNKNOWN accuracy "
        "on 1,000 fresh counterfactual families; paired family-bootstrap 95% interval "
        f"**[{gain['interval'][0] * 100:.2f}, {gain['interval'][1] * 100:.2f}]** "
        "percentage points.",
        "",
        "This interval describes test-family sampling uncertainty for these fixed weights. "
        "It does not measure training-seed uncertainty, and a zero-width interval "
        "does not establish general equivalence.",
        "",
        "## Every retained candidate on fresh tests",
        "",
        "All temperatures below come from the separate final-calibration partition, "
        "after the choices were frozen. Unselected candidate results are diagnostic only.",
        "",
        "| Candidate | Suite | Accuracy | UNKNOWN accuracy | All-family correct "
        "| Raw NLL | Calibrated NLL | Raw ECE | Calibrated ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in evaluation["candidates"].items():
        for split, score in result["splits"].items():
            lines.append(
                f"| {name} | {split.removeprefix('test_')} | {score['raw']['accuracy']:.2%} | "
                f"{score['unknown_accuracy']:.2%} | "
                f"{score['all_questions_in_family_correct']:.2%} | "
                f"{score['raw']['nll']:.4f} | {score['calibrated']['nll']:.4f} | "
                f"{score['raw']['ece']:.4f} | {score['calibrated']['ece']:.4f} |"
            )
    lines += [
        "",
        "## Final calibration",
        "",
        "| Candidate | Temperature | At bound | Calibration raw NLL | Calibration fitted NLL |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for name, value in frozen["final_calibrations"].items():
        lines.append(
            f"| {name} | {value['temperature']:.5f} | {value['at_bound']} | "
            f"{value['raw_nll']:.4f} | {value['calibrated_nll']:.4f} |"
        )
    lines += [
        "",
        "## Paired policy comparisons",
        "",
        "Differences below are percentage points for accuracy and family correctness; "
        "NLL differences are nats/question. Negative NLL differences are improvements. "
        "Full family-bootstrap intervals for all suites are in v3-evidence/statistics.json.",
        "",
        "| Comparison | Counterfactual UNKNOWN gain | Overall gain | All-family gain "
        "| Calibrated NLL difference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in comparisons.items():
        score = value["test_counterfactual"]
        lines.append(
            f"| {name} | {score['unknown_accuracy']['left_minus_right'] * 100:.2f} | "
            f"{score['overall_accuracy']['left_minus_right'] * 100:.2f} | "
            f"{score['all_family_correct']['left_minus_right'] * 100:.2f} | "
            f"{score['calibrated_nll']['left_minus_right']:.4f} |"
        )
    treatment = evaluation["candidates"][choices["treatment"]["calibrated_nll"]]["splits"]
    counterfactual = treatment["test_counterfactual"]
    original = treatment["test_original"]
    wording = treatment["test_wording"]
    single = treatment["test_single_branch"]
    lines += [
        "",
        "## Interpretation",
        "",
        "The late treatment checkpoint's fresh counterfactual accuracy is "
        f"{counterfactual['raw']['accuracy']:.2%}; UNKNOWN accuracy is "
        f"{counterfactual['unknown_accuracy']:.2%}; complete-family accuracy is "
        f"{counterfactual['all_questions_in_family_correct']:.2%}. "
        "Both calibrated-NLL and accuracy selection recover that same retained checkpoint. "
        "The earlier raw-NLL selection discarded useful late decisions because unscaled "
        "overconfident errors dominated the NLL objective.",
        "",
        "Generalization remains limited: original-style UNKNOWN accuracy is "
        f"{original['unknown_accuracy']:.2%}, wording accuracy is "
        f"{wording['raw']['accuracy']:.2%}, and single-branch accuracy is "
        f"{single['raw']['accuracy']:.2%}. These failures remain after calibration, "
        "which leaves the predicted class unchanged.",
        "",
        "The temperature fitted on the original/counterfactual mixture improves confidence "
        "metrics on several difficult suites, but worsens the already excellent counterfactual "
        f"raw NLL from {counterfactual['raw']['nll']:.4f} to "
        f"{counterfactual['calibrated']['nll']:.4f}, and ECE from "
        f"{counterfactual['raw']['ece']:.4f} to {counterfactual['calibrated']['ece']:.4f}. "
        "A single mixture-fitted temperature is not a guarantee of calibration within every "
        "subdistribution. Confidence adjustments must be evaluated on the deployment distribution.",
        "",
        "![Treatment policy comparison](v3-figures/selection.svg)",
        "",
        "![Raw versus calibrated confidence](v3-figures/calibration.svg)",
    ]
    if gain["interval"][0] == gain["interval"][1]:
        lines += [
            "",
            "The primary empirical bootstrap interval collapses because every observed "
            "parent has the same paired UNKNOWN difference. Resampling this finite set "
            "cannot reveal unobserved failure types; it does not imply a guaranteed "
            "population improvement or perfect generalization.",
        ]
    lines += [
        "",
        "## What this establishes",
        "",
        "Checkpoint selection can change decisions; scalar temperature scaling cannot. "
        "Read the changes in accuracy separately from changes in confidence. "
        "Calibration lowers fitted NLL by construction, but fresh-test calibration metrics "
        "and transfer-suite metrics are the relevant checks.",
        "",
        "This is a follow-up motivated by observed V2 behavior, with one training seed, "
        "a restricted candidate pool, shared synthetic motifs, and mechanically rendered language. "
        "It is not a general decision-model benchmark or a replication across seeds. "
        "Exact-world exclusion does not exclude isomorphic structures. Confidence fitted "
        "on one mixture may fail under wording, proof-depth, or single-branch shift. "
        "Secondary comparisons are descriptive without multiplicity correction.",
        "",
        "See docs/V3_PROTOCOL.md for the predeclared design, v3-evidence for compact records, "
        "and local runs/study-v3 for every raw prediction and input.",
        "",
    ]
    destination = Path("reports/V3_RESULTS.md")
    destination.write_text("\n".join(lines))
    print(json.dumps({"primary": primary, "report": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
