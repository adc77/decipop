import argparse
import json
from pathlib import Path

import torch

from decision_lab.evaluation.metrics import paired_family_interval
from decision_lab.publication import atomic_json
from decision_lab.studies.v3.protocol import check_seal
from decision_lab.training.checkpoint import file_hash

ROOT = Path("runs/study-v4")


def read_predictions(arm, split):
    return [
        json.loads(line)
        for line in (ROOT / "predictions" / arm / f"{split}.jsonl").read_text().splitlines()
    ]


def diagnostics():
    result = json.loads((ROOT / "diagnostics.json").read_text())
    lines = [
        "# V4 controlled diagnostics",
        "",
        "Development-only families; no additional training and no final-test estimates.",
        "",
        f"Frozen diagnostic protocol: `{result['protocol_identity']}`.",
        "",
        "The pinned V3 treatment-final head scored 200 new families, eight variants each. "
        "An exact oracle checked every label; all prior V1–V3 canonical worlds were excluded.",
        "",
        "| Variant | Accuracy | Raw high-confidence errors | Calibrated high-confidence errors |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role, scores in result["roles"].items():
        raw, calibrated = scores["raw"], scores["calibrated"]
        lines.append(
            f"| {role} | {raw['accuracy']:.2%} | {raw['high_confidence_wrong']} / "
            f"{raw['questions']} | {calibrated['high_confidence_wrong']} / {raw['questions']} |"
        )
    lines += [
        "",
        "High-confidence threshold: 90%; calibration uses the historical V3 temperature. "
        "FALSE queries in this table retain the original statement and flip its selector.",
        "",
        "| Pair | Oracle label changes | Prediction changes | Both variants correct |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pair, record in result["pairs"].items():
        lines.append(
            f"| {pair} | {record['oracle_label_changes']} / {record['families']} | "
            f"{record['prediction_changed_rate']:.2%} | {record['both_correct_rate']:.2%} |"
        )
    lines += [
        "",
        "The query-entity pairs keep facts, rules, predicate, and polarity fixed. "
        "Renaming is a consistent permutation of existing names. The distractor adds an "
        "opposite query-predicate fact about another person while preserving the query's "
        "oracle label. The premise intervention flips a necessary fact without changing "
        "fact count.",
        "",
        "These observations measure sensitivity to the controlled conditions. They do not "
        "establish how the network represents entities internally. Diagnostic families are "
        "excluded from later selection, calibration, and tests.",
        "",
        "Raw inputs and logits: local `runs/study-v4/diagnostic` and "
        "`runs/study-v4/predictions/parent/diagnostic.jsonl`. "
        "Protocol, audit, and metrics are retained beside them.",
    ]
    Path("reports/V4_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")


def statistics():
    protocol = check_seal(json.loads((ROOT / "training-protocol.json").read_text()))
    if file_hash(Path(__file__)) != protocol["audit"]["analysis_script_sha256"]:
        raise ValueError("Analysis script changed after protocol freeze")
    spec = protocol["settings"]
    output = {"protocol_identity": protocol["identity"], "comparisons": {}}
    for split in (
        "test_original",
        "test_counterfactual",
        "test_depth",
        "test_wording",
        "test_single_branch",
    ):
        control, treatment = (
            read_predictions("control", split),
            read_predictions("treatment", split),
        )
        if [(r["example_id"], r["family_id"], r["label"]) for r in control] != [
            (r["example_id"], r["family_id"], r["label"]) for r in treatment
        ]:
            raise ValueError("Predictions are not paired on identical inputs")
        labels = torch.tensor([r["label"] for r in control])
        left = torch.tensor([r["logits"] for r in treatment], dtype=torch.float64)
        right = torch.tensor([r["logits"] for r in control], dtype=torch.float64)
        families = [r["family_id"] for r in control]
        comparisons = {}
        for condition, mask in (
            ("overall", torch.ones(len(labels), dtype=torch.bool)),
            ("unknown", labels == 2),
        ):
            comparisons[condition] = paired_family_interval(
                left[mask],
                right[mask],
                labels[mask],
                [f for f, keep in zip(families, mask.tolist(), strict=True) if keep],
                spec["bootstrap_repetitions"],
                spec["bootstrap_confidence"],
                spec["bootstrap_seed"],
                "accuracy",
            )
        output["comparisons"][split] = comparisons
    primary = output["comparisons"]["test_original"]["unknown"]
    retention = output["comparisons"]["test_counterfactual"]["unknown"]
    output["engineering_gate"] = {
        "primary_gain_interval_excludes_zero": primary["interval"][0] > 0,
        "counterfactual_noninferiority_margin": spec["noninferiority_margin"],
        "counterfactual_lower_bound_above_negative_margin": retention["interval"][0]
        > -spec["noninferiority_margin"],
        "scope": (
            "paired test-family sampling; one additional-training seed; descriptive transfer suites"
        ),
    }
    atomic_json(ROOT / "statistics.json", output)
    return output


def results():
    output = json.loads((ROOT / "evaluation.json").read_text())
    frozen = check_seal(json.loads((ROOT / "selection-freeze.json").read_text()))
    protocol = check_seal(json.loads((ROOT / "training-protocol.json").read_text()))
    stats = statistics()
    primary = stats["comparisons"]["test_original"]["unknown"]
    retention = stats["comparisons"]["test_counterfactual"]["unknown"]
    lines = [
        "# V4 results: mixed UNKNOWN training",
        "",
        f"Frozen protocol: `{protocol['identity']}`.",
        "",
        "Both arms started from identical retained V3 treatment-final weights and used "
        "fresh optimizers, matched exposure, and a single additional-training seed. "
        "Control used changed-premise examples; treatment replaced approximately half "
        "with original-style query/entity-varied triples. "
        "This evaluates the complete training-mixture intervention.",
        "",
        "| Arm | Calibrated-NLL selected update | Presentations at selection | Final temperature |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ("control", "treatment"):
        name = frozen["choices"][arm]["calibrated_nll"]
        candidate = frozen["candidates"][name]
        lines.append(
            f"| {arm} | {candidate['step']} | {candidate['presentations']} | "
            f"{frozen['calibrations'][name]['temperature']:.5f} |"
        )
    lines += [
        "",
        "| Arm | Suite | Accuracy | UNKNOWN accuracy | Complete-family accuracy "
        "| Raw NLL | Calibrated NLL | Raw ECE | Calibrated ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, result in output["arms"].items():
        for split, score in result["splits"].items():
            lines.append(
                f"| {arm} | {split} | {score['raw']['accuracy']:.2%} | "
                f"{score['unknown_accuracy']:.2%} | "
                f"{score['all_questions_in_family_correct']:.2%} | "
                f"{score['raw']['nll']:.4f} | {score['calibrated']['nll']:.4f} | "
                f"{score['raw']['ece']:.4f} | {score['calibrated']['ece']:.4f} |"
            )
    lines += [
        "",
        f"Primary unsupported-query UNKNOWN gain: **{primary['left_minus_right'] * 100:.2f} "
        f"percentage points**, paired family-bootstrap 95% interval "
        f"**[{primary['interval'][0] * 100:.2f}, {primary['interval'][1] * 100:.2f}]**.",
        "",
        f"Changed-premise UNKNOWN gain: **{retention['left_minus_right'] * 100:.2f} "
        f"percentage points**, interval "
        f"**[{retention['interval'][0] * 100:.2f}, {retention['interval'][1] * 100:.2f}]**.",
        "",
        f"Primary positive-gain gate passed: "
        f"**{stats['engineering_gate']['primary_gain_interval_excludes_zero']}**. "
        f"Changed-premise -2 percentage-point noninferiority gate passed: "
        f"**{stats['engineering_gate']['counterfactual_lower_bound_above_negative_margin']}**.",
        "",
        "These are predeclared engineering gates, without a formal power calculation. "
        "Intervals concern family sampling for fixed models, not training-seed uncertainty. "
        "A collapsed empirical interval cannot establish perfect population generalization.",
        "",
        "## Every selection candidate",
        "",
        "Temperatures are fitted on selection-fit and scored on separate selection families. "
        "Final calibration is distinct; test scoring starts only after freezing selection.",
        "",
        "| Candidate | Selection accuracy | Raw NLL | Calibrated NLL | Fit temperature |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, score in frozen["selection_scores"].items():
        lines.append(
            f"| {name} | {score['accuracy']:.2%} | {score['raw_nll']:.4f} | "
            f"{score['calibrated_nll']:.4f} | "
            f"{frozen['selection_fit_temperatures'][name]['temperature']:.5f} |"
        )
    lines += [
        "",
        "![V4 accuracy by suite](v4-figures/accuracy.svg)",
        "",
        "## Limits",
        "",
        "Exact-world exclusion does not exclude shared abstract motifs. The original-style "
        "UNKNOWN generator, query positions, mechanically rendered English, and restricted "
        "rule language can still admit shortcuts. Improved accuracy is evidence about the "
        "evaluated distributions, not a proven entity-binding mechanism or a general "
        "decision engine. Architecture, loss, wording augmentation, and rule curriculum "
        "were not changed between arms. Transfer comparisons are descriptive; they are "
        "not multiplicity-adjusted. Temperature scaling cannot change labels, and its "
        "confidence metrics may worsen under distribution shift.",
        "",
        "See [controlled diagnostics](V4_DIAGNOSTICS.md), "
        "[frozen protocol](../docs/V4_PROTOCOL.md), "
        "and local `runs/study-v4` for all inputs, predictions, and operational records.",
    ]
    Path("reports/V4_RESULTS.md").write_text("\n".join(lines) + "\n")
    plot(output)


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    suites = (
        "test_original",
        "test_counterfactual",
        "test_depth",
        "test_wording",
        "test_single_branch",
    )
    names = (
        "Original queries",
        "Changed premises",
        "Deeper proofs",
        "New wording",
        "Single branch",
    )
    positions = np.arange(len(suites))
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for index, (arm, color) in enumerate((("control", "#526D82"), ("treatment", "#237A57"))):
        values = [output["arms"][arm]["splits"][s]["raw"]["accuracy"] * 100 for s in suites]
        bars = ax.bar(positions + (index - 0.5) * 0.36, values, 0.36, label=arm, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    ax.set(
        xticks=positions,
        xticklabels=names,
        ylim=(0, 112),
        ylabel="Accuracy (%)",
        title="V4: fresh tests after matched additional fine-tuning",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    directory = Path("reports/v4-figures")
    directory.mkdir(exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(directory / f"accuracy.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("diagnostics", "results"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    diagnostics() if args.stage == "diagnostics" else results()


if __name__ == "__main__":
    main()
