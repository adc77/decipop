import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_lab.evaluation.metrics import paired_family_interval
from decision_lab.publication import atomic_json
from decision_lab.v2.data import ARMS, TESTS


def write_report(root: Path, output: Path):
    comparison = json.loads((root / "comparison.json").read_text())
    result = json.loads((root / "evaluation.json").read_text())
    predictions = {}
    for arm in ARMS:
        predictions[arm] = [
            json.loads(line)
            for line in (root / "predictions" / arm / "test_counterfactual.jsonl")
            .read_text()
            .splitlines()
        ]
    left = predictions["treatment"]
    right = predictions["control"]
    if [r["example_id"] for r in left] != [r["example_id"] for r in right]:
        raise ValueError("Paired prediction IDs disagree")
    if [r["label"] for r in left] != [r["label"] for r in right]:
        raise ValueError("Paired prediction labels disagree")
    statistics = {}
    for name, indices, metric in (
        (
            "primary_unknown_accuracy",
            [i for i, r in enumerate(left) if r["label"] == 2],
            "accuracy",
        ),
        ("counterfactual_raw_nll", list(range(len(left))), "nll"),
    ):
        statistics[name] = paired_family_interval(
            [left[i]["logits"] for i in indices],
            [right[i]["logits"] for i in indices],
            [left[i]["label"] for i in indices],
            [left[i]["family_id"] for i in indices],
            2000,
            0.95,
            17,
            metric,
        )
    atomic_json(root / "statistics.json", statistics)
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = output.parent / "v2-figures"
    figures.mkdir(exist_ok=True)
    for metric, ylabel, name in (
        ("unknown_accuracy", "UNKNOWN accuracy (%)", "unknown-accuracy"),
        (
            "all_questions_in_family_correct",
            "All questions in family correct (%)",
            "family-correctness",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        for index, arm in enumerate(ARMS):
            values = [result["arms"][arm]["splits"][split][metric] * 100 for split in TESTS]
            ax.bar(
                [i + (index - 0.5) * 0.36 for i in range(len(TESTS))], values, width=0.36, label=arm
            )
        ax.set_xticks(range(len(TESTS)), [s.removeprefix("test_").replace("_", " ") for s in TESTS])
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylabel)
        ax.legend()
        for extension in ("svg", "png"):
            fig.savefig(figures / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    primary = statistics["primary_unknown_accuracy"]
    lines = [
        "# V2: changed-premise training results",
        "",
        "Primary UNKNOWN accuracy difference, treatment minus control: "
        f"**{primary['left_minus_right']:.2%}**; 95% paired parent-family bootstrap interval "
        f"[{primary['interval'][0]:.2%}, {primary['interval'][1]:.2%}].",
        "",
        "This is one training seed. The interval measures test-family sampling uncertainty, "
        "not seed stability. "
        "The primary test holds the same question fixed across TRUE, FALSE, and UNKNOWN worlds. "
        "All-three-world correctness distinguishes premise-sensitive decisions "
        "from a constant UNKNOWN answer.",
        "",
        "## Held-out accuracy",
        "",
        "| Suite | Control | Treatment | UNKNOWN control | UNKNOWN treatment |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in TESTS:
        a, b = [result["arms"][arm]["splits"][split] for arm in ARMS]
        lines.append(
            f"| {split} | {a['raw']['accuracy']:.2%} | {b['raw']['accuracy']:.2%} | "
            f"{a['unknown_accuracy']:.2%} | {b['unknown_accuracy']:.2%} |"
        )
    lines += [
        "",
        "![UNKNOWN accuracy](v2-figures/unknown-accuracy.svg)",
        "",
        "![All-family correctness](v2-figures/family-correctness.svg)",
        "",
        "## Family correctness and confidence",
        "",
        "| Suite | Model | All-family correct | Cal NLL | Cal ECE | Wrong at ≥90% confidence |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in TESTS:
        for arm in ARMS:
            m = result["arms"][arm]["splits"][split]
            lines.append(
                f"| {split} | {arm} | {m['all_questions_in_family_correct']:.2%} | "
                f"{m['calibrated']['nll']:.4f} | {m['calibrated']['ece']:.4f} | "
                f"{m['calibrated']['high_confidence_wrong']} |"
            )
    lines += ["", "## Selection and exposure", ""]
    for arm in ARMS:
        state = json.loads((root / arm / "head-state.json").read_text())
        calibration = json.loads((root / "calibration-freeze.json").read_text())["arms"][arm]
        lines.append(
            f"- {arm}: {state['examples_seen']:,} presentations; "
            f"{state['input_tokens']:,} input tokens; {state['step']} updates; "
            f"selected `{Path(state['selected']).name}`; "
            f"temperature {calibration['temperature']:.4f}."
        )
    nll = statistics["counterfactual_raw_nll"]
    lines += [
        "",
        "Counterfactual raw NLL difference, treatment minus control: "
        f"{nll['left_minus_right']:.4f}; "
        f"95% interval [{nll['interval'][0]:.4f}, {nll['interval'][1]:.4f}]. Lower NLL is better.",
        "",
        "## Scope and limitations",
        "",
        "Both arms use the same parent training theories, model initialization, "
        "hyperparameters, exposure, "
        "and development/calibration inputs. The control uses original-style queries "
        "on the new shared "
        "theories, not the exact v1 training set. The treatment changes "
        "both premise validity and query selection. "
        "It is a data-construction intervention, "
        "not an isolated effect of pretraining or one token. "
        "A direct numerical v1/v2 accuracy comparison would mix different test distributions.",
        "",
        "Training contains both premise flips and deletions; "
        "counterfactual test UNKNOWNs use flips only. "
        "The single-branch suite independently tests original-style rules "
        "with a critical fact flip. "
        "It contains two questions per family and 50% UNKNOWN, "
        "so its family correctness and class balance differ from the three-world suites. "
        "Parent worlds are isolated across partitions and checked against v1. "
        "No human paraphrase review or additional training seed was performed.",
        "",
        f"Frozen comparison: `{comparison['identity']}`. "
        "Exact protocol, statistics, raw predictions, ledger, and release checks "
        "remain in `runs/study-v2`. "
        "See [V2_PROTOCOL.md](../docs/V2_PROTOCOL.md) and "
        "[V2_RELEASES.md](V2_RELEASES.md) for execution and artifact provenance.",
        "",
    ]
    output.write_text("\n".join(lines))
    return statistics
