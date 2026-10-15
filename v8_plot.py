import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from v8_lab.protocol import ROOT


def main():
    paths = [Path("reports/V8_RESULTS.png"), Path("reports/V8_RESULTS.pdf")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite the retained V8 figure")
    analyses, choices, raw = {}, {}, {}
    for rep in ("r1", "r2"):
        root = ROOT / rep
        analyses[rep] = json.loads((root / "analysis.json").read_text())
        choices[rep] = json.loads((root / "selection-freeze.json").read_text())
        raw[rep] = json.loads((root / "independent-raw-counts.json").read_text())
        if not raw[rep]["valid"]:
            raise ValueError("Independent raw counts are unverified")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = {"control": "#55738E", "treatment": "#23836A"}
    for arm, offset in (("control", -0.18), ("treatment", 0.18)):
        values = [
            raw[rep]["counts"][arm]["test_paired"]["complete_families"] for rep in ("r1", "r2")
        ]
        bars = axes[0].bar(
            [offset, 1 + offset],
            [100 * v / 378 for v in values],
            width=0.32,
            color=colors[arm],
            label=arm.capitalize(),
        )
        for bar, count in zip(bars, values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{count}/378",
                ha="center",
                fontsize=10,
            )
    labels = []
    for rep, seed in (("r1", 1601), ("r2", 1801)):
        selected = choices[rep]
        steps = [
            selected["candidates"][selected["choices"][arm]["calibrated_nll"]]["step"]
            for arm in ("control", "treatment")
        ]
        labels.append(f"{rep.upper()} · seed {seed}\nSelected updates C={steps[0]}, T={steps[1]}")
    axes[0].set_xticks([0, 1], labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Complete paired worlds (%)")
    axes[0].set_title("Correct before and after edits", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    for i, rep in enumerate(("r1", "r2")):
        primary = analyses[rep]["primary"]
        value = 100 * primary["estimate"]
        low, high = (100 * v for v in primary["interval"])
        axes[1].errorbar(
            value,
            1 - i,
            xerr=[[value - low], [high - value]],
            fmt="o",
            color=colors["treatment"],
            capsize=5,
            markersize=7,
        )
        axes[1].text(
            value, 1 - i + 0.13, f"{value:.2f} [{low:.2f}, {high:.2f}]", ha="center", fontsize=10
        )
    axes[1].set_yticks([1, 0], ["R1", "R2"])
    axes[1].set_ylim(-0.4, 1.5)
    axes[1].axvline(0, color="#777777", linewidth=1)
    intervals = [a["primary"]["interval"] for a in analyses.values()]
    axes[1].set_xlim(
        min(0, *(100 * v[0] for v in intervals)) - 5, max(0, *(100 * v[1] for v in intervals)) + 5
    )
    axes[1].set_xlabel("Treatment − control (percentage points)")
    axes[1].set_title("Gain and 95% world-sampling intervals", loc="left", fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y" if axis is axes[0] else "x", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle("V8: paired consistency on two fresh seeds", fontsize=15, x=0.06, ha="left")
    fig.text(
        0.06,
        0.04,
        "Selected pipelines, 378 fresh paired worlds per repetition. Intervals resample "
        "rotation blocks for fixed models.\nThese intervals do not measure training-seed "
        "uncertainty; no pooled interval is reported.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.23, top=0.83, wspace=0.32)
    for path in paths:
        fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)
    atomic_json(
        ROOT / "figure.json",
        {
            "files": {str(path): file_hash(path) for path in paths},
            "input_receipts": {
                rep: file_hash(ROOT / rep / "independent-raw-counts.json") for rep in ("r1", "r2")
            },
            "additional_model_inference": False,
        },
    )
    print(json.dumps({"figure_files": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
