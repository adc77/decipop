import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("runs/study-v3")
    frozen = json.loads((root / "selection-freeze.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    directory = Path("reports/v3-figures")
    directory.mkdir(exist_ok=True)
    suites = [
        "test_original",
        "test_counterfactual",
        "test_depth",
        "test_wording",
        "test_single_branch",
    ]
    labels = ["Original", "Counterfactual", "Depth", "Wording", "Single branch"]
    positions = np.arange(len(suites))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for offset, policy, color in (
        (-0.18, "raw_nll", "#7f8c8d"),
        (0.18, "calibrated_nll", "#2471a3"),
    ):
        name = frozen["choices"]["treatment"][policy]
        for axis, key, title in zip(
            axes,
            ("unknown_accuracy", "all_questions_in_family_correct"),
            ("UNKNOWN accuracy", "All questions in family correct"),
            strict=True,
        ):
            values = [
                100 * evaluation["candidates"][name]["splits"][suite][key] for suite in suites
            ]
            axis.bar(positions + offset, values, width=0.34, color=color, label=f"{policy}: {name}")
            axis.set_title(title)
            axis.set_xticks(positions, labels, rotation=15)
            axis.set_ylim(0, 106)
            axis.set_ylabel("Percent")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle("Fresh-family V3 test: treatment checkpoint selection", fontsize=13)
    for extension in ("svg", "png"):
        figure.savefig(directory / f"selection.{extension}", dpi=180)
    plt.close(figure)
    name = frozen["choices"]["treatment"]["calibrated_nll"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for offset, stage, color in ((-0.18, "raw", "#7f8c8d"), (0.18, "calibrated", "#2471a3")):
        for axis, key, title in zip(
            axes,
            ("nll", "ece"),
            ("Negative log-likelihood", "Ten-bin calibration error"),
            strict=True,
        ):
            values = [
                evaluation["candidates"][name]["splits"][suite][stage][key] for suite in suites
            ]
            axis.bar(positions + offset, values, width=0.34, color=color, label=stage)
            axis.set_title(title)
            axis.set_xticks(positions, labels, rotation=15)
            axis.set_ylabel("Nats per question" if key == "nll" else "Absolute calibration error")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    temperature = frozen["final_calibrations"][name]["temperature"]
    figure.suptitle(
        f"{name}: fresh calibration temperature {temperature:.2f}; unchanged class decisions",
        fontsize=12,
    )
    for extension in ("svg", "png"):
        figure.savefig(directory / f"calibration.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
