import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    exposure = json.loads(Path("runs/study-v2/selected-exposure.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for arm, state in exposure.items():
        history = state["development_history"]
        selected = next(row for row in history if row["step"] == state["selected_step"])
        for axis, metric in zip(axes, ("accuracy", "nll"), strict=True):
            axis.plot(
                [row["step"] for row in history],
                [row[metric] for row in history],
                marker="o",
                label=arm,
            )
            axis.scatter(
                [selected["step"]],
                [selected[metric]],
                s=120,
                facecolors="none",
                edgecolors="black",
            )
            axis.set_xlabel("Training update")
            axis.set_ylabel("Development " + ("accuracy" if metric == "accuracy" else "NLL"))
            axis.legend()
    axes[0].set_ylim(0, 1)
    axes[1].axhline(1.0986122886681098, color="gray", linestyle=":", label="Uniform NLL")
    axes[1].legend()
    directory = Path("reports/v2-figures")
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(directory / f"development-selection.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
