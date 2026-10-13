import json
from pathlib import Path

import numpy as np

from decision_lab.evaluation.metrics import paired_family_interval


def read_predictions(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def depth_interval(id_rows, depth_rows, repetitions, confidence, seed):
    rng = np.random.default_rng(seed)

    def grouped(rows):
        families = {}
        for row in rows:
            families.setdefault(row["family_id"], []).append(int(row["prediction"] == row["label"]))
        return np.array([sum(values) for values in families.values()]), np.array(
            [len(values) for values in families.values()]
        )

    id_correct, id_count = grouped(id_rows)
    depth_correct, depth_count = grouped(depth_rows)
    values = []
    for _ in range(repetitions):
        left = rng.integers(0, len(id_count), len(id_count))
        right = rng.integers(0, len(depth_count), len(depth_count))
        values.append(
            id_correct[left].sum() / id_count[left].sum()
            - depth_correct[right].sum() / depth_count[right].sum()
        )
    tail = (1 - confidence) / 2
    return {
        "id_minus_depth_accuracy": float(
            id_correct.sum() / id_count.sum() - depth_correct.sum() / depth_count.sum()
        ),
        "interval": np.quantile(values, [tail, 1 - tail]).tolist(),
        "confidence": confidence,
        "repetitions": repetitions,
        "convention": "independent family resampling in ID and depth pools; one training seed",
    }


def analyze(study: Path):
    protocol = json.loads((study / "protocol.json").read_text())
    config = protocol["evaluation"]
    rows = {
        name: read_predictions(study / "predictions" / name / "test_id.jsonl")
        for name in ("token", "head", "scratch")
    }
    if any(
        [row["example_id"] for row in rows[name]] != [row["example_id"] for row in rows["token"]]
        for name in rows
    ):
        raise ValueError("Paired analysis inputs do not match")
    labels = [row["label"] for row in rows["token"]]
    families = [row["family_id"] for row in rows["token"]]
    comparison = {}
    for metric in ("accuracy", "nll"):
        comparison["token_minus_head_" + metric] = paired_family_interval(
            [row["logits"] for row in rows["token"]],
            [row["logits"] for row in rows["head"]],
            labels,
            families,
            config["bootstrap_repetitions"],
            config["bootstrap_confidence"],
            config["seed"],
            metric,
        )
    for name in rows:
        depth = read_predictions(study / "predictions" / name / "test_depth.jsonl")
        comparison[name + "_depth_drop"] = depth_interval(
            rows[name],
            depth,
            config["bootstrap_repetitions"],
            config["bootstrap_confidence"],
            config["seed"],
        )
    (study / "statistics.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def plot_results(result: dict, baselines: dict, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    names = ["majority", "lexical", "pretrained", "token", "head", "scratch"]
    models = result["models"] | baselines["models"]
    splits = ["test_id", "test_depth", "test_wording", "test_structure"]
    figure, axes = plt.subplots(1, 1, figsize=(10, 4), layout="constrained")
    positions = np.arange(len(splits))
    width = 0.13
    for index, name in enumerate(names):
        values = [100 * models[name]["splits"][split]["raw"]["accuracy"] for split in splits]
        axes.bar(positions + (index - 2.5) * width, values, width, label=name)
    axes.set_xticks(positions, ["ID", "Depth 4–5", "Wording + names", "Diamond structure"])
    axes.set_ylim(0, 100)
    axes.set_ylabel("Accuracy (%)")
    axes.legend(ncols=3, loc="lower left")
    axes.grid(axis="y", alpha=0.2)
    for extension in ("svg", "png"):
        figure.savefig(output / f"accuracy.{extension}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for axis, name in zip(axes, ("token", "head", "scratch"), strict=True):
        groups = models[name]["splits"]["test_id"]["groups"]["proof_depth"]
        deeper = models[name]["splits"]["test_depth"]["groups"]["proof_depth"]
        groups = groups | {key: value for key, value in deeper.items() if key not in groups}
        depths = sorted(int(key) for key in groups if key != "None")
        axis.plot(depths, [groups[str(key)]["accuracy"] for key in depths], marker="o")
        axis.axvspan(3.5, 5.5, alpha=0.1, color="red")
        axis.set(
            ylim=(0, 1),
            xticks=depths,
            title=name,
            xlabel="Shortest proof depth",
            ylabel="Accuracy (TRUE/FALSE)",
        )
    for extension in ("svg", "png"):
        figure.savefig(output / f"accuracy-by-depth.{extension}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for axis, name in zip(axes, ("token", "head", "scratch"), strict=True):
        matrix = models[name]["splits"]["test_id"]["raw"]["confusion_matrix_actual_rows"]
        axis.imshow(matrix, cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
        axis.set(
            title=name,
            xticks=range(3),
            yticks=range(3),
            xticklabels=["T", "F", "U"],
            yticklabels=["T", "F", "U"],
            xlabel="Predicted",
            ylabel="Actual",
        )
    for extension in ("svg", "png"):
        figure.savefig(output / f"confusion.{extension}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    for axis, name in zip(axes.flat, names, strict=True):
        axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        for key, label in (("raw", "Raw"), ("calibrated", "Temperature-scaled")):
            bins = models[name]["splits"]["test_id"][key]["ece_bins"]
            points = [row for row in bins if row["count"]]
            axis.plot(
                [row["confidence"] for row in points],
                [row["accuracy"] for row in points],
                marker="o",
                label=label,
            )
        axis.set(xlim=(0, 1), ylim=(0, 1), title=name, xlabel="Mean confidence", ylabel="Accuracy")
        axis.legend(fontsize=8)
    for extension in ("svg", "png"):
        figure.savefig(output / f"reliability.{extension}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, split in zip(axes, ("test_id", "test_depth"), strict=True):
        for name in ("pretrained", "token", "head", "scratch"):
            curve = models[name]["splits"][split]["calibrated"]["risk_coverage"]
            axis.plot(
                [row["coverage"] for row in curve], [row["risk"] for row in curve], label=name
            )
        axis.set(xlim=(0, 1), ylim=(0, 1), title=split, xlabel="Coverage", ylabel="Error rate")
        axis.legend(fontsize=8)
    for extension in ("svg", "png"):
        figure.savefig(output / f"risk-coverage.{extension}", dpi=160)
    plt.close(figure)


def plot_learning(study: Path, output: Path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name in ("token", "head", "scratch"):
        history = json.loads((study / f"{name}-state.json").read_text())["history"]
        axes[0].plot(
            [row["step"] for row in history],
            [row["accuracy"] for row in history],
            marker="o",
            label=name,
        )
        axes[1].plot(
            [row["step"] for row in history],
            [row["nll"] for row in history],
            marker="o",
            label=name,
        )
    axes[0].set(xlabel="Optimizer update", ylabel="Development accuracy", ylim=(0, 1))
    axes[1].set(xlabel="Optimizer update", ylabel="Development restricted-class NLL")
    for axis in axes:
        axis.legend()
    for extension in ("svg", "png"):
        figure.savefig(output / f"learning.{extension}", dpi=160)
    plt.close(figure)


def write_report(study: Path, report: Path, baselines: dict):
    result = json.loads((study / "evaluation.json").read_text())
    statistics = analyze(study)
    plot_results(result, baselines, report.parent / "figures")
    plot_learning(study, report.parent / "figures")
    calibration = json.loads((study / "calibration-freeze.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    ledger = json.loads((study / "ledger.json").read_text())
    allocated = ledger["initial_pilot_allocation_usd"] + sum(
        row["maximum_estimated_usd"] for row in ledger["invocations"]
    )
    models = result["models"] | baselines["models"]
    lines = [
        "# Decision Model Lab: v1 results",
        "",
        "A single-seed study of three-class reasoning over supplied facts and explicit rules. "
        "TRUE means derivable, FALSE means the explicit negation is derivable, "
        "and UNKNOWN means neither. "
        "Abstention is a separate confidence-based policy.",
        "",
        "## Held-out accuracy",
        "",
        "| Model | ID | Depth 4–5 | Wording + names | Diamond structure |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("majority", "lexical", "pretrained", "token", "head", "scratch"):
        values = [
            models[name]["splits"][split]["raw"]["accuracy"]
            for split in ("test_id", "test_depth", "test_wording", "test_structure")
        ]
        lines.append("| " + name + " | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    lines += [
        "",
        "![Accuracy](figures/accuracy.svg)",
        "",
        "![Accuracy by proof depth](figures/accuracy-by-depth.svg)",
        "",
        "The proof-depth plot excludes UNKNOWN because it has no proof. "
        "Depths 0–3 come from ID tests and depths 4–5 from the deeper test pool; "
        "these are separate family distributions.",
        "",
        "![ID confusion matrices](figures/confusion.svg)",
        "",
        "## Answerable versus UNKNOWN",
        "",
        "| Model | Split | TRUE/FALSE accuracy | UNKNOWN accuracy |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in ("token", "head", "scratch"):
        for split in ("test_id", "test_depth"):
            groups = models[name]["splits"][split]["groups"]["label"]
            count = sum(groups[label]["questions"] for label in ("TRUE", "FALSE"))
            accuracy = (
                sum(
                    groups[label]["questions"] * groups[label]["accuracy"]
                    for label in ("TRUE", "FALSE")
                )
                / count
            )
            lines.append(
                f"| {name} | {split} | {accuracy:.2%} | {groups['UNKNOWN']['accuracy']:.2%} |"
            )
    lines += [
        "",
        "## ID reliability",
        "",
        "| Model | Macro F1 | Raw NLL | Cal NLL | Raw ECE | Cal ECE | Cal Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in models:
        row = models[name]["splits"]["test_id"]
        raw, calibrated = row["raw"], row["calibrated"]
        lines.append(
            f"| {name} | {raw['macro_f1']:.4f} | {raw['nll']:.4f} | {calibrated['nll']:.4f} | "
            f"{raw['ece']:.4f} | {calibrated['ece']:.4f} | "
            f"{calibrated['brier_sum_over_classes']:.4f} |"
        )
    lines += [
        "",
        "Brier scores sum squared errors over all three classes. ECE uses 10 equal-width bins. "
        "AURC averages risk at retained counts 1..N; stable input order breaks confidence ties. "
        "The local evaluation JSON contains full bin counts, confusion matrices, "
        "risk–coverage curves, and abstention results.",
        "",
        "![Reliability](figures/reliability.svg)",
        "",
        "![Risk and coverage](figures/risk-coverage.svg)",
        "",
        "## Confidence under distribution shift",
        "",
        "| Model | Split | Cal NLL | Cal ECE | Cal AURC | Wrong with confidence ≥90% |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("token", "head", "scratch"):
        for split, metrics in models[name]["splits"].items():
            row = metrics["calibrated"]
            lines.append(
                f"| {name} | {split} | {row['nll']:.4f} | {row['ece']:.4f} | "
                f"{row['aurc']:.4f} | {row['high_confidence_wrong']} |"
            )
    lines += [
        "",
        "## Deferral with calibration-selected thresholds",
        "",
        "Each cell is achieved test coverage / error rate among retained answers. "
        "The column's target is chosen on calibration data. "
        "All predictions tied at a threshold are retained, so achieved coverage can "
        "exceed its target; a flat-confidence baseline can retain every answer.",
        "",
        "| Model | Split | Target 50% | Target 75% | Target 90% |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ("majority", "pretrained", "token", "head", "scratch"):
        selected_splits = (
            ("test_id",)
            if name == "majority"
            else ("test_id", "test_depth", "test_wording", "test_structure")
        )
        for split in selected_splits:
            values = []
            for target in ("0.5", "0.75", "0.9"):
                row = models[name]["splits"][split]["abstention"][target]
                risk = "—" if row["risk"] is None else f"{row['risk']:.2%}"
                values.append(f"{row['coverage']:.2%} / {risk}")
            lines.append(f"| {name} | {split} | " + " | ".join(values) + " |")
    lines += [
        "",
    ]
    lines += [
        "## Token-output validity",
        "",
        "| Model | Split | Restricted accuracy | Unrestricted accuracy | "
        "Invalid answers | Outside mass |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("pretrained", "token"):
        for split, row in models[name]["splits"].items():
            lines.append(
                f"| {name} | {split} | {row['raw']['accuracy']:.2%} | "
                f"{row['unrestricted_accuracy']:.2%} | "
                f"{row['unrestricted_invalid_answer_rate']:.2%} | "
                f"{row['mean_probability_outside_answer_tokens']:.4f} |"
            )
    lines += [
        "",
        "Unrestricted accuracy treats any vocabulary token outside A/B/C as an error. "
        "Restricted scores renormalize probabilities over A/B/C; these answer different "
        "questions and are shown separately.",
        "",
        "## Family uncertainty",
        "",
    ]
    for name, value in statistics.items():
        effect = (
            value["left_minus_right"]
            if "left_minus_right" in value
            else value["id_minus_depth_accuracy"]
        )
        lower, upper = value["interval"]
        lines.append(
            f"- {name}: {effect:.4f}; 95% family-bootstrap interval [{lower:.4f}, {upper:.4f}]."
        )
    lines += [
        "",
        "Token-versus-head comparisons resample the same theory families for both models. "
        "ID-versus-depth drops resample the two disjoint family pools independently. "
        "These intervals measure test-family sampling uncertainty, not training-seed stability.",
        "",
        "## Robustness",
        "",
        "| Model | Transformation | Accuracy | Same-label consistency | Changed-label accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ("token", "head", "scratch"):
        for transformation, row in models[name]["robustness"].items():
            consistency = (
                f"{row['label_preserving_consistency']:.2%}"
                if row["label_preserving_consistency"] is not None
                else "—"
            )
            changed = (
                f"{row['changed_label_accuracy']:.2%}"
                if row["changed_label_accuracy"] is not None
                else "—"
            )
            lines.append(
                f"| {name} | {transformation} | {row['raw']['accuracy']:.2%} | "
                f"{consistency} | {changed} |"
            )
    lines += [
        "",
        "Each transformation uses 100 eligible held-out families and three questions per family. "
        "The oracle recomputes labels after premise deletion and fact changes. "
        "Consistency can include wrong answers; accuracy is reported separately. "
        "Paraphrases were mechanically checked against the formal source "
        "without independent human review.",
        "",
        "## Inference measurements",
        "",
        "| Model | B1 median (ms) | B1 p95 (ms) | B32 decisions/s | B32 cost / 1,000 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("pretrained", "token", "head", "scratch"):
        performance = models[name]["performance"]
        lines.append(
            f"| {name} | {performance['batch_one_median_ms']:.2f} | "
            f"{performance['batch_one_p95_ms']:.2f} | "
            f"{performance['decisions_per_second']:.1f} | "
            f"${performance['batch_cost_per_1000_usd']:.6f} |"
        )
    lines += [
        "",
        "Benchmarks use the same 500 semantic inputs selected before scoring, one L4, "
        "BF16 autocast, GPU synchronization, 20 warmups, and fixed batch padding. "
        "Batch-1 measures 500 forward passes; batch-32 uses 480 requests in full batches. "
        "Forward timings exclude tokenization and transfers; tokenization is reported separately. "
        "Costs use requested L4 + 4 physical CPU cores + 32 GiB RAM rates. "
        "Model loading is separate; platform cold-start and queuing are not isolated. "
        "See [Modal pricing](https://modal.com/pricing). These are not reconciled invoice costs.",
        "",
    ]
    lines += [
        "| Model | B1 mean preparation (ms) | B32 preparation (s) | Model loading (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("pretrained", "token", "head", "scratch"):
        model = models[name]
        performance = model["performance"]
        lines.append(
            f"| {name} | {performance['batch_one_mean_preparation_ms']:.3f} | "
            f"{performance['batched_preparation_seconds']:.4f} | "
            f"{model['load_seconds']:.2f} |"
        )
    lines += [
        "",
        "Preparation includes allocation, padding, and host-to-device copies; "
        "it is measured separately from forward computation. Tokenization precedes "
        "both. These are warm inference measurements, not end-to-end API latency.",
        "",
        "## Training and accounting",
        "",
        "![Development learning curves](figures/learning.svg)",
        "",
    ]
    lines += [
        "| Model | Worker seconds | Peak allocated GPU GiB | Selected development NLL |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("token", "head", "scratch"):
        attempts = [json.loads(path.read_text()) for path in study.glob(f"{name}-attempt-*.json")]
        state = json.loads((study / f"{name}-state.json").read_text())
        lines.append(
            f"| {name} | {sum(row['worker_seconds'] for row in attempts):.1f} | "
            f"{max(row['peak_allocated_gpu_bytes'] for row in attempts) / 2**30:.3f} | "
            f"{min(row['nll'] for row in state['history']):.6f} |"
        )
    lines += [
        "",
        "Worker seconds include setup, updates, development scoring, and checkpoint "
        "serialization. They exclude platform startup/queue time and the subsequent "
        "Volume commits, so they are not total billable GPU duration.",
        "Only successfully returned chunks contribute to that table; failed calls remain "
        "in the invocation ledger and the provider's app-level billing records.",
        "",
    ]
    for name in ("token", "head", "scratch"):
        state = json.loads((study / f"{name}-state.json").read_text())
        history = state["history"]
        chosen = min(history, key=lambda row: (row["nll"], -row["accuracy"]))
        lines.append(
            f"- {name}: {state['parameters']:,} parameters; "
            f"{state['examples_seen']:,} examples; {state['input_tokens']:,} input tokens; "
            f"{state['step']:,} optimizer updates; selected step {chosen['step']}; "
            f"temperature {calibration['models'][name]['temperature']:.4f}."
        )
    lines += [
        "",
        f"Primary seed: {protocol['primary_seed']}. "
        "Each model completed three passes over 20,001 training questions. "
        "The conservative cumulative invocation allocation, including $1 for the initial pilots, "
        f"is ${allocated:.2f}. "
        "This is a timeout-based planning allocation, not actual invoiced spend. "
        "Image preparation and retained storage are tracked separately. "
        "Actual account usage remains unreconciled.",
        "",
        f"Frozen protocol: `{protocol['identity']}`. "
        f"Calibration and checkpoint freeze: `{calibration['identity']}`. "
        "Selected weights, temperatures, thresholds, test variants, "
        "and performance input IDs were fixed before test scoring.",
        "",
        "## Limits and artifacts",
        "",
        "This experiment tests synthetic, bounded rule reasoning. "
        "It does not establish general decision ability. "
        "The token and direct-head approaches differ in output layer and loss objective; "
        "the new head also uses a 1e-4 learning rate while the shared backbones and "
        "token projection use 2e-5. "
        "the scratch comparison also differs in model size, tokenizer, and encoder architecture. "
        "One seed cannot establish training stability. "
        "In-distribution calibration does not guarantee OOD calibration.",
        "",
        "Model artifacts contain FP32 safetensors, custom code, tokenizer, calibration, protocol, "
        "evaluation metrics, license attribution, and fixed loading checks. "
        "Hugging Face upload and remote-download verification are a separate release step. "
        "Implementation and reports remain local and uncommitted.",
        "",
        "## Scope and pilot notes",
        "",
        "The final preset uses 3–5 entities, 4–8 extra facts, an actual 514-entry scratch "
        "tokenizer inside an 8,192-slot embedding, and microbatch 16 for every architecture "
        "after the throughput gate. The common exposure cap remained three passes. "
        "Only seed 17 was run; the optional second seed and external RuleTaker test were not run. "
        "Paraphrase review and isolated platform cold-start/pure-transfer measurements remain "
        "limitations; model loading, tokenization, and preparation including copies were "
        "measured separately. Dispatch-to-worker timings include queueing and startup, "
        "and are recorded in the phase timing JSON files.",
        "",
        "The first 40-step pilot did not establish full tiny-set fit; a 120-step pilot "
        "then fit both pretrained approaches to 100%. The scratch gate fit its 32 examples "
        "to 100% in 400 steps. These were wiring checks, not final test results. "
        "See GPU_PILOT.md and runs/study-v1/gate-*.json for retained measurements.",
    ]
    if "operational_amendment" in ledger:
        lines += [
            "",
            "A local DNS disconnect canceled a head call after its 1,250-update "
            "development check but before a complete recovery save. The first retry "
            "collided with its orphaned selection directory. That orphan was preserved "
            "under a diagnostic filename and the conflicting path cleared. Both retries "
            "started from the verified 1,000-update recovery. Two failed 250-update "
            "attempts added 500 executed updates and 16,000 example presentations "
            "that are absent from the final learned trajectory. The selected final "
            "model still received the same 60,003-example exposure as the other models. "
            "One additional head call was explicitly allowed beyond the original "
            "nine-call operational cap. The scientific protocol, $20 compute limit, "
            "and $10 reserve were unchanged. See operational-invocation-amendment.json "
            "and interruption.json in the local study records. The completed 250-update "
            "replay reproduced byte-identical selected safetensors; "
            "replayed-checkpoint-verification.json records that comparison.",
        ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return {"report": str(report), "statistics": statistics}
