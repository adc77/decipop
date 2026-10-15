import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.studies.v8.analysis import predictions
from decision_lab.studies.v8.protocol import ROOT, verify_study
from decision_lab.training.checkpoint import file_hash

PREFIX = ["uv", "run", "--extra", "train", "--extra", "cloud", "--extra", "analysis"]


def execute(stage, command):
    print(json.dumps({"stage": stage, "status": "started"}), flush=True)
    with (ROOT / f"close-{stage}.log").open("x") as output:
        subprocess.run(PREFIX + command, stdout=output, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"stage": stage, "status": "complete"}), flush=True)


def examples():
    lines, records = ["\n## Preselected premise-removal examples\n"], {}
    for rep in ("r1", "r2"):
        root = ROOT / rep
        protocol = json.loads((root / "protocol.json").read_text())
        family = protocol["audit"]["preselected_family"]
        registry = {
            r["family_id"]: r
            for r in (
                json.loads(line) for line in (root / "data/families.jsonl").read_text().splitlines()
            )
        }
        target = registry[family]["target_slot"]
        frozen = json.loads((root / "selection-freeze.json").read_text())
        output, facts = {}, {}
        for arm in ("parent", "control", "treatment"):
            rows, logits, labels = predictions(root, arm, "test_paired")
            if arm == "parent":
                temperature = json.loads((root / "parent/calibration.json").read_text())[
                    "temperature"
                ]
            else:
                choice = frozen["choices"][arm]["calibrated_nll"]
                temperature = frozen["calibrations"][choice]["temperature"]
            output[arm] = {}
            for i, row in enumerate(rows):
                if row.family_id != family:
                    continue
                variant, slot = row.example_id.rsplit("-", 2)[-2:]
                if int(slot) != target:
                    continue
                probabilities = (logits[i] / temperature).softmax(-1)
                decision = int(logits[i].argmax())
                output[arm][variant] = dict(
                    gold=protocol["settings"]["class_labels"][labels[i]],
                    prediction=protocol["settings"]["class_labels"][decision],
                    confidence=float(probabilities[decision]),
                )
                facts[variant] = row.facts
                question, rules = row.question, row.rules
            if set(output[arm]) != {"base", "support_removal"}:
                raise ValueError("Preselected family is not the declared premise-removal pair")
        if (
            output["parent"]["base"]["gold"] != "TRUE"
            or output["parent"]["support_removal"]["gold"] != "UNKNOWN"
        ):
            raise ValueError("Preselected oracle transition differs")
        removed = sorted(set(facts["base"]) - set(facts["support_removal"]))
        added = sorted(set(facts["support_removal"]) - set(facts["base"]))
        records[rep] = dict(
            family_id=family,
            question=question,
            base_facts=facts["base"],
            rules=rules,
            removed_facts=removed,
            added_facts=added,
            outputs=output,
            input_sha256=file_hash(root / "data/test_paired.jsonl"),
            prediction_sha256={
                arm: file_hash(root / "predictions" / arm / "test_paired.jsonl") for arm in output
            },
            additional_model_inference=False,
        )
        lines += [f"### {rep}: {question}\n", "Base facts:\n"]
        lines += [f"- {fact}" for fact in facts["base"]]
        lines += ["\nRules:\n"] + [f"- {rule}" for rule in rules]
        lines += [
            f"\nRemoved: {removed}. Added irrelevant filler: {added}.\n",
            "| Input | Gold | Parent | Paired CE | Added consistency |",
            "| --- | --- | --- | --- | --- |",
        ]
        for variant in ("base", "support_removal"):
            cells = [
                f"{output[arm][variant]['prediction']} ({output[arm][variant]['confidence']:.2%})"
                for arm in ("parent", "control", "treatment")
            ]
            lines.append(
                f"| {variant} | {output['parent'][variant]['gold']} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    lines += [
        "Examples were fixed in the data freeze, without selecting a favorable model result. "
        "Confidence comes from retained BF16 logits and each saved temperature; no extra "
        "inference was run. These cases illustrate behavior without estimating population "
        "performance or proving a mechanism.\n"
    ]
    atomic_json(ROOT / "preselected-examples.json", records)
    return "\n".join(lines)


def document():
    study = verify_study()
    counts = {}
    checks = {}
    releases = {}
    for rep in ("r1", "r2"):
        root = ROOT / rep
        raw = json.loads((root / "independent-raw-counts.json").read_text())["counts"]
        analysis = json.loads((root / "analysis.json").read_text())
        counts[rep] = {
            arm: raw[arm]["test_paired"]["complete_families"]
            for arm in ("parent", "control", "treatment")
        }
        checks[rep] = analysis["engineering_gate"]
        releases[rep] = json.loads((root / "huggingface-release.json").read_text())
    summary = json.loads((ROOT / "replication-summary.json").read_text())
    improves = {
        arm: all(
            json.loads((ROOT / rep / "analysis.json").read_text())["arms_against_parent"][arm][
                "interval"
            ][0]
            > 0
            for rep in ("r1", "r2")
        )
        for arm in ("control", "treatment")
    }
    if summary["all_engineering_checks_pass_in_both_repetitions"]:
        decision = (
            "Added consistency passes the declared paired-world and retention "
            "checks on both repetitions."
        )
    else:
        decision = "Added consistency does not pass the full declared replicated decision rule."
    if not any(improves.values()):
        decision += (
            " Neither arm shows a positive lower bound versus parent on both "
            "repetitions; close this fine-tuning line."
        )
    elif improves["control"] and not summary["primary_gains_positive_in_both_repetitions"]:
        decision += (
            " Paired CE improves versus parent in both repetitions; the added consistency "
            "term has no replicated primary advantage. Credit the paired data."
        )
    else:
        decision += (
            " Treatment improves over the unchanged parent in both repetitions, but "
            "the control comparison remains mixed and transfer remains weak. Close this "
            "bounded loss experiment; any architecture study needs a separate scope."
        )
    billing = json.loads((ROOT / "billing-attribution.json").read_text())
    cleanup = json.loads((ROOT / "cleanup.json").read_text())
    audited = {
        rep: json.loads((ROOT / rep / "final-audit.json").read_text()) for rep in ("r1", "r2")
    }
    models = []
    for rep, records in releases.items():
        for arm, record in records.items():
            models.append(f"- {rep} {arm}: `{record['repo_id']}` at `{record['commit']}`.")
    footer = (
        "\n## Verified completion\n\n" + decision + "\n\n"
        f"Complete paired-world counts (parent/control/treatment): {counts}. "
        f"Replicated positive parent contrasts: {improves}. "
        "These secondary parent checks do not add a joint confidence guarantee.\n\n"
        f"All 126 closing CPU tests pass. Final audits verify "
        f"{sum(a['prediction_rows_verified'] for a in audited.values()):,} prediction rows "
        f"across {sum(a['prediction_files_verified'] for a in audited.values())} files. "
        "Both independent raw-count audits and all 44 final metric recomputations pass. "
        "Four selected bundles pass fresh pinned-Hub CPU reference checks; every scheduled "
        "candidate is retained privately. Original preparation, two discarded pilots, "
        "source-guard rejection and CPU seed-replay failure remain recorded. The corrected "
        "pilots explicitly consumed the original retry allowances. A later expired-JWT request "
        "was rejected before the final chunk started; one explicit operational amendment "
        "resumes update 756 through 1,134 with saved optimizer/RNG state and unchanged "
        "trajectory prefix. The failed reservation remains in the ledger; total reservation "
        "stays $9, reallocated to $8.50 compute plus $0.50 storage/transfer. "
        "A later cancelled selection call retained all 30 candidate prediction files. "
        "Those 30,240 rows were reused with the unchanged selector; separate calibration "
        "and final tests completed inside the originally reserved test call, with no "
        "completed inference repeated or extra GPU allocation.\n\n"
        f"All {len(cleanup['apps'])} corrected-study owned apps are stopped with zero tasks, "
        "and its temporary Volume is deleted. The original three apps and Volume were "
        "verified separately before deletion. Full optimizer/RNG states were temporary; "
        "frozen data, source, logits, trajectories and model weights remain.\n\n"
        f"Observed owned metering, including original pilot resources: "
        f"${billing['owned_resource_metered_usd']} at {billing['snapshot_utc']}; "
        f"workspace billed ${billing['workspace_billed_usd']}. "
        "This is a snapshot, not an invoice or live wallet balance.\n\n"
        + "\n".join(models)
        + "\n\n"
        "Code, reports, raw records and source archives remain local, uncommitted and unpushed. "
        "The final archive manifest and external closure receipt record archival verification.\n"
    )
    report = Path("reports/V8_RESULTS.md")
    text = report.read_text()
    text = text.replace(
        "Both primary intervals positive: False. All engineering checks pass: False.",
        "The declared replication check fails: the first interval includes zero. "
        "All three retention checks pass in both runs. This is mixed evidence for "
        "an advantage from the added consistency loss, not evidence of equivalence.",
    )
    for rep in ("r1", "r2"):
        analysis = json.loads((ROOT / rep / "analysis.json").read_text())
        begin = text.index("Parent contrasts", text.index(f"## {rep}\n"))
        end = text.index("| Edit / person", begin)
        rows = [
            "Parent comparisons (differences in percentage points):\n",
            "| Check | Difference | 95% world-sampling interval |",
            "| --- | ---: | ---: |",
        ]
        comparisons = [
            ("CE: complete paired worlds", analysis["arms_against_parent"]["control"]),
            ("Consistency: complete paired worlds", analysis["arms_against_parent"]["treatment"]),
            ("Consistency: complete base worlds", analysis["base_world_retention_against_parent"]),
            (
                "Consistency: original UNKNOWN",
                analysis["retention_against_parent"]["test_original"],
            ),
            (
                "Consistency: counterfactual UNKNOWN",
                analysis["retention_against_parent"]["test_counterfactual"],
            ),
        ]
        for label, comparison in comparisons:
            value = comparison.get("estimate", comparison.get("left_minus_right"))
            low, high = comparison["interval"]
            rows.append(f"| {label} | {100 * value:.2f} | [{100 * low:.2f}, {100 * high:.2f}] |")
        text = text[:begin] + "\n".join(rows) + "\n\n" + text[end:]
    text = text.replace(
        "If only paired CE improves, credit the paired data. If neither arm improves faithful "
        "edits, close this fine-tuning line; no coefficient sweep or extra seeds are authorized.",
        "Treatment improves over parent in both runs; CE's second-run parent interval "
        "includes zero. The added loss does not pass the declared replicated advantage "
        "over CE. No coefficient sweep or extra seeds are authorized.",
    )
    report.write_text(text + examples() + footer)
    execution = Path("docs/V8_EXECUTION.md")
    with execution.open("a") as output:
        output.write(
            "\n## Completed execution and retention\n\n"
            + footer.split("## Verified completion\n\n", 1)[1]
        )
    journal = Path("RESEARCH_LOG.md")
    text = journal.read_text()
    start, end = (
        text.index("**Current state:**"),
        text.index("\n\n", text.index("**Current state:**")),
    )
    state = (
        "**Current state:** V1–V8 execution, artifact verification and owned-resource cleanup "
        f"are complete. V8 complete paired-world counts (parent/control/treatment): {counts}. "
        + decision
        + " All 126 closing CPU tests, both independent raw counts and 44 metric "
        f"recomputations pass; {sum(a['prediction_rows_verified'] for a in audited.values()):,} "
        "prediction rows are verified. Twenty-one selected bundles remain private on HF, "
        "including all four V8 bundles, with all scheduled V8 weights pinned privately. "
        "Original preparation errors and two discarded pilots remain preserved; corrected "
        "pilots used the reserved retry allowances, and no further experiment is submitted. "
        f"Observed owned V8 metering ${billing['owned_resource_metered_usd']}, with "
        f"${billing['workspace_billed_usd']} workspace billed at {billing['snapshot_utc']}; "
        "this is an observation, not an invoice. V5–V8 source/reports/raw evidence remain "
        "local, uncommitted and unpushed. The final source archive and external closure "
        "receipt govern archival verification; optimizer/RNG states were temporary."
    )
    text = text[:start] + state + text[end:]
    text += (
        f"\n### {datetime.now(UTC).isoformat()}: V8 verified completion\n\n"
        + footer.split("## Verified completion\n\n", 1)[1]
    )
    journal.write_text(text)
    readme = Path("README.md")
    text = readme.read_text()
    begin, end = text.index("**Status:"), text.index("**", text.index("**Status:") + 2) + 2
    text = (
        text[:begin] + "**Status: V1–V8 execution, artifact verification and owned-resource "
        "cleanup are complete.**" + text[end:]
    )
    text += (
        "\nV8 completes the paired-loss comparison. "
        + decision
        + " Read [V8 results](reports/V8_RESULTS.md), [protocol](docs/V8_PROTOCOL.md) "
        "and [execution record](docs/V8_EXECUTION.md). No further training is submitted.\n"
    )
    text = text.replace(
        "Seventeen selected research model bundles", "Twenty-one selected research model bundles", 1
    )
    text = text.replace(
        "New V5–V7 implementation and evidence", "New V5–V8 implementation and evidence", 1
    )
    text = text.replace(
        "All V1–V7 owned temporary Modal Volumes", "All V1–V8 owned temporary Modal Volumes", 1
    )
    text = text.replace(
        "Full V5–V7 raw records and available source archives remain local.",
        "Full V5–V8 raw records and available source archives remain local.",
    )
    text = text.replace("full V5–V7 raw records", "full V5–V8 raw records", 1)
    text = text.replace(
        "The [paired-loss proposal](docs/V8_PROPOSAL.md) is authorized under "
        "[V8's protocol](docs/V8_PROTOCOL.md). Its first two discarded pilots are preserved; "
        "a target-position correlation was caught before full training, and corrected "
        "preparation receives a new freeze. Read the journal for current execution status.",
        "V8 completed the [paired-loss proposal](docs/V8_PROPOSAL.md) under "
        "[its protocol](docs/V8_PROTOCOL.md) and documented operational amendments. "
        "Its preparation failures and discarded pilots remain preserved in the journal.",
    )
    readme.write_text(text)
    atomic_json(
        ROOT / "completion-decision.json",
        dict(
            study_identity=study["identity"],
            counts=counts,
            declared_checks=checks,
            replicated_parent_improvements=improves,
            decision=decision,
            additional_training_submitted=False,
            report_sha256=file_hash(Path("reports/V8_RESULTS.md")),
        ),
    )


def main():
    status = json.loads((ROOT / "finish-status.json").read_text())
    if status["stage"] != "costs" or status["status"] != "complete":
        raise ValueError("Pipeline must complete before final archival")
    execute("metrics", ["python", "src/decision_lab/studies/v8/metric_precheck.py"])
    execute("figure", ["python", "src/decision_lab/studies/v8/plot.py"])
    with Path("artifacts/v8-cpu-closing.log").open("x") as output:
        subprocess.run(
            PREFIX + ["--extra", "baseline", "python", "-m", "pytest", "-q"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    document()
    execute("lint", ["ruff", "check", "."])
    execute("format", ["ruff", "format", "--check", "."])
    execute("closure", ["python", "src/decision_lab/studies/v8/close.py"])


if __name__ == "__main__":
    main()
