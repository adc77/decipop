import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import modal
import torch
from modal._utils.async_utils import synchronize_api

from decision_lab.publication import atomic_json, release_inventory
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal
from decision_lab.v3.study import metrics
from v4_cleanup import inspect_apps
from v6_metric_audit import verify_metrics
from v8_lab.analysis import analyse, predictions
from v8_lab.data import TESTS
from v8_lab.protocol import ROOT, verify, verify_study

inspect_owned_apps = synchronize_api(inspect_apps)


def cli(arguments):
    result = subprocess.run(
        ["uv", "run", "--extra", "cloud", "modal", *arguments, "--profile", "amandogra2016"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def audit(root):
    protocol = verify(root)
    spec = protocol["settings"]
    frozen = check_seal(json.loads((root / "selection-freeze.json").read_text()))
    evaluation = json.loads((root / "evaluation.json").read_text())
    if (evaluation["protocol_identity"], evaluation["selection_identity"]) != (
        protocol["identity"],
        frozen["identity"],
    ):
        raise ValueError("Evaluation identities differ")
    manifest = json.loads((root / "data/manifest.json").read_text())
    for arm in ("control", "treatment"):
        state = json.loads((root / f"state-{arm}.json").read_text())
        updates = json.loads((root / f"trajectory-{arm}.json").read_text())["updates"]
        expected = manifest["splits"][f"train_{arm}"]["questions"] * spec["epochs"]
        if (
            not state["complete"]
            or state["step"] != 1134
            or state["presentations"] != expected
            or state["protocol_identity"] != protocol["identity"]
        ):
            raise ValueError("Training state/exposure differs")
        if [r["step"] for r in updates] != list(range(1, 1135)) or sum(
            r["examples"] for r in updates
        ) != expected:
            raise ValueError("Training trajectory differs")
        tokens = (
            protocol["audit"]["token_audit"][f"train_{arm}"]["total_input_tokens"] * spec["epochs"]
        )
        if sum(r["input_tokens"] for r in updates) != tokens:
            raise ValueError("Token exposure differs")
    for name, digest in json.loads((root / "retained-records.json").read_text()).items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Retained record differs: {name}")
    expected_files = {
        root / "predictions" / name / f"{split}.jsonl"
        for name, files in frozen["prediction_files"].items()
        for split in files
    }
    expected_files |= {
        root / "predictions" / arm / f"{split}.jsonl"
        for arm in ("control", "treatment")
        for split in TESTS
    }
    expected_files |= {
        root / "predictions/parent" / f"{split}.jsonl"
        for split in ("test_paired", "test_multi", "test_original", "test_counterfactual")
    }
    if set((root / "predictions").rglob("*.jsonl")) != expected_files:
        raise ValueError("Prediction inventory differs")
    total, hashes, checks = 0, {}, {}
    for path in sorted(expected_files):
        name, split = path.parent.name, path.stem
        rows, logits, labels = predictions(root, name, split)
        digest = file_hash(path)
        hashes[str(path)] = digest
        total += len(rows)
        if name in frozen["prediction_files"] and digest != frozen["prediction_files"][name][split]:
            raise ValueError("Frozen selection predictions changed")
        if name in ("control", "treatment"):
            if digest != evaluation["arms"][name]["prediction_files"][split]:
                raise ValueError("Frozen final predictions changed")
            selected = frozen["choices"][name]["calibrated_nll"]
            temperature = frozen["calibrations"][selected]["temperature"]
            rebuilt = metrics(rows, {"logits": logits, "labels": labels}, temperature, spec)
            checks[f"{name}/{split}"] = verify_metrics(
                rebuilt, evaluation["arms"][name]["splits"][split], logits, labels, temperature
            )
    expected_total = sum(manifest["splits"][p.stem]["questions"] for p in expected_files)
    if total != expected_total:
        raise ValueError("Prediction count differs")
    storage = json.loads((root / "checkpoint-storage.json").read_text())
    if (
        storage["candidates_verified"] != 10
        or not storage["all_additional_weight_metadata_verified"]
    ):
        raise ValueError("Candidate storage is incomplete")
    for record in json.loads((root / "retained-candidates.json").read_text()).values():
        if record["path"] is not None and file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError("Local selected/initial candidate differs")
    for arm in ("control", "treatment"):
        manifest = release_inventory(Path("artifacts/v8-releases") / root.name / arm)
        if manifest["protocol_identity"] != protocol["identity"]:
            raise ValueError("Selected release protocol differs")
    atomic_json(root / "retained-predictions.json", hashes)
    result = {
        "valid": True,
        "protocol_identity": protocol["identity"],
        "prediction_rows_verified": total,
        "prediction_rows_expected": expected_total,
        "prediction_files_verified": len(expected_files),
        "candidate_count": 10,
        "metric_recomputation_checks": checks,
        "analysis_sha256": file_hash(root / "analysis.json"),
    }
    atomic_json(root / "final-audit.json", result)
    return {k: v for k, v in result.items() if k != "metric_recomputation_checks"}


def historical():
    from v5_lab.protocol import verify as verify_v5
    from v6_lab.protocol import verify as verify_v6

    prior_identities = {
        "v5": verify_v5(Path("runs/study-v5"))["identity"],
        "v6": verify_v6(Path("runs/study-v6"))["identity"],
    }
    from v7_lab.protocol import verify_study as verify_v7

    prior_identities["v7"] = verify_v7()["identity"]
    exclusions = {}
    for repetition in ("r1", "r2"):
        for name, digest in verify(ROOT / repetition)["audit"]["exclusion_files"].items():
            if file_hash(Path(name)) != digest:
                raise ValueError(f"Historical exclusion registry changed: {name}")
            exclusions[name] = digest
    manifest = json.loads(
        Path("artifacts/v7-source-snapshot/manifest-final-with-reports.json").read_text()
    )
    checked = {}
    for name, digest in manifest["files"].items():
        if (
            name.startswith(
                (
                    "src/",
                    "v4_lab/",
                    "v5_lab/",
                    "v6_lab/",
                    "v7_lab/",
                    "tests/",
                    "configs/",
                    "reports/v5-",
                    "reports/v6-",
                    "reports/v7-",
                )
            )
            or name
            in (
                "pyproject.toml",
                "uv.lock",
                "reports/V5_RESULTS.md",
                "reports/V6_RESULTS.md",
                "reports/V7_RESULTS.md",
            )
            or Path(name).name.startswith(("v6_", "v7_"))
        ):
            if file_hash(Path(name)) != digest:
                raise ValueError(f"Historical scientific/report file changed: {name}")
            checked[name] = digest
    result = {
        "valid": True,
        "files_checked": len(checked),
        "files": checked,
        "prior_protocol_identities": prior_identities,
        "historical_models_and_data_unchanged": True,
        "exclusion_files_checked": exclusions,
    }
    atomic_json(ROOT / "historical-check.json", result)
    return {"historical_files_unchanged": len(checked)}


def report():
    retention = ROOT / "r1/early-retention.json"
    deadline = time.monotonic() + 3600
    while retention.exists():
        stages = json.loads(retention.read_text())["stages"]
        if any(v["status"] == "failed" for v in stages.values()):
            raise ValueError("Inspect the preserved early r1 retention failure")
        if stages.get("publish", {}).get("status") == "complete":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Early r1 retention did not complete before reporting")
        time.sleep(10)
    freeze = verify_study()
    destination = Path("reports/V8_RESULTS.md")
    if destination.exists():
        raise FileExistsError(destination)
    lines = [
        "# V8 results: paired prediction consistency",
        "",
        "Scientific scoring is complete. Artifact verification, private release and cleanup "
        "are recorded separately in the execution receipt.",
        "",
        "Both arms start from the pinned V7 r1 treatment and see identical paired examples. "
        "Control uses CE; treatment adds 0.5 times label-aligned Jensen-Shannon divergence. "
        "Seeds 1601/1801 have disjoint canonical worlds. Aggregate inference uses BF16; "
        "FP32 GPU/CPU reference checks cover eight inputs per selected export.",
        "",
        "| Seed | Control paired worlds | Treatment paired worlds | Gain (95% interval) |",
        "| --- | ---: | ---: | ---: |",
    ]
    analyses = {}
    for rep in ("r1", "r2"):
        root = ROOT / rep
        analysis = json.loads((root / "analysis.json").read_text())
        raw = json.loads((root / "independent-raw-counts.json").read_text())["counts"]
        analyses[rep] = analysis
        c, t = (raw[arm]["test_paired"] for arm in ("control", "treatment"))
        primary = analysis["primary"]
        lower, upper = primary["interval"]
        seed = verify(root)["settings"]["model_seed"]
        lines.append(
            f"| {seed} | {c['complete_families']}/{c['families']} | "
            f"{t['complete_families']}/{t['families']} | "
            f"{primary['estimate'] * 100:.2f} pp "
            f"[{lower * 100:.2f}, {upper * 100:.2f}] |"
        )
    summary = {
        "study_identity": freeze["identity"],
        "primary_gains_positive_in_both_repetitions": all(
            a["engineering_gate"]["primary_gain_interval_excludes_zero"] for a in analyses.values()
        ),
        "all_engineering_checks_pass_in_both_repetitions": all(
            all(v for k, v in a["engineering_gate"].items() if k != "retention_unknown_margin")
            for a in analyses.values()
        ),
    }
    atomic_json(ROOT / "replication-summary.json", summary)
    lines += [
        "",
        "Intervals resample three-world label-rotation blocks within entity-count "
        "and edit-type strata. They describe world sampling for each trained run; "
        "there is no pooled training-seed interval or joint confidence guarantee.",
        "",
        "Both primary intervals positive: "
        f"{summary['primary_gains_positive_in_both_repetitions']}. "
        "All engineering checks pass: "
        f"{summary['all_engineering_checks_pass_in_both_repetitions']}.",
    ]
    for rep, analysis in analyses.items():
        root = ROOT / rep
        frozen = json.loads((root / "selection-freeze.json").read_text())
        evaluation = json.loads((root / "evaluation.json").read_text())
        lines += [
            "",
            f"## {rep}",
            "",
            "| Arm | Selected update | Temperature |",
            "| --- | ---: | ---: |",
        ]
        for arm in ("control", "treatment"):
            choice = frozen["choices"][arm]["calibrated_nll"]
            step = frozen["candidates"][choice]["step"]
            temp = frozen["calibrations"][choice]["temperature"]
            lines.append(f"| {arm} | {step} | {temp:.6f} |")
        lines += [
            "",
            "| Suite | Control accuracy | Treatment accuracy | Treatment complete families |",
            "| --- | ---: | ---: | ---: |",
        ]
        for split in TESTS:
            c, t = (evaluation["arms"][arm]["splits"][split] for arm in ("control", "treatment"))
            lines.append(
                f"| {split} | {c['raw']['accuracy']:.2%} | "
                f"{t['raw']['accuracy']:.2%} | {t['all_questions_in_family_correct']:.2%} |"
            )
        lines += ["", "Parent contrasts (estimate and separate world-sampling interval):", ""]
        for key, contrast in analysis["arms_against_parent"].items():
            lines.append(
                f"- {key} paired worlds versus parent: {contrast['estimate']:.2%}, "
                f"interval {contrast['interval']}."
            )
        lines.append(
            f"- Treatment base-world retention: {analysis['base_world_retention_against_parent']}."
        )
        for split, contrast in analysis["retention_against_parent"].items():
            lines.append(f"- Treatment {split} UNKNOWN retention: {contrast}.")
        lines += [
            "",
            "| Edit / person | Control both correct | Treatment both correct |",
            "| --- | ---: | ---: |",
        ]
        for key, t in analysis["entity_checks"]["treatment"].items():
            c = analysis["entity_checks"]["control"][key]
            lines.append(
                f"| {key} | {c['both_correct']}/{c['pairs']} | {t['both_correct']}/{t['pairs']} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is a synthetic, oracle-supervised decision task. Matched exposure isolates "
        "the added training objective within this pipeline; it does not prove an internal "
        "reasoning mechanism, unrestricted generalization or business usefulness. "
        "Base UNKNOWN cases can be identity edits; target behavior for known-answer removals "
        "must be read separately. Calibration changes confidence, not predicted classes. "
        "Correct complete paired worlds are stricter than question accuracy.",
        "",
        "If only paired CE improves, credit the paired data. If neither arm improves faithful "
        "edits, close this fine-tuning line; no coefficient sweep or extra seeds are authorized. "
        "All scheduled weights are retained privately; optimizer/RNG state is temporary. "
        "V1–V7 scientific results remain historical, rather than new confirmatory tests.",
        "",
        "See [protocol](../docs/V8_PROTOCOL.md), [execution](../docs/V8_EXECUTION.md), "
        "[journal](../RESEARCH_LOG.md) and runs/study-v8 for identities, raw counts and closure.",
        "",
    ]
    destination.write_text("\n".join(lines))
    return summary


def cleanup():
    ownership = json.loads((ROOT / "modal-ownership.json").read_text())
    if ownership["volume"] != "decision-model-lab-v8-position-runs":
        raise ValueError("Unexpected owned Volume")
    for rep in ("r1", "r2"):
        root = ROOT / rep
        if not json.loads((root / "final-audit.json").read_text())["valid"]:
            raise ValueError("Final audit is incomplete")
        for name, digest in json.loads((root / "retained-predictions.json").read_text()).items():
            if file_hash(Path(name)) != digest:
                raise ValueError("Retained prediction checksum differs")
        releases = json.loads((root / "huggingface-release.json").read_text())
        if set(releases) != {"control", "treatment"}:
            raise ValueError("Selected release inventory differs")
        for arm, record in releases.items():
            checked = record["remote_download_verification"]
            if not checked["verified"] or checked["device"] != "cpu" or checked["inputs"] != 8:
                raise ValueError("Fresh private release verification is incomplete")
            for directory in (
                Path("artifacts/v8-releases") / rep / arm,
                Path(record["verified_download_directory"]),
            ):
                release_inventory(directory)
                if (
                    file_hash(directory / "artifact-manifest.json")
                    != record["artifact_manifest_sha256"]
                ):
                    raise ValueError("Verified selected bundle differs")
    volume = modal.Volume.from_name(ownership["volume"], create_if_missing=False)
    volume.hydrate()
    if volume.object_id != ownership["volume_id"]:
        raise ValueError("Owned Volume ID differs")
    stopped = inspect_owned_apps(ownership["app_ids"])
    deleted = cli(["volume", "delete", ownership["volume"], "--yes"])
    if any(
        row["name"] == ownership["volume"] for row in json.loads(cli(["volume", "list", "--json"]))
    ):
        raise ValueError("Owned Volume remains")
    result = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "owned_volume_absent": True,
        "apps": inspect_owned_apps(ownership["app_ids"]),
        "zero_owned_tasks": True,
        "before_delete": stopped,
        "volume_delete_stdout": deleted,
        "unrelated_resources_modified": False,
    }
    atomic_json(ROOT / "cleanup.json", result)
    return result


def costs():
    if (ROOT / "billing-attribution.json").exists():
        raise FileExistsError("Preserve the prior billing observation")
    start = datetime.fromisoformat(
        json.loads((ROOT / "preflight.json").read_text())["snapshot_utc"]
    )
    end = datetime.now(UTC).date() + timedelta(days=1)
    resolution = "h" if (datetime.now(UTC) - start).days < 7 else "d"
    rows = json.loads(
        cli(
            [
                "billing",
                "report",
                "--start",
                start.date().isoformat(),
                "--end",
                end.isoformat(),
                "--resolution",
                resolution,
                "--show-resources",
                "--json",
            ]
        )
    )
    summary = json.loads(cli(["billing", "summary", "--json"]))
    atomic_json(ROOT / "billing-final-resources.json", rows)
    atomic_json(ROOT / "billing-final.json", summary)
    ownership = json.loads((ROOT / "modal-ownership.json").read_text())
    ids = (
        set(ownership["app_ids"])
        | {ownership["volume_id"]}
        | set(ownership["original_resource_ids"])
    )
    selected = [r for r in rows if r["object_id"] in ids]
    grouped = defaultdict(Decimal)
    for row in selected:
        grouped[row["resource"]] += Decimal(row["cost"])
    result = {
        "snapshot_utc": datetime.now(UTC).isoformat(),
        "owned_resource_metered_usd": str(sum(grouped.values(), Decimal(0))),
        "resources_usd": {k: str(v) for k, v in grouped.items()},
        "workspace_billed_usd": summary["billed_cost"],
        "scoped_rows": selected,
        "requested_interval_utc": {"start": start.date().isoformat(), "end": end.isoformat()},
        "resolution": resolution,
        "scope": "Owned V8 resources only; observed snapshot, not invoice",
        "raw_report_sha256": file_hash(ROOT / "billing-final-resources.json"),
    }
    atomic_json(ROOT / "billing-attribution.json", result)
    return {k: v for k, v in result.items() if k != "scoped_rows"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("analyse", "audit", "report", "historical", "cleanup", "costs")
    )
    parser.add_argument("--repetition", choices=("r1", "r2"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action in ("analyse", "audit"):
        if args.repetition is None:
            raise ValueError("A repetition is required")
        root = ROOT / args.repetition
        if args.action == "analyse":
            if (root / "analysis.json").exists():
                raise FileExistsError("Frozen analysis already exists")
            result = analyse(root)
            result = {"primary": result["primary"], "engineering_gate": result["engineering_gate"]}
        else:
            result = audit(root)
    else:
        result = globals()[args.action]()
    print(json.dumps({"action": args.action, "result": result}))


if __name__ == "__main__":
    main()
