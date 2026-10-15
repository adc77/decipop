import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from v8_lab.protocol import ROOT, verify_study

PREFIX = ["uv", "run", "--extra", "train", "--extra", "cloud", "--extra", "analysis"]


def execute(stage, command):
    atomic_json(
        ROOT / "finish-status.json",
        {
            "stage": stage,
            "status": "started",
            "at_utc": datetime.now(UTC).isoformat(),
        },
    )
    print(json.dumps({"stage": stage, "status": "started"}), flush=True)
    environment = dict(os.environ)
    environment["HF_XET_FIXED_UPLOAD_CONCURRENCY"] = "4"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    with (ROOT / f"finish-{stage}.log").open("x") as log:
        try:
            subprocess.run(
                PREFIX + command, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment
            )
        except BaseException:
            atomic_json(
                ROOT / "finish-status.json",
                {
                    "stage": stage,
                    "status": "failed",
                    "at_utc": datetime.now(UTC).isoformat(),
                },
            )
            raise
    with Path("RESEARCH_LOG.md").open("a") as output:
        output.write(
            f"\n### {datetime.now(UTC).isoformat()}: V8 execution milestone\n\n"
            f"Completed **{stage}**. Exact output is retained in "
            f"`runs/study-v8/finish-{stage}.log`. "
            "No Git commit or push is part of this controller.\n"
        )
    atomic_json(
        ROOT / "finish-status.json",
        {
            "stage": stage,
            "status": "complete",
            "at_utc": datetime.now(UTC).isoformat(),
        },
    )


def main():
    verify_study()
    for repetition in ("r1", "r2"):
        execute(
            f"{repetition}-pilot",
            [
                "modal",
                "run",
                "--profile",
                "amandogra2016",
                "v8_app.py",
                "--repetition",
                repetition,
                "--action",
                "position-revalidation",
            ],
        )
    for repetition in ("r1", "r2"):
        for action in ("train", "selection", "test"):
            execute(
                f"{repetition}-{action}",
                [
                    "modal",
                    "run",
                    "--profile",
                    "amandogra2016",
                    "v8_app.py",
                    "--repetition",
                    repetition,
                    "--action",
                    action,
                ],
            )
        execute(f"{repetition}-records", ["python", "v8_artifacts.py", repetition, "records"])
        execute(
            f"{repetition}-analysis", ["python", "v8_ops.py", "analyse", "--repetition", repetition]
        )
        execute(f"{repetition}-raw-counts", ["python", "v8_raw_audit.py", repetition])
    execute("report", ["python", "v8_ops.py", "report"])
    for repetition in ("r1", "r2"):
        for action in ("candidates", "releases"):
            execute(f"{repetition}-{action}", ["python", "v8_artifacts.py", repetition, action])
        execute(f"{repetition}-audit", ["python", "v8_ops.py", "audit", "--repetition", repetition])
        execute(f"{repetition}-publish", ["python", "v8_artifacts.py", repetition, "publish"])
    for action in ("historical", "cleanup", "costs"):
        execute(action, ["python", "v8_ops.py", action])
    print(
        "V8 execution, retention and cleanup complete; final documentation/archive pending",
        flush=True,
    )


if __name__ == "__main__":
    main()
