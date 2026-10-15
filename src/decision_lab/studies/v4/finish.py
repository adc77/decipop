import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json

ROOT = Path("runs/study-v4")
PREFIX = ["uv", "run", "--extra", "train", "--extra", "cloud", "--extra", "analysis"]


def journal(message):
    now = datetime.now(UTC).isoformat()
    path = Path("RESEARCH_LOG.md")
    content = path.read_text()
    marker = "## How to update this journal"
    if content.count(marker) != 1:
        raise ValueError("Journal insertion point is ambiguous")
    entry = f"### {now}: V4 execution milestone\n\n{message}\n\n"
    path.write_text(content.replace(marker, entry + marker))


def execute(stage, arguments):
    atomic_json(
        ROOT / "finish-status.json",
        {"stage": stage, "status": "started", "at_utc": datetime.now(UTC).isoformat()},
    )
    print(json.dumps({"stage": stage, "status": "started"}), flush=True)
    subprocess.run(PREFIX + arguments, check=True)
    journal(
        f"Completed **{stage}**. The exact command and output are retained in "
        "`runs/study-v4/finish.log`; stage-specific evidence remains in the study directory. "
        "No Git commit or push is part of this controller."
    )
    atomic_json(
        ROOT / "finish-status.json",
        {"stage": stage, "status": "complete", "at_utc": datetime.now(UTC).isoformat()},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-timeout-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60 or args.wait_timeout_seconds <= 0:
        raise ValueError("Invalid polling limits")
    atomic_json(
        ROOT / "finish-status.json",
        {"stage": "training", "status": "waiting", "at_utc": datetime.now(UTC).isoformat()},
    )
    deadline = time.monotonic() + args.wait_timeout_seconds
    while True:
        ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
        if any(r["status"] == "failed" for r in ledger):
            raise RuntimeError("A call failed; stop automation and review the retained ledger")
        paths = [ROOT / f"training-call-{arm}-3.json" for arm in ("control", "treatment")]
        if all(p.exists() and json.loads(p.read_text())["complete"] for p in paths):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Training did not complete within the controller wait allowance")
        time.sleep(args.poll_seconds)
    journal(
        "Both training arms completed the frozen exposure. "
        "Per-call records and the invocation ledger confirm completion; "
        "checkpoint selection and untouched final tests follow."
    )
    operations = [
        (
            "checkpoint selection",
            ["modal", "run", "src/decision_lab/studies/v4/app.py", "--action", "selection"],
        ),
        (
            "final test scoring and GPU export verification",
            ["modal", "run", "src/decision_lab/studies/v4/app.py", "--action", "test"],
        ),
        ("raw record retention", ["python", "src/decision_lab/studies/v4/retain.py", "records"]),
        (
            "complete candidate retention",
            ["python", "src/decision_lab/studies/v4/retain.py", "candidates"],
        ),
        (
            "selected release retention",
            ["python", "src/decision_lab/studies/v4/retain.py", "releases"],
        ),
        (
            "frozen statistical analysis and plots",
            ["python", "src/decision_lab/studies/v4/report.py", "results"],
        ),
        ("numerical learning walkthrough", ["python", "src/decision_lab/studies/v4/learning.py"]),
        ("scientific and retention audit", ["python", "src/decision_lab/studies/v4/audit.py"]),
        (
            "private publication and pinned CPU verification",
            ["python", "src/decision_lab/studies/v4/release.py"],
        ),
    ]
    for stage, command in operations:
        execute(stage, command)
    atomic_json(
        ROOT / "finish-status.json",
        {
            "stage": "verification complete; cleanup pending",
            "status": "complete",
            "at_utc": datetime.now(UTC).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
