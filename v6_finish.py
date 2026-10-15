import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json

ROOT = Path("runs/study-v6")
PREFIX = ["uv", "run", "--extra", "train", "--extra", "cloud", "--extra", "analysis"]


def journal(message):
    with Path("RESEARCH_LOG.md").open("a") as output:
        output.write(
            f"\n### {datetime.now(UTC).isoformat()}: V6 execution milestone\n\n{message}\n"
        )


def execute(stage, command):
    started = datetime.now(UTC).isoformat()
    atomic_json(
        ROOT / "finish-status.json", {"stage": stage, "status": "started", "at_utc": started}
    )
    print(json.dumps({"stage": stage, "status": "started"}), flush=True)
    environment = dict(os.environ)
    environment["HF_XET_FIXED_UPLOAD_CONCURRENCY"] = "4"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    with (ROOT / f"finish-{stage}.log").open("x") as log:
        try:
            subprocess.run(
                PREFIX + command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
        except BaseException:
            atomic_json(
                ROOT / "finish-status.json",
                {"stage": stage, "status": "failed", "at_utc": datetime.now(UTC).isoformat()},
            )
            raise
    journal(
        f"Completed **{stage}**; exact command output is retained in "
        f"`runs/study-v6/finish-{stage}.log`. "
        "No Git commit or push is part of this controller."
    )
    atomic_json(
        ROOT / "finish-status.json",
        {"stage": stage, "status": "complete", "at_utc": datetime.now(UTC).isoformat()},
    )


def main():
    deadline = time.monotonic() + 7200
    atomic_json(
        ROOT / "finish-status.json",
        {"stage": "training", "status": "waiting", "at_utc": datetime.now(UTC).isoformat()},
    )
    while True:
        ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
        if any(r["status"] == "failed" for r in ledger):
            failed = {r["name"] for r in ledger if r["status"] == "failed"}
            receipt = ROOT / "training-call-control-1.json"
            acknowledged = (
                receipt.exists()
                and json.loads(receipt.read_text()).get("recovered_from_call") == "control-1"
            )
            completed = any(
                r["name"] == "recovery-control" and r["status"] == "complete" for r in ledger
            )
            if failed != {"control-1"} or not acknowledged or not completed:
                raise RuntimeError(
                    "GPU call failed; preserve reservations and review before recovery"
                )
        paths = [ROOT / f"training-call-{arm}-3.json" for arm in ("control", "treatment")]
        if all(p.exists() and json.loads(p.read_text())["complete"] for p in paths):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Training exceeded the controller wait allowance")
        time.sleep(20)
    journal(
        "Both arms completed the frozen training exposure; "
        "checkpoint selection and untouched final tests follow."
    )
    stages = [
        (
            "selection",
            ["modal", "run", "--profile", "amandogra2016", "v6_app.py", "--action", "selection"],
        ),
        ("test", ["modal", "run", "--profile", "amandogra2016", "v6_app.py", "--action", "test"]),
        ("records", ["python", "v6_retain.py", "records"]),
        ("analysis", ["python", "v6_ops.py", "analyse"]),
        ("report", ["python", "v6_ops.py", "report"]),
        ("candidates", ["python", "v6_retain.py", "candidates"]),
        ("releases", ["python", "v6_retain.py", "releases"]),
        ("audit", ["python", "v6_ops.py", "audit"]),
        ("private-releases", ["python", "v6_release.py"]),
        ("cleanup", ["python", "v6_ops.py", "cleanup"]),
        ("billing", ["python", "v6_ops.py", "costs"]),
    ]
    for stage, command in stages:
        execute(stage, command)
    print(
        json.dumps(
            {
                "status": (
                    "Scientific execution, retention, private verification and cleanup complete; "
                    "final documentation/archive pending"
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
