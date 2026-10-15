import json
import subprocess

from decision_lab.publication import atomic_json
from decision_lab.studies.v8.pipeline import execute
from decision_lab.studies.v8.protocol import ROOT


def stop_owned_call(repetition, name):
    ledger = json.loads((ROOT / "allocation-ledger.json").read_text())
    matches = [r for r in ledger if r["repetition"] == repetition and r["name"] == name]
    if len(matches) != 1 or matches[0]["status"] != "complete":
        raise ValueError("Only this controller's completed call may be stopped")
    app_id = matches[0]["app_id"]
    result = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "app",
            "stop",
            app_id,
            "--yes",
            "--profile",
            "amandogra2016",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "already stopped" not in (result.stdout + result.stderr).lower():
        raise RuntimeError("Owned detached app stop failed")
    atomic_json(
        ROOT / f"stop-{repetition}-{name}.json",
        {
            "app_id": app_id,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )


def main():
    execute(
        "r2-auth-recovery",
        [
            "modal",
            "run",
            "--detach",
            "--profile",
            "amandogra2016",
            "src/decision_lab/studies/v8/auth_recovery_app.py",
        ],
    )
    stop_owned_call("r2", "jwt-recovery")
    for action in ("selection", "test"):
        execute(
            f"r2-{action}",
            [
                "modal",
                "run",
                "--detach",
                "--profile",
                "amandogra2016",
                "src/decision_lab/studies/v8/app.py",
                "--repetition",
                "r2",
                "--action",
                action,
            ],
        )
        stop_owned_call("r2", action)
    execute("r2-records", ["python", "src/decision_lab/studies/v8/artifacts.py", "r2", "records"])
    execute(
        "r2-analysis",
        ["python", "src/decision_lab/studies/v8/ops.py", "analyse", "--repetition", "r2"],
    )
    execute("r2-raw-counts", ["python", "src/decision_lab/studies/v8/raw_audit.py", "r2"])
    execute("report", ["python", "src/decision_lab/studies/v8/ops.py", "report"])
    for rep in ("r1", "r2"):
        for action in ("candidates", "releases"):
            execute(
                f"{rep}-{action}",
                ["python", "src/decision_lab/studies/v8/artifacts.py", rep, action],
            )
        execute(
            f"{rep}-audit",
            ["python", "src/decision_lab/studies/v8/ops.py", "audit", "--repetition", rep],
        )
        execute(
            f"{rep}-publish", ["python", "src/decision_lab/studies/v8/artifacts.py", rep, "publish"]
        )
    for action in ("historical", "cleanup", "costs"):
        execute(action, ["python", "src/decision_lab/studies/v8/ops.py", action])
    with (ROOT / "finalization.log").open("x") as output:
        subprocess.run(
            [
                "uv",
                "run",
                "--extra",
                "train",
                "--extra",
                "cloud",
                "--extra",
                "analysis",
                "python",
                "src/decision_lab/studies/v8/finalize.py",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    main()
