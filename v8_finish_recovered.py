import json
import subprocess

from decision_lab.publication import atomic_json
from v8_lab.protocol import ROOT
from v8_pipeline import execute


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
        ["modal", "run", "--detach", "--profile", "amandogra2016", "v8_auth_recovery_app.py"],
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
                "v8_app.py",
                "--repetition",
                "r2",
                "--action",
                action,
            ],
        )
        stop_owned_call("r2", action)
    execute("r2-records", ["python", "v8_artifacts.py", "r2", "records"])
    execute("r2-analysis", ["python", "v8_ops.py", "analyse", "--repetition", "r2"])
    execute("r2-raw-counts", ["python", "v8_raw_audit.py", "r2"])
    execute("report", ["python", "v8_ops.py", "report"])
    for rep in ("r1", "r2"):
        for action in ("candidates", "releases"):
            execute(f"{rep}-{action}", ["python", "v8_artifacts.py", rep, action])
        execute(f"{rep}-audit", ["python", "v8_ops.py", "audit", "--repetition", rep])
        execute(f"{rep}-publish", ["python", "v8_artifacts.py", rep, "publish"])
    for action in ("historical", "cleanup", "costs"):
        execute(action, ["python", "v8_ops.py", action])
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
                "v8_finalize.py",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    main()
