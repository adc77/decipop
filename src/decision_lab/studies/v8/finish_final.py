import json
import subprocess
import time

from decision_lab.studies.v8.finish_recovered import stop_owned_call
from decision_lab.studies.v8.pipeline import PREFIX, execute
from decision_lab.studies.v8.protocol import ROOT


def main():
    execute(
        "r2-final-evaluation-submit",
        [
            "modal",
            "run",
            "--detach",
            "--profile",
            "amandogra2016",
            "src/decision_lab/studies/v8/final_eval_app.py",
        ],
    )
    deadline = time.monotonic() + 3600
    log = ROOT / "final-evaluation-collection.log"
    with log.open("x") as output:
        while not (ROOT / "final-evaluation-collected.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Collect the persisted call ID without resubmitting it")
            subprocess.run(
                PREFIX + ["python", "src/decision_lab/studies/v8/collect_final.py"],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
            output.flush()
            if not (ROOT / "final-evaluation-collected.json").exists():
                time.sleep(20)
    stop_owned_call("r2", "test")
    for stage, command in (
        ("r2-records", ["python", "src/decision_lab/studies/v8/artifacts.py", "r2", "records"]),
        (
            "r2-analysis",
            ["python", "src/decision_lab/studies/v8/ops.py", "analyse", "--repetition", "r2"],
        ),
        ("r2-raw-counts", ["python", "src/decision_lab/studies/v8/raw_audit.py", "r2"]),
        ("report", ["python", "src/decision_lab/studies/v8/ops.py", "report"]),
    ):
        execute(stage, command)
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
            PREFIX + ["python", "src/decision_lab/studies/v8/finalize.py"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(json.dumps({"verified_closed": True}), flush=True)


if __name__ == "__main__":
    main()
