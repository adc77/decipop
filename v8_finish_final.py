import json
import subprocess
import time

from v8_finish_recovered import stop_owned_call
from v8_lab.protocol import ROOT
from v8_pipeline import PREFIX, execute


def main():
    execute(
        "r2-final-evaluation-submit",
        ["modal", "run", "--detach", "--profile", "amandogra2016", "v8_final_eval_app.py"],
    )
    deadline = time.monotonic() + 3600
    log = ROOT / "final-evaluation-collection.log"
    with log.open("x") as output:
        while not (ROOT / "final-evaluation-collected.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Collect the persisted call ID without resubmitting it")
            subprocess.run(
                PREFIX + ["python", "v8_collect_final.py"],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
            output.flush()
            if not (ROOT / "final-evaluation-collected.json").exists():
                time.sleep(20)
    stop_owned_call("r2", "test")
    for stage, command in (
        ("r2-records", ["python", "v8_artifacts.py", "r2", "records"]),
        ("r2-analysis", ["python", "v8_ops.py", "analyse", "--repetition", "r2"]),
        ("r2-raw-counts", ["python", "v8_raw_audit.py", "r2"]),
        ("report", ["python", "v8_ops.py", "report"]),
    ):
        execute(stage, command)
    for rep in ("r1", "r2"):
        for action in ("candidates", "releases"):
            execute(f"{rep}-{action}", ["python", "v8_artifacts.py", rep, action])
        execute(f"{rep}-audit", ["python", "v8_ops.py", "audit", "--repetition", rep])
        execute(f"{rep}-publish", ["python", "v8_artifacts.py", rep, "publish"])
    for action in ("historical", "cleanup", "costs"):
        execute(action, ["python", "v8_ops.py", action])
    with (ROOT / "finalization.log").open("x") as output:
        subprocess.run(
            PREFIX + ["python", "v8_finalize.py"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(json.dumps({"verified_closed": True}), flush=True)


if __name__ == "__main__":
    main()
