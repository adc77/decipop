import json
import time
from pathlib import Path

from decision_lab.studies.v6.finish import execute, main


def continue_study():
    root = Path("runs/study-v6")
    deadline = time.monotonic() + 1600
    while True:
        ledger = json.loads((root / "allocation-ledger.json").read_text())
        recovery = [r for r in ledger if r["name"] == "recovery-control"]
        if recovery and recovery[0]["status"] == "failed":
            raise RuntimeError("Reserved recovery failed; no further GPU retry is authorized")
        receipt = root / "training-call-control-1.json"
        if recovery and recovery[0]["status"] == "complete" and receipt.exists():
            record = json.loads(receipt.read_text())
            if record["recovered_from_call"] != "control-1" or record["start_step"] != 250:
                raise ValueError("Recovery receipt differs")
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Recovery did not complete within its reserved allowance")
        time.sleep(10)
    execute(
        "training-resumed",
        [
            "modal",
            "run",
            "--profile",
            "amandogra2016",
            "src/decision_lab/studies/v6/app.py",
            "--action",
            "train",
        ],
    )
    main()


if __name__ == "__main__":
    continue_study()
