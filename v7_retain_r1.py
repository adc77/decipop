import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from v7_lab.protocol import ROOT, verify
from v7_pipeline import PREFIX


def main():
    root = ROOT / "r1"
    protocol = verify(root)
    receipt = root / "early-retention.json"
    if receipt.exists() or not (root / "analysis.json").exists():
        raise ValueError(
            "Completed r1 scientific execution and a new retention receipt are required"
        )
    record = {"protocol_identity": protocol["identity"], "gpu_calls_added": 0, "stages": {}}
    environment = dict(os.environ)
    environment["HF_XET_FIXED_UPLOAD_CONCURRENCY"] = "4"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    for stage in ("candidates", "releases", "audit", "publish"):
        command = (
            ["python", "v7_ops.py", "audit", "--repetition", "r1"]
            if stage == "audit"
            else ["python", "v7_artifacts.py", "r1", stage]
        )
        record["stages"][stage] = {"status": "started", "at_utc": datetime.now(UTC).isoformat()}
        atomic_json(receipt, record)
        print(json.dumps({"early_r1_retention": stage, "status": "started"}), flush=True)
        with (ROOT / f"retention-r1-{stage}.log").open("x") as output:
            try:
                subprocess.run(
                    PREFIX + command,
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            except BaseException:
                record["stages"][stage] = {
                    "status": "failed",
                    "at_utc": datetime.now(UTC).isoformat(),
                }
                atomic_json(receipt, record)
                raise
        record["stages"][stage] = {"status": "complete", "at_utc": datetime.now(UTC).isoformat()}
        atomic_json(receipt, record)
        with Path("RESEARCH_LOG.md").open("a") as journal:
            journal.write(
                f"\n### {datetime.now(UTC).isoformat()}: V7 r1 retention milestone\n\n"
                f"Completed **r1 {stage}** while the independent r2 GPU execution proceeds. "
                f"Exact output is retained in `runs/study-v7/retention-r1-{stage}.log`. "
                "No GPU scoring/training is repeated. The main controller will recheck the "
                "idempotent retention stages before global cleanup; Git remains local.\n"
            )
    print("R1 private retention and verification complete", flush=True)


if __name__ == "__main__":
    main()
