from decision_lab.studies.v7.pipeline import execute


def main():
    execute(
        "r1-pilot-recovery",
        [
            "modal",
            "run",
            "--profile",
            "amandogra2016",
            "src/decision_lab/studies/v7/app.py",
            "--repetition",
            "r1",
            "--action",
            "pilot-recovery",
        ],
    )
    execute(
        "r2-pilot",
        [
            "modal",
            "run",
            "--profile",
            "amandogra2016",
            "src/decision_lab/studies/v7/app.py",
            "--repetition",
            "r2",
            "--action",
            "pilot",
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
                    "src/decision_lab/studies/v7/app.py",
                    "--repetition",
                    repetition,
                    "--action",
                    action,
                ],
            )
        execute(
            f"{repetition}-records",
            ["python", "src/decision_lab/studies/v7/artifacts.py", repetition, "records"],
        )
        execute(
            f"{repetition}-analysis",
            ["python", "src/decision_lab/studies/v7/ops.py", "analyse", "--repetition", repetition],
        )
    execute("report", ["python", "src/decision_lab/studies/v7/ops.py", "report"])
    for repetition in ("r1", "r2"):
        for action in ("candidates", "releases"):
            execute(
                f"{repetition}-{action}",
                ["python", "src/decision_lab/studies/v7/artifacts.py", repetition, action],
            )
        execute(
            f"{repetition}-audit",
            ["python", "src/decision_lab/studies/v7/ops.py", "audit", "--repetition", repetition],
        )
        execute(
            f"{repetition}-publish",
            ["python", "src/decision_lab/studies/v7/artifacts.py", repetition, "publish"],
        )
    for action in ("historical", "cleanup", "costs"):
        execute(action, ["python", "src/decision_lab/studies/v7/ops.py", action])
    print("V7 corrected execution and cleanup complete; final archive pending", flush=True)


if __name__ == "__main__":
    main()
