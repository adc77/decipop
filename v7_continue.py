from v7_pipeline import execute


def main():
    execute(
        "r1-pilot-recovery",
        [
            "modal",
            "run",
            "--profile",
            "amandogra2016",
            "v7_app.py",
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
            "v7_app.py",
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
                    "v7_app.py",
                    "--repetition",
                    repetition,
                    "--action",
                    action,
                ],
            )
        execute(f"{repetition}-records", ["python", "v7_artifacts.py", repetition, "records"])
        execute(
            f"{repetition}-analysis", ["python", "v7_ops.py", "analyse", "--repetition", repetition]
        )
    execute("report", ["python", "v7_ops.py", "report"])
    for repetition in ("r1", "r2"):
        for action in ("candidates", "releases"):
            execute(f"{repetition}-{action}", ["python", "v7_artifacts.py", repetition, action])
        execute(f"{repetition}-audit", ["python", "v7_ops.py", "audit", "--repetition", repetition])
        execute(f"{repetition}-publish", ["python", "v7_artifacts.py", repetition, "publish"])
    for action in ("historical", "cleanup", "costs"):
        execute(action, ["python", "v7_ops.py", action])
    print("V7 corrected execution and cleanup complete; final archive pending", flush=True)


if __name__ == "__main__":
    main()
