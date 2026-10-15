from decision_lab.studies.v6.finish import execute


def main():
    stages = [
        ("candidates-resumed", ["python", "src/decision_lab/studies/v6/retain.py", "candidates"]),
        ("releases", ["python", "src/decision_lab/studies/v6/retain.py", "releases"]),
        ("audit", ["python", "src/decision_lab/studies/v6/ops.py", "audit"]),
        ("private-releases", ["python", "src/decision_lab/studies/v6/release.py"]),
        ("cleanup", ["python", "src/decision_lab/studies/v6/ops.py", "cleanup"]),
        ("billing", ["python", "src/decision_lab/studies/v6/ops.py", "costs"]),
    ]
    for stage, command in stages:
        execute(stage, command)
    print(
        "V6 verified storage, publication and cleanup completed; final archive pending", flush=True
    )


if __name__ == "__main__":
    main()
