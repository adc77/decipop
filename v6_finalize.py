from v6_finish import execute


def main():
    stages = [
        ("candidates-resumed", ["python", "v6_retain.py", "candidates"]),
        ("releases", ["python", "v6_retain.py", "releases"]),
        ("audit", ["python", "v6_ops.py", "audit"]),
        ("private-releases", ["python", "v6_release.py"]),
        ("cleanup", ["python", "v6_ops.py", "cleanup"]),
        ("billing", ["python", "v6_ops.py", "costs"]),
    ]
    for stage, command in stages:
        execute(stage, command)
    print(
        "V6 verified storage, publication and cleanup completed; final archive pending", flush=True
    )


if __name__ == "__main__":
    main()
