from pathlib import Path

from decision_lab.publication import publish_private_releases, verify_hub_downloads


def main():
    receipt = Path("runs/study-v4/huggingface-release.json")
    names = {
        "control": "decision-lab-v4-smollm2-360m-control",
        "treatment": "decision-lab-v4-smollm2-360m-mixed-unknown",
    }
    publish_private_releases(
        Path("artifacts/v4-releases"), receipt, "axondendriteplus", model_names=names
    )
    verify_hub_downloads(receipt, Path("artifacts/v4-hub-verification"))


if __name__ == "__main__":
    main()
