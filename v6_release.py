from pathlib import Path

from decision_lab.publication import publish_private_releases, verify_hub_downloads


def main():
    receipt = Path("runs/study-v6/huggingface-release.json")
    names = {
        "control": "decision-lab-v6-smollm2-360m-control",
        "treatment": "decision-lab-v6-smollm2-360m-multi-entity",
    }
    publish_private_releases(
        Path("artifacts/v6-releases"), receipt, "axondendriteplus", model_names=names
    )
    verify_hub_downloads(receipt, Path("artifacts/v6-hub-verification"))


if __name__ == "__main__":
    main()
