import argparse
import json
from pathlib import Path

from decision_lab.publication import (
    annotate_counterfactual_results,
    download_modal_releases,
    publish_private_releases,
    verify_hub_downloads,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "publish", "verify"])
    parser.add_argument("--study", default="study-v1")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--source", type=Path, default=Path("artifacts/releases"))
    parser.add_argument(
        "--verification-output", type=Path, default=Path("artifacts/hub-verification")
    )
    args = parser.parse_args()
    local = Path("runs") / args.study
    verification = json.loads((local / "artifact-verification.json").read_text())
    if set(verification) != {"token", "head", "scratch"} or not all(
        record["verified"] for record in verification.values()
    ):
        raise ValueError("All three model bundles must pass GPU reload verification first")
    receipt = local / "huggingface-release.json"
    if args.action == "download":
        import modal

        from decision_lab.model_config import load_pilot_config

        pilot = load_pilot_config(Path("configs/pilot.yaml"))
        download_modal_releases(modal.Volume.from_name(pilot.volume_name), args.study, args.source)
    elif args.action == "publish":
        annotate_counterfactual_results(args.source, local)
        publish_private_releases(args.source, receipt, args.namespace)
    else:
        verify_hub_downloads(receipt, args.verification_output)


if __name__ == "__main__":
    main()
