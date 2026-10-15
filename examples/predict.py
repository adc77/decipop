import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from decision_lab.publication import release_inventory
from decision_lab.release import predict_release


def main():
    parser = argparse.ArgumentParser(
        description="Run a Premise research checkpoint on saved prompts"
    )
    parser.add_argument("--model", default="axondendriteplus/premise-v8-r2-consistency")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("prompts.json"))
    args = parser.parse_args()
    directory = Path(snapshot_download(args.model, revision=args.revision))
    release_inventory(directory)
    prompts = json.loads(args.input.read_text())
    print(json.dumps(predict_release(directory, prompts), indent=2))


if __name__ == "__main__":
    main()
