import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from decision_lab.publication import release_inventory
from decision_lab.release import predict_release


def main():
    parser = argparse.ArgumentParser(
        description="Run a decipop research checkpoint on saved prompts"
    )
    parser.add_argument("--model", default="axondendriteplus/decipop-v8-r2-consistency")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("prompts.json"))
    args = parser.parse_args()
    identity = hashlib.sha256(f"{args.model}@{args.revision}".encode()).hexdigest()[:20]
    directory = Path(
        snapshot_download(
            args.model, revision=args.revision, local_dir=Path(".cache/decipop") / identity
        )
    )
    release_inventory(directory)
    prompts = json.loads(args.input.read_text())
    print(json.dumps(predict_release(directory, prompts), indent=2))


if __name__ == "__main__":
    main()
