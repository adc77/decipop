import argparse
import json
from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import generate_dataset
from decision_lab.data.validate import validate_dataset


def main():
    parser = argparse.ArgumentParser(prog="decision-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    baseline = commands.add_parser("baseline")
    baseline.add_argument("--dataset", type=Path, required=True)
    baseline.add_argument("--seed", type=int, required=True)
    audit = commands.add_parser("audit-tokens")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--pilot-config", type=Path, required=True)
    audit.add_argument("--scratch-config", type=Path, required=True)
    preflight = commands.add_parser("pilot-preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--remaining-usd", type=float, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_dataset(load_dataset_config(args.config), args.output)
    elif args.command == "validate":
        result = validate_dataset(args.dataset)
    elif args.command == "baseline":
        from decision_lab.baseline import lexical_baseline

        result = lexical_baseline(args.dataset, args.seed)
    elif args.command == "audit-tokens":
        from decision_lab.model_config import load_pilot_config, load_scratch_config
        from decision_lab.tokenization import audit_tokens

        result = audit_tokens(
            args.dataset,
            args.output,
            load_pilot_config(args.pilot_config),
            load_scratch_config(args.scratch_config),
        )
    else:
        from decision_lab.budget import pilot_preflight
        from decision_lab.model_config import load_pilot_config

        result = pilot_preflight(load_pilot_config(args.config), args.remaining_usd)
    print(json.dumps(result, indent=2))
