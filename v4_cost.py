import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash


def fetch(arguments, destination):
    if destination.exists():
        raise FileExistsError("Preserve the previous billing observation before a new one")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "cloud",
            "modal",
            "billing",
            *arguments,
            "--profile",
            "amandogra2016",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    atomic_json(destination, value)
    return value


def main():
    root = Path("runs/study-v4")
    ownership = json.loads((root / "modal-ownership.json").read_text())
    ids = set(ownership["app_ids"]) | {ownership["volume_id"]}
    preflight = json.loads((root / "preflight.json").read_text())
    start = (
        datetime.fromisoformat(preflight["snapshot_utc"])
        .astimezone(UTC)
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )
    elapsed = datetime.now(UTC) - start
    if elapsed.total_seconds() < 0:
        raise ValueError("Billing start is in the future")
    resolution = "h" if elapsed.total_seconds() <= 7 * 86400 else "d"
    rows = fetch(
        [
            "report",
            "--start",
            start.date().isoformat(),
            "--resolution",
            resolution,
            "--show-resources",
            "--json",
        ],
        root / "billing-final-resources.json",
    )
    summary = fetch(["summary", "--json"], root / "billing-final.json")
    scoped = [row for row in rows if row["object_id"] in ids]
    resources = defaultdict(Decimal)
    for row in scoped:
        resources[row["resource"]] += Decimal(row["cost"])
    before = json.loads((root / "billing-before.json").read_text())
    record = {
        "snapshot_utc": datetime.now(UTC).isoformat(),
        "report_start_utc": start.isoformat(),
        "report_resolution": resolution,
        "owned_resources": ownership,
        "metered_owned_resources_usd": str(sum(resources.values(), Decimal(0))),
        "resources_usd": {k: str(v) for k, v in sorted(resources.items())},
        "workspace_billed_usd": summary["billed_cost"],
        "workspace_metered_delta_usd": str(
            Decimal(summary["metered_cost"]) - Decimal(before["metered_cost"])
        ),
        "raw_report_sha256": file_hash(root / "billing-final-resources.json"),
        "scoped_rows": scoped,
        "scope": "Owned V4 app/Volume rows only; observed snapshot, not a final invoice. "
        "Workspace aggregate delta is not this project's exact cost.",
    }
    atomic_json(root / "billing-attribution.json", record)
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("scoped_rows", "owned_resources")})
    )


if __name__ == "__main__":
    main()
