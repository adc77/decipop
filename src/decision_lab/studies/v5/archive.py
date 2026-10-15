import hashlib
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.publishing.legacy_archive import scan, scan_archive
from decision_lab.studies.v5.protocol import verify
from decision_lab.training.checkpoint import file_hash

ROOT = Path("runs/study-v5")
DESTINATION = Path("artifacts/v5-source-snapshot")


def close():
    protocol = verify(ROOT)
    for name, field in (
        ("uv.lock", "lock_sha256"),
        ("docs/V5_PROTOCOL.md", "protocol_markdown_sha256"),
    ):
        if file_hash(Path(name)) != protocol[field]:
            raise ValueError(f"Frozen auxiliary file changed: {name}")
    audit = json.loads((ROOT / "final-audit.json").read_text())
    if not audit["valid"] or audit["analysis_sha256"] != file_hash(ROOT / "analysis.json"):
        raise ValueError("Verified analysis differs")
    cleanup = json.loads((ROOT / "cleanup.json").read_text())
    if not cleanup["owned_volume_absent"] or not cleanup["zero_owned_tasks"]:
        raise ValueError("Cleanup evidence is incomplete")
    if any(row["state"] != "stopped" or row["tasks"] for row in cleanup["apps"]):
        raise ValueError("Owned app evidence is not stopped with zero tasks")
    evidence = Path("reports/v5-evidence")
    inventory = json.loads((evidence / "inventory.json").read_text())
    if set(inventory) != {p.name for p in evidence.iterdir() if p.name != "inventory.json"}:
        raise ValueError("Compact evidence inventory differs")
    for name, digest in inventory.items():
        if file_hash(evidence / name) != digest:
            raise ValueError(f"Compact evidence checksum differs: {name}")
    tests = ROOT / "cpu-tests-validated.log"
    if "99 passed" not in tests.read_text():
        raise ValueError("Validated CPU evidence is missing")
    archive = DESTINATION / "source-final-with-reports.tar.gz"
    manifest = DESTINATION / "manifest-final-with-reports.json"
    receipt = ROOT / "closure-receipt.json"
    if any(path.exists() for path in (archive, manifest, receipt)):
        raise FileExistsError("Final closure already exists")
    files = {
        p
        for directory in (
            "src",
            "src/decision_lab/studies/v5",
            "tests",
            "configs",
            "docs",
            "reports",
        )
        for p in Path(directory).rglob("*")
        if p.is_file() and not set(p.parts) & {"__pycache__", ".cache", ".DS_Store"}
    }
    files.update(Path().glob("*.py"))
    files.update(
        Path(name)
        for name in (
            "AGENTS.md",
            "README.md",
            "RESEARCH_LOG.md",
            "PLAN.md",
            "LICENSE",
            "NOTICE",
            "pyproject.toml",
            "uv.lock",
            ".gitignore",
            ".gitattributes",
        )
    )
    hashes = {}
    for path in sorted(files):
        if any(part in {".env", ".modal.toml", "secrets", "credentials"} for part in path.parts):
            raise ValueError(f"Credential filename: {path}")
        scan(path.read_bytes(), str(path))
        hashes[str(path)] = file_hash(path)
    with tarfile.open(archive, "w:gz", dereference=True) as output:
        for path in sorted(files):
            output.add(path, arcname=str(path), recursive=False)
    observed = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if not member.isfile() or member.name in observed:
                raise ValueError("Unexpected or duplicate source archive member")
            with source.extractfile(member) as payload:
                observed[member.name] = hashlib.file_digest(payload, "sha256").hexdigest()
    if observed != hashes:
        raise ValueError("Independent source archive checksums differ")
    scan_archive(archive)
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": hashes,
        "archive": str(archive),
        "archive_sha256": file_hash(archive),
        "archive_members_verified": True,
        "credential_pattern_scan_passed": True,
        "protocol_identity": protocol["identity"],
        "scope": "Source, reports and compact evidence; raw inputs/logits remain in runs/study-v5.",
    }
    atomic_json(manifest, result)
    billing = json.loads((ROOT / "billing-attribution.json").read_text())
    closure = {
        "closed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_identity": protocol["identity"],
        "analysis_sha256": audit["analysis_sha256"],
        "prediction_rows_verified": audit["prediction_rows_verified"],
        "cpu_tests_passed": 99,
        "cpu_test_log_sha256": file_hash(tests),
        "historical_scientific_build_files_unchanged": audit[
            "historical_scientific_build_files_unchanged"
        ],
        "source_archive_sha256": result["archive_sha256"],
        "source_archive_members": len(hashes),
        "source_manifest_sha256": file_hash(manifest),
        "evidence_inventory_sha256": file_hash(evidence / "inventory.json"),
        "cleanup_sha256": file_hash(ROOT / "cleanup.json"),
        "observed_owned_metered_usd": billing["owned_resource_metered_usd"],
        "billing_snapshot_utc": billing["snapshot_utc"],
        "workspace_billed_usd": billing["workspace_billed_usd"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_committed_or_pushed": False,
        "new_model_uploaded": False,
        "next_training_submitted": False,
    }
    atomic_json(receipt, closure)
    return closure


if __name__ == "__main__":
    print(json.dumps(close()))
