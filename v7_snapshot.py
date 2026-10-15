import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from publish_research_archive import scan, scan_archive


def snapshot(name):
    destination = Path("artifacts/v7-source-snapshot")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"source-{name}.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    files = {
        p
        for folder in (
            "src",
            "v4_lab",
            "v5_lab",
            "v6_lab",
            "v7_lab",
            "tests",
            "configs",
            "docs",
            "reports",
        )
        for p in Path(folder).rglob("*")
        if p.is_file() and not set(p.parts) & {"__pycache__", ".cache", ".DS_Store"}
    }
    files.update(Path().glob("*.py"))
    files.update(
        Path(p)
        for p in (
            "AGENTS.md",
            "README.md",
            "RESEARCH_LOG.md",
            "PLAN.md",
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
            "NOTICE",
            ".gitignore",
            ".gitattributes",
        )
    )
    inventory = {}
    for path in sorted(files):
        if path.is_symlink():
            raise ValueError(f"Unexpected source symlink: {path}")
        scan(path.read_bytes(), str(path))
        inventory[str(path)] = file_hash(path)
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(files):
            stream.add(path, arcname=str(path), recursive=False)
    observed = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if not member.isfile() or member.name in observed:
                raise ValueError("Duplicate or unexpected archive member")
            with stream.extractfile(member) as payload:
                observed[member.name] = hashlib.file_digest(payload, "sha256").hexdigest()
    if observed != inventory:
        raise ValueError("Independent archive checksums differ")
    scan_archive(archive)
    record = {
        "files": inventory,
        "archive_sha256": file_hash(archive),
        "archive_members_verified": True,
        "credential_pattern_scan_passed": True,
    }
    atomic_json(destination / f"manifest-{name}.json", record)
    return {"archive": str(archive), "sha256": record["archive_sha256"], "members": len(inventory)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", choices=("before-execution", "before-pilot-recovery", "final-with-reports")
    )
    print(json.dumps(snapshot(parser.parse_args().name)))
