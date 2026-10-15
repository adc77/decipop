import argparse
import json
import tarfile
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    if args.name not in ("before-diagnostics", "before-training", "final-with-reports"):
        raise ValueError("Unknown snapshot stage")
    destination = Path("artifacts/v4-source-snapshot")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"source-{args.name}.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    files = {
        p
        for folder in ("src", "src/decision_lab/studies/v4", "tests", "configs", "docs", "reports")
        for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml", ".md", ".json", ".svg", ".png", ".txt"}
    }
    files.update(Path.cwd().glob("v4*.py"))
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
        )
    )
    files = sorted({p.relative_to(Path.cwd()) if p.is_absolute() else p for p in files})
    inventory = {str(p): file_hash(p) for p in files}
    with tarfile.open(archive, "w:gz") as stream:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Symlink in source snapshot: {path}")
            stream.add(path, arcname=str(path), recursive=False)
    with tarfile.open(archive, "r:gz") as stream:
        import hashlib

        for member in stream.getmembers():
            data = stream.extractfile(member)
            if data is None or hashlib.sha256(data.read()).hexdigest() != inventory[member.name]:
                raise ValueError("Archive member hash mismatch")
    record = {
        "files": inventory,
        "archive_sha256": file_hash(archive),
        "archive_members_verified": True,
    }
    atomic_json(destination / f"manifest-{args.name}.json", record)
    print(
        json.dumps(
            {"archive": str(archive), "sha256": record["archive_sha256"], "members": len(inventory)}
        )
    )


if __name__ == "__main__":
    main()
