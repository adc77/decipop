import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

ROOT = Path.cwd()
DESTINATION = ROOT / "artifacts" / "research-backup-2026-10-13"
RECEIPT = ROOT / "runs" / "publication-2026-10-13" / "archive-publication.json"
REPO_ID = "axondendriteplus/decision-lab-research-archive"
PATTERNS = (
    rb"hf_[A-Za-z0-9]{25,}",
    rb"(?:ak|as|tk)-[A-Za-z0-9_-]{25,}",
    rb"gh[pousr]_[A-Za-z0-9]{25,}",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    rb"https://modal[.]com/token-flow/[^\s]+",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def scan(payload, name):
    for pattern in PATTERNS:
        if re.search(pattern, payload):
            raise RuntimeError(f"Credential-pattern match in {name}")


def scan_archive(path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise RuntimeError(f"Unsafe archived path in {path}")
            if any(part in {".env", ".modal.toml", "secrets", "credentials"} for part in parts):
                raise RuntimeError(f"Credential filename in {path}")
            if member.isfile():
                with archive.extractfile(member) as source:
                    scan(source.read(), f"{path}:{member.name}")


def prepare():
    if DESTINATION.exists():
        raise RuntimeError("Backup directory already exists; use upload to resume")
    DESTINATION.mkdir(parents=True)
    published = {}
    selected = []
    for receipt in sorted((ROOT / "reports").glob("*evidence/huggingface-release.json")):
        for record in json.loads(receipt.read_text()).values():
            path = Path(
                hf_hub_download(
                    record["repo_id"], "artifact-manifest.json", revision=record["commit"]
                )
            )
            if sha256(path) != record["artifact_manifest_sha256"]:
                raise RuntimeError("Selected manifest checksum differs")
            reference = {key: record[key] for key in ("repo_id", "commit")}
            reference["filename"] = "model.safetensors"
            published[json.loads(path.read_text())["files"]["model.safetensors"]] = reference
            selected.append(record["repo_id"])
    if len(selected) != 11:
        raise RuntimeError("Expected eleven selected releases")
    records, weights, inode_hashes, missing = {}, {}, {}, {}
    for root in (ROOT / "data", ROOT / "runs", ROOT / "artifacts"):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if "publication-2026-10-13" in relative.parts or DESTINATION in path.parents:
                continue
            if set(relative.parts) & {".cache", "__pycache__", ".DS_Store"}:
                continue
            if path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise RuntimeError(f"Unexpected symlink: {relative}")
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino, stat.st_size)
            if inode not in inode_hashes:
                inode_hashes[inode] = sha256(path)
            digest = inode_hashes[inode]
            entry = {"sha256": digest, "bytes": stat.st_size}
            if path.suffix in {".safetensors", ".pt", ".pth", ".bin", ".onnx"}:
                if digest in published:
                    entry["remote"] = published[digest]
                else:
                    target = f"weights/{digest}{path.suffix}"
                    entry["archive_path"] = target
                    missing[target] = entry.copy()
                    staged = DESTINATION / target
                    if not staged.exists():
                        staged.parent.mkdir(exist_ok=True)
                        os.link(path, staged)
                weights[str(relative)] = entry
            else:
                if path.name.endswith(".tar.gz"):
                    scan_archive(path)
                else:
                    scan(path.read_bytes(), str(relative))
                records[str(relative)] = entry
    bundle = DESTINATION / "records.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=6, dereference=True) as archive:
        for relative in records:
            archive.add(ROOT / relative, arcname=relative, recursive=False)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "records": records,
        "weights": weights,
        "uploaded_files": {
            "records.tar.gz": {"sha256": sha256(bundle), "bytes": bundle.stat().st_size},
            **missing,
        },
        "excluded": ["SDK caches", "compiled Python", "publication operation files"],
        "scope": (
            "Existing local data, runs and artifacts; duplicate weights map to exact "
            "private selected releases or one uploaded SHA-256 blob."
        ),
    }
    save(DESTINATION / "manifest.json", manifest)
    (DESTINATION / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\n# Decision Model Lab research archive\n\n"
        "Private V1–V4 research backup. These are experimental models with "
        "documented generalization failures.\n\n"
        "Code, reports and journal: "
        "https://github.com/adc77/decision-model-lab/tree/feat/data-foundation\n\n"
        "`manifest.json` records every retained non-cache file under local "
        "`data`, `runs` and `artifacts`. "
        "Non-weight files preserve their original paths inside `records.tar.gz`. "
        "Weight entries map their original paths "
        "to either an exact pinned private model release or a content-addressed "
        "blob in this repository. "
        "Verify SHA-256 before using or restoring files into a new directory. Do "
        "not overwrite an existing study.\n\n"
        "Temporary Modal optimizer recovery state was discarded during verified "
        "study cleanup; it is not retained here. "
        "Prior discarded V2 intermediate checkpoints are also unavailable. SDK "
        "caches, compiled Python and this "
        "publication operation's files are excluded. This archive preserves the "
        "available research record, "
        "not unrecorded earlier conversations. Raw weights require the matching "
        "configs/tokenizers and code. "
        "Use selected private release bundles for supported inference entry points.\n"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "weight_paths": len(weights),
                "new_weight_blobs": len(missing),
                "archive_bytes": bundle.stat().st_size,
            }
        ),
        flush=True,
    )


def upload():
    api = HfApi()
    manifest = json.loads((DESTINATION / "manifest.json").read_text())
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        if receipt["repo_id"] != REPO_ID or not receipt["created_private_by_this_publication"]:
            raise RuntimeError("Invalid publication ownership receipt")
    else:
        try:
            api.model_info(REPO_ID)
        except RepositoryNotFoundError:
            api.create_repo(REPO_ID, private=True, exist_ok=False)
        else:
            raise RuntimeError("Repository already exists without an ownership receipt")
        receipt = {"repo_id": REPO_ID, "created_private_by_this_publication": True}
        save(RECEIPT, receipt)
    if not api.model_info(REPO_ID).private:
        raise RuntimeError("Research archive must be private")
    files = dict(manifest["uploaded_files"])
    for name in ("manifest.json", "README.md"):
        path = DESTINATION / name
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for name, entry in files.items():
        if sha256(DESTINATION / name) != entry["sha256"]:
            raise RuntimeError(f"Staged checksum changed: {name}")
    if "commit" not in receipt:
        commit = api.create_commit(
            REPO_ID,
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(DESTINATION / name))
                for name in files
            ],
            commit_message="Back up V1–V4 raw records, source archives and retained checkpoints",
        )
        receipt["commit"] = commit.oid
        save(RECEIPT, receipt)
    info = api.model_info(REPO_ID, revision=receipt["commit"])
    if not info.private or info.sha != receipt["commit"]:
        raise RuntimeError("Pinned private archive verification failed")
    remote = {
        item.path: item
        for item in api.get_paths_info(REPO_ID, list(files), revision=receipt["commit"])
    }
    for name, entry in files.items():
        item = remote[name]
        if item.size != entry["bytes"]:
            raise RuntimeError(f"Remote size differs: {name}")
        if item.lfs is not None:
            if item.lfs.sha256 != entry["sha256"]:
                raise RuntimeError(f"Remote LFS checksum differs: {name}")
        else:
            downloaded = Path(
                hf_hub_download(REPO_ID, name, revision=receipt["commit"], force_download=True)
            )
            if sha256(downloaded) != entry["sha256"]:
                raise RuntimeError(f"Downloaded checksum differs: {name}")
    downloaded = Path(
        hf_hub_download(REPO_ID, "records.tar.gz", revision=receipt["commit"], force_download=True)
    )
    if sha256(downloaded) != files["records.tar.gz"]["sha256"]:
        raise RuntimeError("Fresh downloaded records archive differs")
    seen = set()
    with tarfile.open(downloaded, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise RuntimeError("Unexpected record archive member")
            with archive.extractfile(member) as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != manifest["records"][member.name]["sha256"]:
                raise RuntimeError(f"Downloaded record differs: {member.name}")
            seen.add(member.name)
    if seen != set(manifest["records"]):
        raise RuntimeError("Downloaded records inventory differs")
    receipt.update(
        {
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "private": True,
            "remote_files_verified": len(files),
            "fresh_download_records_verified": len(seen),
            "weight_paths_preserved": len(manifest["weights"]),
            "new_weight_blobs": len(files) - 3,
            "manifest_sha256": files["manifest.json"]["sha256"],
            "verification": (
                "Fresh records/member hashes; uploaded weight blob SHA-256 and size from "
                "pinned remote LFS metadata; selected-release manifests verified "
                "separately."
            ),
        }
    )
    save(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "upload"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    else:
        upload()
