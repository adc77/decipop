import os
import shutil
from pathlib import Path

from decision_lab.training.checkpoint import file_hash


def ensure_space(bytes_required):
    if shutil.disk_usage(Path.cwd()).free < bytes_required + 512 * 1024**2:
        raise OSError("Insufficient free space for a bounded owned artifact transfer")


def download(volume, remote, target, expected=None):
    if target.exists():
        if expected is not None and file_hash(target) == expected:
            return
        raise FileExistsError(f"Refusing to overwrite retained file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Preserve the failed partial download before resuming: {temporary}")
    with temporary.open("xb") as output:
        for chunk in volume.read_file(remote):
            output.write(chunk)
    if expected is not None and file_hash(temporary) != expected:
        raise ValueError(f"Retained checksum differs: {remote}")
    os.replace(temporary, target)


def deduplicate_verified_weight(fresh, retained, digest):
    if fresh.is_symlink() or retained.is_symlink():
        raise ValueError("Owned model weights must be regular files")
    if file_hash(fresh) != digest or file_hash(retained) != digest:
        raise ValueError("Verified and retained weights differ")
    temporary = fresh.with_suffix(".deduplicated")
    if temporary.exists():
        raise FileExistsError(temporary)
    os.link(retained, temporary)
    os.replace(temporary, fresh)
    if file_hash(fresh) != digest or not os.path.samefile(fresh, retained):
        raise ValueError("Owned verified-weight deduplication failed")
