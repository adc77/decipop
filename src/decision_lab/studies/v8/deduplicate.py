import json
import os
from pathlib import Path

from decision_lab.publication import atomic_json
from decision_lab.studies.v7.files import deduplicate_verified_weight
from decision_lab.training.checkpoint import file_hash


def main():
    destination = Path("artifacts/v8-preparation/disk-deduplication.json")
    if destination.exists():
        raise FileExistsError(destination)
    files = sorted(
        p
        for folder in ("artifacts", "runs")
        for p in Path(folder).rglob("*.safetensors")
        if p.is_file()
    )
    hashes, inodes, retained, changes = {}, {}, {}, []
    before = os.statvfs(".").f_bavail * os.statvfs(".").f_frsize
    for path in files:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in inodes:
            digest = inodes[key]
        else:
            digest = file_hash(path)
            inodes[key] = digest
        hashes[str(path)] = digest
        if digest not in retained:
            retained[digest] = path
        elif not os.path.samefile(path, retained[digest]):
            deduplicate_verified_weight(path, retained[digest], digest)
            changes.append(
                {
                    "path": str(path),
                    "linked_to": str(retained[digest]),
                    "sha256": digest,
                    "bytes": stat.st_size,
                }
            )
            print(json.dumps(changes[-1]), flush=True)
    verified = {}
    for path in files:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key not in verified:
            verified[key] = file_hash(path)
        if verified[key] != hashes[str(path)]:
            raise ValueError("Historical weight changed")
    after = os.statvfs(".").f_bavail * os.statvfs(".").f_frsize
    atomic_json(
        destination,
        {
            "all_paths_and_hashes_preserved": True,
            "files": hashes,
            "changes": changes,
            "observed_free_byte_change": after - before,
        },
    )
    print(json.dumps({"files": len(files), "changed": len(changes), "free_delta": after - before}))


if __name__ == "__main__":
    main()
