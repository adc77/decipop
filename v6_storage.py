import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from v4_retain import download
from v6_lab.protocol import verify

ROOT = Path("runs/study-v6")
REPOSITORY = "axondendriteplus/decision-lab-v6-checkpoints"


def verify_weight(api, filename, revision, digest, size):
    info = api.model_info(REPOSITORY, revision=revision, files_metadata=True)
    if not info.private or info.sha != revision:
        raise ValueError("Private checkpoint revision differs")
    entry = next(s for s in info.siblings if s.rfilename == filename)
    if entry.lfs.sha256 != digest or entry.size != size:
        raise ValueError("Pinned remote checkpoint checksum/size differs")


def retain_candidates(volume):
    selection = json.loads((ROOT / "selection-freeze.json").read_text())
    selected = {p["calibrated_nll"] for p in selection["choices"].values()}
    protocol = verify(ROOT)
    if selection["protocol_identity"] != protocol["identity"]:
        raise ValueError("Selection belongs to another active protocol")
    api = HfApi()
    if api.whoami()["name"] != "axondendriteplus":
        raise ValueError("HF account differs")
    path = ROOT / "checkpoint-storage.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if (
            receipt["repository"] != REPOSITORY
            or receipt["protocol_identity"] != protocol["identity"]
        ):
            raise ValueError("Checkpoint ownership receipt differs")
    else:
        try:
            api.model_info(REPOSITORY)
        except RepositoryNotFoundError:
            api.create_repo(REPOSITORY, private=True)
        else:
            raise FileExistsError("Checkpoint repository exists without this study's receipt")
        receipt = {
            "repository": REPOSITORY,
            "protocol_identity": protocol["identity"],
            "weights": {},
        }
        atomic_json(path, receipt)
    retained = {}
    for name, candidate in selection["candidates"].items():
        digest = candidate["weights_sha256"]
        if candidate["step"] == 0:
            target = ROOT / "parent/model.safetensors"
            if file_hash(target) != digest:
                raise ValueError("Initial candidate differs")
            retained[name] = {
                "path": str(target),
                "sha256": digest,
                "repo_id": protocol["settings"]["parent_repo"],
                "commit": protocol["settings"]["parent_commit"],
                "filename": "model.safetensors",
            }
            continue
        if name not in receipt["weights"]:
            stage = Path("artifacts/v6-checkpoint-staging") / name / "model.safetensors"
            download(volume, f"/study-v6/{candidate['directory']}/model.safetensors", stage, digest)
            filename = f"weights/{digest}.safetensors"
            size = stage.stat().st_size
            commit = api.upload_file(
                path_or_fileobj=stage,
                path_in_repo=filename,
                repo_id=REPOSITORY,
                commit_message=f"Retain V6 {name}",
            )
            verify_weight(api, filename, commit.oid, digest, size)
            local = None
            if name in selected:
                local = ROOT / candidate["directory"] / "model.safetensors"
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    if file_hash(local) != digest:
                        raise ValueError("Retained selected weight differs")
                else:
                    os.link(stage, local)
            receipt["weights"][name] = {
                "repo_id": REPOSITORY,
                "commit": commit.oid,
                "filename": filename,
                "sha256": digest,
                "size": size,
                "path": str(local) if local is not None else None,
                "verification": (
                    "Pinned remote LFS size/SHA metadata; no intermediate inference claim"
                ),
            }
            atomic_json(path, receipt)
            stage.unlink()
            stage.parent.rmdir()
            print(
                json.dumps(
                    {"candidate_archived": name, "selected_retained_locally": local is not None}
                ),
                flush=True,
            )
        record = receipt["weights"][name]
        verify_weight(api, record["filename"], record["commit"], record["sha256"], record["size"])
        if record["path"] is not None and file_hash(Path(record["path"])) != digest:
            raise ValueError("Selected local checksum differs")
        retained[name] = record
    manifest = ROOT / "retained-candidates.json"
    atomic_json(manifest, retained)
    commit = api.upload_file(
        path_or_fileobj=manifest,
        path_in_repo="checkpoints.json",
        repo_id=REPOSITORY,
        commit_message="Record verified V6 candidate inventory",
    )
    for record in receipt["weights"].values():
        verify_weight(api, record["filename"], commit.oid, record["sha256"], record["size"])
    fresh = Path(hf_hub_download(REPOSITORY, "checkpoints.json", revision=commit.oid))
    if file_hash(fresh) != file_hash(manifest):
        raise ValueError("Fresh pinned checkpoint inventory differs")
    receipt |= {
        "final_commit": commit.oid,
        "candidate_inventory_sha256": file_hash(manifest),
        "candidates_verified": len(retained),
        "all_additional_weight_metadata_verified": True,
    }
    atomic_json(path, receipt)
    return retained
