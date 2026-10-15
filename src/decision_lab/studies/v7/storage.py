import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from decision_lab.publication import atomic_json
from decision_lab.studies.v7.files import download, ensure_space
from decision_lab.studies.v7.protocol import verify
from decision_lab.training.checkpoint import file_hash


def verify_weight(api, repository, filename, revision, digest, size):
    info = api.model_info(repository, revision=revision, files_metadata=True)
    if not info.private or info.sha != revision:
        raise ValueError("Private checkpoint revision differs")
    entry = next(s for s in info.siblings if s.rfilename == filename)
    if entry.lfs.sha256 != digest or entry.size != size:
        raise ValueError("Pinned remote checkpoint checksum/size differs")


def retain_candidates(root, volume):
    repository = f"axondendriteplus/decision-lab-v7-{root.name}-checkpoints"
    selection = json.loads((root / "selection-freeze.json").read_text())
    selected = {p["calibrated_nll"] for p in selection["choices"].values()}
    protocol = verify(root)
    if selection["protocol_identity"] != protocol["identity"]:
        raise ValueError("Selection belongs to another active protocol")
    api = HfApi()
    if api.whoami()["name"] != "axondendriteplus":
        raise ValueError("HF account differs")
    path = root / "checkpoint-storage.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if (
            receipt["repository"] != repository
            or receipt["protocol_identity"] != protocol["identity"]
        ):
            raise ValueError("Checkpoint ownership receipt differs")
    else:
        try:
            api.model_info(repository)
        except RepositoryNotFoundError:
            api.create_repo(repository, private=True)
        else:
            raise FileExistsError("Checkpoint repository exists without this study's receipt")
        receipt = {
            "repository": repository,
            "protocol_identity": protocol["identity"],
            "weights": {},
        }
        atomic_json(path, receipt)
    retained = {}
    for name, candidate in selection["candidates"].items():
        digest = candidate["weights_sha256"]
        if candidate["step"] == 0:
            target = root / "parent/model.safetensors"
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
            stage = Path("artifacts/v7-checkpoint-staging") / root.name / name / "model.safetensors"
            ensure_space((root / "parent/model.safetensors").stat().st_size)
            download(
                volume,
                f"/study-v7/{root.name}/{candidate['directory']}/model.safetensors",
                stage,
                digest,
            )
            filename = f"weights/{digest}.safetensors"
            size = stage.stat().st_size
            commit = api.upload_file(
                path_or_fileobj=stage,
                path_in_repo=filename,
                repo_id=repository,
                commit_message=f"Retain V7 {name}",
            )
            verify_weight(api, repository, filename, commit.oid, digest, size)
            local = None
            if name in selected:
                local = root / candidate["directory"] / "model.safetensors"
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    if file_hash(local) != digest:
                        raise ValueError("Retained selected weight differs")
                else:
                    os.link(stage, local)
            receipt["weights"][name] = {
                "repo_id": repository,
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
        verify_weight(
            api, repository, record["filename"], record["commit"], record["sha256"], record["size"]
        )
        if record["path"] is not None and file_hash(Path(record["path"])) != digest:
            raise ValueError("Selected local checksum differs")
        retained[name] = record
    manifest = root / "retained-candidates.json"
    if "final_commit" in receipt:
        if (
            file_hash(manifest) != receipt["candidate_inventory_sha256"]
            or json.loads(manifest.read_text()) != retained
        ):
            raise ValueError("Retained pinned candidate inventory changed")
        revision = receipt["final_commit"]
    else:
        atomic_json(manifest, retained)
        commit = api.upload_file(
            path_or_fileobj=manifest,
            path_in_repo="checkpoints.json",
            repo_id=repository,
            commit_message="Record verified V7 candidate inventory",
        )
        revision = commit.oid
    for record in receipt["weights"].values():
        verify_weight(
            api, repository, record["filename"], revision, record["sha256"], record["size"]
        )
    fresh = Path(hf_hub_download(repository, "checkpoints.json", revision=revision))
    if file_hash(fresh) != file_hash(manifest):
        raise ValueError("Fresh pinned checkpoint inventory differs")
    receipt |= {
        "final_commit": revision,
        "candidate_inventory_sha256": file_hash(manifest),
        "candidates_verified": len(retained),
        "all_additional_weight_metadata_verified": True,
    }
    atomic_json(path, receipt)
    return retained
