import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/premise-release"
RECEIPT = ROOT / "runs/premise-publication/huggingface.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(records):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temp = RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(records, indent=2) + "\n")
    os.replace(temp, RECEIPT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "public"))
    action = parser.parse_args().action
    api = HfApi()
    if api.whoami()["name"] != "axondendriteplus":
        raise ValueError("Unexpected Hugging Face account")
    prepared = json.loads((STAGE / "prepared.json").read_text())
    records = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    entries = [(prepared["dataset"], "dataset")] + [(m, "model") for m in prepared["models"]]
    for entry, kind in entries:
        repo, folder = entry["repo_id"], Path(entry["directory"])
        if action == "public":
            if not records[repo].get("verified"):
                raise ValueError("Verify the private publication before changing visibility")
            api.update_repo_settings(repo_id=repo, repo_type=kind, private=False)
            info = api.repo_info(repo_id=repo, repo_type=kind)
            if info.private:
                raise ValueError("Visibility change was not applied")
            records[repo]["public"] = True
            save(records)
            print(json.dumps(dict(public=repo, commit=info.sha)), flush=True)
            continue
        if repo not in records:
            if api.repo_exists(repo, repo_type=kind):
                raise FileExistsError(f"Refusing to overwrite an unowned destination: {repo}")
            api.create_repo(repo, repo_type=kind, private=True)
            records[repo] = dict(created_private=True, repo_type=kind)
            save(records)
        record = records[repo]
        if record.get("verified"):
            if not record.get("public"):
                api.update_repo_settings(repo_id=repo, repo_type=kind, private=False)
                if api.repo_info(repo_id=repo, repo_type=kind).private:
                    raise ValueError("Verified release visibility did not change")
                record["public"] = True
                save(records)
            print(json.dumps(dict(already_verified=repo)), flush=True)
            continue
        if "commit" not in record:
            commit = api.upload_folder(
                repo_id=repo,
                repo_type=kind,
                folder_path=folder,
                commit_message="Release Premise research data and verified model artifacts",
            )
            record["commit"] = commit.oid
            save(records)
        revision = record["commit"]
        info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
        remote_files = {f.rfilename: f for f in info.siblings}
        for path in folder.rglob("*"):
            if not path.is_file() or ".cache" in path.parts:
                continue
            relative = str(path.relative_to(folder))
            remote = remote_files[relative]
            if remote.size != path.stat().st_size:
                raise ValueError(f"Remote size differs: {repo}/{relative}")
            if remote.lfs is not None and remote.lfs.sha256 != digest(path):
                raise ValueError(f"Remote LFS digest differs: {repo}/{relative}")
        if kind == "model":
            fresh = STAGE / "fresh-verification" / repo.split("/")[1]
            snapshot_download(repo, revision=revision, local_dir=fresh)
            output = subprocess.check_output(
                [
                    "uv",
                    "run",
                    "--extra",
                    "train",
                    "python",
                    "-c",
                    "import json,sys,torch; from pathlib import Path; "
                    "torch.set_num_threads(4); "
                    "from decision_lab.release import verify_release; "
                    "print(json.dumps(verify_release(Path(sys.argv[1]))))",
                    str(fresh),
                ],
                cwd=ROOT,
                text=True,
            )
            checked = json.loads(output)
            if not checked["verified"] or checked["inputs"] != 8:
                raise ValueError("CPU reference check failed")
            record["fresh_download_cpu"] = checked
            weight = fresh / "model.safetensors"
            record["weight_sha256"] = digest(weight)
            weight.unlink()
        else:
            check = hf_hub_download(repo, "sha256.json", repo_type=kind, revision=revision)
            if digest(Path(check)) != digest(folder / "sha256.json"):
                raise ValueError("Pinned dataset manifest differs")
            record["dataset_rows"] = entry["rows"]
            record["data_lfs_hashes_verified"] = True
        record["verified"] = True
        save(records)
        api.update_repo_settings(repo_id=repo, repo_type=kind, private=False)
        if api.repo_info(repo_id=repo, repo_type=kind).private:
            raise ValueError("Verified release visibility did not change")
        record["public"] = True
        save(records)
        print(json.dumps(dict(verified=repo, commit=revision)), flush=True)


if __name__ == "__main__":
    main()
