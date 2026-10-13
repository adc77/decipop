import json
import os
import subprocess
import sys
from pathlib import Path

from decision_lab.training.checkpoint import file_hash

MODEL_NAMES = {
    "token": "decision-lab-smollm2-360m-token",
    "head": "decision-lab-smollm2-360m-head",
    "scratch": "decision-lab-transformer-14m",
}


def release_inventory(directory: Path):
    manifest = json.loads((directory / "artifact-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise ValueError(f"Unsafe release path: {name}")
        if file_hash(path) != digest:
            raise ValueError(f"Release checksum mismatch: {name}")
    return manifest


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def annotate_counterfactual_results(source: Path, study: Path):
    protocol = json.loads((study / "protocol.json").read_text())
    breakdown = json.loads((study / "counterfactual-breakdown.json").read_text())
    for architecture in MODEL_NAMES:
        directory = source / architecture
        manifest = release_inventory(directory)
        filename = "posthoc-counterfactual-breakdown.json"
        if filename in manifest["files"]:
            continue
        path = study / "predictions" / architecture / "robust_flip_fact.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        direct = [row for row in rows if row["label_changed"] and row["label"] != 2]
        stats = breakdown["models"][architecture]
        metadata = {
            "analysis": "post hoc breakdown of the frozen fact-flip suite; no retraining",
            "protocol_identity": protocol["identity"],
            "variant_file_sha256": protocol["input_files"]["test-inputs/flip_fact.jsonl"],
            "prediction_file_sha256": file_hash(path),
            "broken_support_definition": "oracle label changes from TRUE/FALSE to UNKNOWN",
            "broken_support": stats,
            "direct_fact_reversals": {
                "questions": len(direct),
                "correct": sum(row["prediction"] == row["label"] for row in direct),
            },
            "metadata_added_after_initial_gpu_reload_verification": True,
            "weights_code_tokenizer_and_calibration_unchanged": True,
        }
        atomic_json(directory / filename, metadata)
        paragraph = (
            "\n## Counterfactual limits\n\n"
            f"In the frozen fact-flip suite, this checkpoint gets {stats['correct']} of "
            f"{stats['questions']} broken-support questions correct: the oracle answer "
            "becomes UNKNOWN when a changed premise prevents derivation. "
            f"It retains the original label on {stats['kept_original_label']} of these cases. "
            f"{stats['high_confidence_wrong']} wrong answers have calibrated confidence ≥90%. "
            "High ID accuracy or low ID ECE does not establish reliable rule following. "
            "See posthoc-counterfactual-breakdown.json and evaluation.json. "
            "This metadata was added after the initial GPU reload check; model weights, "
            "inference code, tokenizer, and calibration are unchanged. "
            "The published bundle is checked again from a pinned Hub download.\n"
        )
        card = directory / "README.md"
        card.write_text(card.read_text() + paragraph)
        manifest["files"]["README.md"] = file_hash(card)
        manifest["files"][filename] = file_hash(directory / filename)
        atomic_json(directory / "artifact-manifest.json", manifest)
        release_inventory(directory)


def download_modal_releases(volume, study: str, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for architecture in MODEL_NAMES:
        directory = destination / architecture
        directory.mkdir(exist_ok=True)
        remote = f"/{study}/releases/{architecture}"
        payload = b"".join(volume.read_file(remote + "/artifact-manifest.json"))
        manifest = json.loads(payload)
        for name, digest in manifest["files"].items():
            target = directory / name
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Unsafe remote artifact path: {name}")
            if target.exists() and file_hash(target) == digest:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(remote + "/" + name):
                    stream.write(chunk)
            if file_hash(temporary) != digest:
                raise ValueError(f"Downloaded artifact checksum mismatch: {name}")
            os.replace(temporary, target)
        (directory / "artifact-manifest.json").write_bytes(payload)
        release_inventory(directory)
        print(json.dumps({"downloaded_release": architecture}), flush=True)


def publish_private_releases(source: Path, receipt: Path, namespace: str, model_names=None):
    from huggingface_hub import HfApi

    api = HfApi()
    if api.whoami()["name"] != namespace:
        raise ValueError("Hugging Face login does not match the requested personal namespace")
    records = json.loads(receipt.read_text()) if receipt.exists() else {}
    names = MODEL_NAMES if model_names is None else model_names
    for architecture, suffix in names.items():
        directory = source / architecture
        manifest = release_inventory(directory)
        repo = namespace + "/" + suffix
        if architecture in records:
            record = records[architecture]
            if (
                record["repo_id"] != repo
                or record["protocol_identity"] != manifest["protocol_identity"]
            ):
                raise ValueError("Existing publication receipt belongs to another study")
        else:
            if api.repo_exists(repo):
                raise FileExistsError(
                    f"Refusing to overwrite a repository without our receipt: {repo}"
                )
            api.create_repo(repo_id=repo, private=True, repo_type="model")
            records[architecture] = {
                "repo_id": repo,
                "protocol_identity": manifest["protocol_identity"],
                "created_private_by_this_release": True,
            }
            atomic_json(receipt, records)
        info = api.model_info(repo)
        if not info.private:
            raise ValueError(f"Model repository is not private: {repo}")
        if "commit" not in records[architecture]:
            commit = api.upload_folder(
                repo_id=repo,
                folder_path=directory,
                allow_patterns=[*manifest["files"], "artifact-manifest.json"],
                commit_message="Upload verified Decision Model Lab research artifact",
                parent_commit=info.sha,
            )
            records[architecture] |= {
                "commit": commit.oid,
                "url": f"https://huggingface.co/{repo}",
                "artifact_manifest_sha256": file_hash(directory / "artifact-manifest.json"),
            }
            atomic_json(receipt, records)
        elif records[architecture]["artifact_manifest_sha256"] != file_hash(
            directory / "artifact-manifest.json"
        ):
            raise ValueError("Local release differs from its previously published artifact")
        print(
            json.dumps({"published_private": repo, "commit": records[architecture]["commit"]}),
            flush=True,
        )
    return records


def verify_hub_downloads(receipt: Path, destination: Path):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    records = json.loads(receipt.read_text())
    for architecture, record in records.items():
        if "remote_download_verification" in record:
            print(json.dumps({"hub_download_already_verified": architecture}), flush=True)
            continue
        output = destination / architecture
        if output.exists():
            raise FileExistsError(f"Hub verification requires a fresh directory: {output}")
        info = HfApi().model_info(record["repo_id"], revision=record["commit"])
        if not info.private or info.sha != record["commit"]:
            raise ValueError("Published model privacy or revision verification failed")
        manifest_path = Path(
            hf_hub_download(record["repo_id"], "artifact-manifest.json", revision=record["commit"])
        )
        if file_hash(manifest_path) != record["artifact_manifest_sha256"]:
            raise ValueError("Hub artifact manifest changed")
        manifest = json.loads(manifest_path.read_text())
        snapshot_download(
            record["repo_id"],
            revision=record["commit"],
            local_dir=output,
            allow_patterns=[*manifest["files"], "artifact-manifest.json"],
        )
        release_inventory(output)
        code = (
            "import json,sys,torch; from pathlib import Path; "
            "torch.set_num_threads(2); "
            "root=Path(sys.argv[1]); sys.path.insert(0,str(root/'code')); "
            "from decision_lab.release import verify_release; "
            "print(json.dumps(verify_release(root)))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(output.resolve())],
            cwd=output,
            check=True,
            capture_output=True,
            text=True,
        )
        verification = json.loads(result.stdout)
        records[architecture]["remote_download_verification"] = verification
        records[architecture]["verified_download_directory"] = str(output.resolve())
        atomic_json(receipt, records)
        print(json.dumps({"hub_download_verified": architecture, **verification}), flush=True)
    return records
