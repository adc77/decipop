import json

import pytest

from decision_lab.publication import publish_private_releases, release_inventory
from decision_lab.training.checkpoint import file_hash


def bundle(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "protocol_identity": "fixture",
                "files": {"config.json": file_hash(path / "config.json")},
            }
        )
    )


def test_publication_refuses_an_existing_repo_without_its_receipt(tmp_path, monkeypatch):
    import huggingface_hub

    class ExistingRepo:
        def whoami(self):
            return {"name": "personal"}

        def repo_exists(self, _repo):
            return True

        def create_repo(self, **_kwargs):
            pytest.fail("Existing repositories must never be overwritten")

    bundle(tmp_path / "source/token")
    monkeypatch.setattr(huggingface_hub, "HfApi", ExistingRepo)
    with pytest.raises(FileExistsError, match="without our receipt"):
        publish_private_releases(tmp_path / "source", tmp_path / "receipt.json", "personal")


def test_release_inventory_rejects_tampering_and_directory_escape(tmp_path):
    bundle(tmp_path / "bundle")
    assert release_inventory(tmp_path / "bundle")["protocol_identity"] == "fixture"
    (tmp_path / "bundle/config.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        release_inventory(tmp_path / "bundle")
    manifest = {"files": {"../outside": "fake"}}
    (tmp_path / "bundle/artifact-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        release_inventory(tmp_path / "bundle")
