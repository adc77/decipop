import hashlib
from types import SimpleNamespace

import pytest

import v4_retain


class FakeVolume:
    def __init__(self, payloads):
        self.payloads = payloads
        self.reads = []

    def read_file(self, path):
        self.reads.append(path)
        payload = self.payloads[path]
        yield payload[:3]
        yield payload[3:]

    def iterdir(self, path, recursive):
        assert path == "/study-v4" and recursive
        return [
            SimpleNamespace(path=name.removeprefix("/"), type=SimpleNamespace(name="FILE"))
            for name in self.payloads
        ]


def test_download_checks_hash_before_atomic_publication(tmp_path):
    payload = b"verified-model-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    volume = FakeVolume({"/model": payload})
    target = tmp_path / "model.safetensors"
    v4_retain.download(volume, "/model", target, digest)
    assert target.read_bytes() == payload
    v4_retain.download(volume, "/model", target, digest)
    assert volume.reads == ["/model"]
    with pytest.raises(FileExistsError, match="overwrite"):
        v4_retain.download(volume, "/model", target, "wrong")
    incomplete = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError, match="checksum"):
        v4_retain.download(volume, "/model", incomplete, "wrong")
    assert not incomplete.exists()


def test_records_use_owned_relative_paths_and_reject_divergent_local_files(tmp_path, monkeypatch):
    monkeypatch.setattr(v4_retain, "ROOT", tmp_path)
    payload = b'{"identity":"frozen"}\n'
    volume = FakeVolume(
        {
            "/study-v4/selection-freeze.json": payload,
            "/study-v4/training/train_control.jsonl": b"already retained",
            "/study-v4/recovery/control/progress.json": b"temporary",
        }
    )
    recorded = v4_retain.records(volume)
    assert set(recorded) == {"selection-freeze.json"}
    assert volume.reads == ["/study-v4/selection-freeze.json"]
    assert (tmp_path / "selection-freeze.json").read_bytes() == payload
    assert not (tmp_path / "training").exists()
    (tmp_path / "selection-freeze.json").write_text("changed")
    with pytest.raises(ValueError, match="differ"):
        v4_retain.records(volume)
