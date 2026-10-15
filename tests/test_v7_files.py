import os

import pytest

from decision_lab.studies.v7.files import deduplicate_verified_weight, download
from decision_lab.training.checkpoint import file_hash


def test_verified_deduplication_preserves_bundle_bytes(tmp_path):
    fresh, retained = tmp_path / "fresh.safetensors", tmp_path / "retained.safetensors"
    fresh.write_bytes(b"verified weights")
    retained.write_bytes(fresh.read_bytes())
    digest = file_hash(fresh)
    deduplicate_verified_weight(fresh, retained, digest)
    assert os.path.samefile(fresh, retained) and file_hash(fresh) == digest


def test_deduplication_rejects_a_different_retained_weight(tmp_path):
    fresh, retained = tmp_path / "fresh.safetensors", tmp_path / "retained.safetensors"
    fresh.write_bytes(b"verified weights")
    retained.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="differ"):
        deduplicate_verified_weight(fresh, retained, file_hash(fresh))
    assert fresh.read_bytes() == b"verified weights"


def test_failed_download_bytes_are_never_silently_overwritten(tmp_path):
    target = tmp_path / "model.safetensors"
    partial = target.with_suffix(".safetensors.partial")
    partial.write_bytes(b"retained interrupted transfer")
    with pytest.raises(FileExistsError, match="partial"):
        download(None, "/owned/model.safetensors", target)
    assert partial.read_bytes() == b"retained interrupted transfer"
