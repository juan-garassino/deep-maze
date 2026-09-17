"""docker/upload_artifacts.py — the GCS UPLOAD counterpart to sync_assets.

In-process with a fake google.cloud.storage (no SDK / network). Covers the two
behaviors the ephemeral training job depends on: walking OUTPUT_BASE and
uploading each file under ${ARTIFACTS_PREFIX}, and resolving an INIT_BUNDLE
gs:// URI back to a local path for cross-rung warm-start.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "upload_artifacts_under_test", root / "docker" / "upload_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubBlob:
    def __init__(self, store: dict, name: str):
        self._store = store
        self.name = name

    def upload_from_filename(self, path: str) -> None:
        self._store[self.name] = Path(path).read_bytes()

    def download_to_filename(self, dst: str) -> None:
        Path(dst).write_bytes(self._store[self.name])


class _StubBucket:
    def __init__(self, store: dict, name: str):
        self._store = store
        self.name = name

    def blob(self, name: str) -> _StubBlob:
        return _StubBlob(self._store, name)


class _StubClient:
    def __init__(self, store: dict):
        self._store = store

    def bucket(self, name: str) -> _StubBucket:
        return _StubBucket(self._store, name)


def _install_fake_storage(store: dict):
    google_pkg = types.ModuleType("google")
    google_cloud_pkg = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = lambda: _StubClient(store)
    google_cloud_pkg.storage = storage_mod
    google_pkg.cloud = google_cloud_pkg
    sys.modules["google"] = google_pkg
    sys.modules["google.cloud"] = google_cloud_pkg
    sys.modules["google.cloud.storage"] = storage_mod


def test_upload_walks_tree_and_applies_prefix(tmp_path):
    store: dict = {}
    _install_fake_storage(store)
    src = tmp_path / "out"
    (src / "assets" / "drqn").mkdir(parents=True)
    (src / "result.json").write_text("{}")
    (src / "assets" / "drqn" / "model.pt").write_bytes(b"weights")

    mod = _load_module()
    n = mod.upload(src, "garassino-ml-artifacts", "deepmaze-runs/cloudrun/tag/")

    assert n == 2
    assert "deepmaze-runs/cloudrun/tag/result.json" in store
    assert store["deepmaze-runs/cloudrun/tag/assets/drqn/model.pt"] == b"weights"


def test_download_blob_roundtrip(tmp_path):
    store = {"deepmaze/run/model.best.pt": b"ckpt"}
    _install_fake_storage(store)
    mod = _load_module()
    dest = tmp_path / "init" / "model.pt"
    out = mod.download_blob("gs://garassino-ml-artifacts/deepmaze/run/model.best.pt", dest)
    assert Path(out).read_bytes() == b"ckpt"


def test_download_blob_rejects_non_gs(tmp_path):
    mod = _load_module()
    with pytest.raises(ValueError):
        mod.download_blob("/local/path", tmp_path / "x")
