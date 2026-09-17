"""web/server.py POST /api/resync — runtime asset refresh so a freshly promoted
bundle is visible to a warm instance without a restart. Fake google.cloud.storage,
no SDK / network."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "web"))

try:
    from fastapi.testclient import TestClient
    from server import create_app  # noqa: E402

    create_app()  # smoke-construct; skip on fastapi/starlette env mismatch
except Exception as _e:  # pragma: no cover — env-specific
    pytest.skip(f"fastapi/starlette env mismatch: {_e}", allow_module_level=True)


class _StubBlob:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload

    def download_to_filename(self, dst: str) -> None:
        Path(dst).write_bytes(self._payload)


class _StubBucket:
    def __init__(self, blobs):
        self._blobs = blobs

    def list_blobs(self, **kwargs):
        prefix = kwargs.get("prefix")
        if not prefix:
            return list(self._blobs)
        return [b for b in self._blobs if b.name.startswith(prefix)]


class _StubClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        return self._bucket


def _install_fake_storage(bucket):
    google_pkg = types.ModuleType("google")
    google_cloud_pkg = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = lambda: _StubClient(bucket)
    google_cloud_pkg.storage = storage_mod
    google_pkg.cloud = google_cloud_pkg
    sys.modules["google"] = google_pkg
    sys.modules["google.cloud"] = google_cloud_pkg
    sys.modules["google.cloud.storage"] = storage_mod


def test_resync_noop_without_bucket(monkeypatch):
    monkeypatch.delenv("ASSETS_BUCKET", raising=False)
    r = TestClient(create_app()).post("/api/resync")
    assert r.status_code == 200
    assert r.json()["synced"] == 0


def test_resync_pulls_bundle_and_models_lists_it(tmp_path, monkeypatch):
    blobs = [
        _StubBlob("deepmaze/drqn_demo/config.json", b'{"agent_type": "drqn"}'),
        _StubBlob("deepmaze/drqn_demo/model.pt", b"weights"),
    ]
    _install_fake_storage(_StubBucket(blobs))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSETS_BUCKET", "garassino-ml-artifacts")
    monkeypatch.setenv("ASSETS_PREFIX", "deepmaze/")
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path / "assets"))

    client = TestClient(create_app())
    r = client.post("/api/resync")
    assert r.status_code == 200
    assert r.json()["synced"] == 2

    # The freshly-synced bundle is now discoverable by /api/models.
    names = [m["name"] for m in client.get("/api/models").json()["models"]]
    assert "drqn_demo" in names
