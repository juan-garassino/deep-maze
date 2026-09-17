"""scripts/train_runpod.py — the result.json summary + INIT_BUNDLE wiring the
Prefect ladder depends on. Pure-logic only (no training); skips if the trainer's
heavy deps are absent in this env."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("torch")


def _load(monkeypatch, env=None):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "train_runpod_under_test", root / "scripts" / "train_runpod.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_stage(**over):
    base = {
        "agent_type": "drqn", "run_name": "drqn_t_s0_8x8", "run_id": "x",
        "eval_success_rate": 0.62, "best_success_rate": 0.66,
        "eval_revisit_rate": 0.18, "out_dir": "/tmp/a/drqn_t_s0_8x8",
        "model_path": "/tmp/a/drqn_t_s0_8x8/model.pt",
        "best_model_path": "/tmp/a/drqn_t_s0_8x8/model.best.pt",
    }
    base.update(over)
    return base


def test_summarize_shape(monkeypatch):
    mod = _load(monkeypatch)
    s = mod._summarize({"drqn": [_fake_stage()]}, "t")
    assert s["run_tag"] == "t"
    a = s["agents"]["drqn"]
    assert a["bundle_name"] == "drqn_t_s0_8x8"
    assert a["eval_success_rate"] == 0.62
    assert a["eval_revisit_rate"] == 0.18
    assert a["best_model_path"].endswith("model.best.pt")


def test_summarize_uses_last_stage(monkeypatch):
    mod = _load(monkeypatch)
    s = mod._summarize(
        {"drqn": [_fake_stage(eval_success_rate=0.1),
                  _fake_stage(eval_success_rate=0.9, run_name="drqn_t_s1_10x10")]}, "t")
    assert s["agents"]["drqn"]["bundle_name"] == "drqn_t_s1_10x10"
    assert s["agents"]["drqn"]["eval_success_rate"] == 0.9


def test_summarize_skips_empty(monkeypatch):
    mod = _load(monkeypatch)
    assert mod._summarize({"drqn": []}, "t") == {"run_tag": "t", "agents": {}}


def test_init_bundle_env_read(monkeypatch):
    assert _load(monkeypatch).INIT_BUNDLE is None
    assert _load(monkeypatch, {"INIT_BUNDLE": "/tmp/init/model.pt"}).INIT_BUNDLE == "/tmp/init/model.pt"


def test_odd_snaps_even_dims(monkeypatch):
    mod = _load(monkeypatch)
    assert [mod._odd(n) for n in (5, 6, 7, 22, 40)] == [5, 5, 7, 21, 39]


def test_default_curriculum_is_odd(monkeypatch):
    """Every stage of the default curriculum trains on an odd grid."""
    mod = _load(monkeypatch)
    for w, h, *_ in mod.STAGES:
        assert w % 2 == 1 and h % 2 == 1, f"even stage dims {w}x{h}"
