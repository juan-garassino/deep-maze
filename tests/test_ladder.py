"""flows/ladder.py — the pure climb gate + the rung table. Loaded standalone
(dataclasses only, no prefect/httpx) so it runs in the base test env."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ladder_under_test", root / "flows" / "ladder.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass under `from __future__ import annotations`
    # resolves field types via sys.modules[cls.__module__] at class-creation.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_has_learned_gate():
    m = _load()
    assert m.has_learned({"eval_success_rate": 0.6, "best_success_rate": 0.3}, 0.5)
    assert m.has_learned({"eval_success_rate": 0.2, "best_success_rate": 0.55}, 0.5)
    assert not m.has_learned({"eval_success_rate": 0.2, "best_success_rate": 0.1}, 0.5)
    assert not m.has_learned(None, 0.5)
    assert not m.has_learned({}, 0.5)


def test_has_learned_tolerates_none_values():
    m = _load()
    assert not m.has_learned({"eval_success_rate": None, "best_success_rate": None}, 0.1)
    assert m.has_learned({"eval_success_rate": None, "best_success_rate": 0.5}, 0.4)


def test_ladder_table_wellformed():
    m = _load()
    assert len(m.LADDER) >= 3
    for rung in m.LADDER:
        parts = rung.curriculum.split(",")
        assert len(parts) >= 5  # W,H,n_treasures,episodes,max_steps[,seq_len]
        assert all(p.strip().isdigit() for p in parts)
        assert rung.agent in ("drqn", "dtqn")
        assert 0.0 <= rung.learn_gate <= 1.0
    # The smoke rung always "learns" (gate 0) and forces NANO.
    assert m.NANO_RUNG.learn_gate == 0.0
    assert m.NANO_RUNG.extra_env.get("NANO") == "true"


def test_ladder_dims_are_odd():
    """Every rung trains on an odd grid (DFS tiles cleanly only on odd sizes)."""
    m = _load()
    for rung in [*m.LADDER, m.NANO_RUNG]:
        w, h = (int(x) for x in rung.curriculum.split(",")[:2])
        assert w % 2 == 1 and h % 2 == 1, f"{rung.name} has even dims {w}x{h}"
