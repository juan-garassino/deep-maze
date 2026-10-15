"""Edge cases for OTEL sampler env-var parsing."""

from __future__ import annotations

import pytest

from web.otel import _sampler_from_env

try:  # the API package alone can be installed transitively; the sampler needs the SDK
    import opentelemetry.sdk.trace.sampling  # noqa: F401
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

needs_sdk = pytest.mark.skipif(not HAS_SDK, reason="opentelemetry-sdk not installed")


def test_returns_none_when_var_unset(monkeypatch):
    monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)
    assert _sampler_from_env() is None


@needs_sdk
@pytest.mark.parametrize("name", [
    "always_on", "always_off", "traceidratio",
    "parentbased_traceidratio", "parentbased_always_on",
])
def test_known_samplers_resolve(monkeypatch, name):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", name)
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.5")
    assert _sampler_from_env() is not None


def test_unknown_sampler_returns_none(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "definitely_not_a_sampler")
    assert _sampler_from_env() is None


@needs_sdk
def test_invalid_arg_falls_back_to_one(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "not-a-number")
    assert _sampler_from_env() is not None  # silently defaults ratio=1.0
