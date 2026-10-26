"""Tests for jitter-model selection through the public transform APIs."""

# ruff: noqa: INP001

from types import SimpleNamespace

import numpy as np
import pytest

from torchsig.transforms import functional as F  # noqa: N812
from torchsig.transforms.transforms import ClockJitter


@pytest.mark.parametrize("jitter_model", ["independent", "period", "ar1"])
def test_clock_jitter_functional_forwards_model_parameters(monkeypatch, jitter_model: str) -> None:
    captured = {}

    def fake_sampling_clock_impairments(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(F, "_sampling_clock_impairments", fake_sampling_clock_impairments)
    data = np.ones(16, dtype=np.complex64)

    F.clock_jitter(
        data,
        jitter_ppm=25.0,
        jitter_model=jitter_model,
        jitter_alpha=-0.4,
        rng=np.random.default_rng(3),
    )

    assert captured["jitter_model"] == jitter_model
    assert captured["jitter_alpha"] == -0.4


@pytest.mark.parametrize("jitter_model", ["independent", "period", "ar1"])
def test_clock_jitter_transform_forwards_model_parameters(monkeypatch, jitter_model: str) -> None:
    captured = {}

    def fake_clock_jitter(**kwargs):
        captured.update(kwargs)
        return kwargs["data"]

    monkeypatch.setattr(F, "clock_jitter", fake_clock_jitter)
    transform = ClockJitter(
        jitter_ppm=(25.0, 25.0),
        jitter_model=jitter_model,
        jitter_alpha=-0.4,
        seed=7,
    )

    transform.__apply__(SimpleNamespace(data=np.ones(16, dtype=np.complex64)))

    assert transform.jitter_model == jitter_model
    assert transform.jitter_alpha == -0.4
    assert captured["jitter_model"] == jitter_model
    assert captured["jitter_alpha"] == -0.4


@pytest.mark.parametrize("public_api", [F.clock_jitter, ClockJitter])
def test_public_jitter_apis_reject_invalid_model(public_api) -> None:
    kwargs = {"data": np.ones(8, dtype=np.complex64)} if public_api is F.clock_jitter else {}

    with pytest.raises(ValueError, match="jitter_model"):
        public_api(jitter_model="unknown", **kwargs)


@pytest.mark.parametrize("public_api", [F.clock_jitter, ClockJitter])
@pytest.mark.parametrize("jitter_alpha", [-1.0, 1.0, np.inf, np.nan])
def test_public_jitter_apis_reject_invalid_alpha(public_api, jitter_alpha: float) -> None:
    kwargs = {"data": np.ones(8, dtype=np.complex64)} if public_api is F.clock_jitter else {}

    with pytest.raises(ValueError, match="jitter_alpha"):
        public_api(jitter_alpha=jitter_alpha, **kwargs)


@pytest.mark.parametrize("jitter_model", ["independent", "period", "ar1"])
def test_clock_jitter_models_are_seeded_and_reproducible(jitter_model: str) -> None:
    time = np.arange(256)
    data = np.exp(2j * np.pi * 0.1 * time).astype(np.complex64)
    kwargs = {
        "jitter_ppm": 20_000.0,
        "jitter_model": jitter_model,
        "jitter_alpha": 0.6,
    }

    first = F.clock_jitter(data, rng=np.random.default_rng(19), **kwargs)
    second = F.clock_jitter(data, rng=np.random.default_rng(19), **kwargs)

    np.testing.assert_array_equal(first, second)


def test_clock_jitter_default_model_is_unchanged(monkeypatch) -> None:
    captured = {}

    def fake_sampling_clock_impairments(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(F, "_sampling_clock_impairments", fake_sampling_clock_impairments)

    F.clock_jitter(np.ones(8, dtype=np.complex64), rng=np.random.default_rng(5))

    assert captured["jitter_model"] == "independent"
    assert captured["jitter_alpha"] == 0.9
