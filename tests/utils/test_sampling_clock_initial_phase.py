"""Tests for sampling-clock initial phase control."""

from types import SimpleNamespace

import numpy as np
import pytest

from torchsig.transforms import functional as F
from torchsig.transforms.transforms import ClockDrift, ClockJitter
from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


def _kwargs() -> dict:
    return {
        "h": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "x": np.array([1.0, 2.0], dtype=np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 0.0,
        "drift_ppm": 0.0,
        "rng": np.random.default_rng(7),
    }


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_default_initial_phase_preserves_existing_behavior(implementation) -> None:
    implicit = implementation(**_kwargs())
    explicit = implementation(**_kwargs(), initial_phase=0.0)

    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("initial_phase", [0.0, 0.25, 0.5, 0.999])
def test_representative_initial_phases_are_supported(implementation, initial_phase: float) -> None:
    output = implementation(**_kwargs(), initial_phase=initial_phase)

    assert output.dtype == np.complex64
    assert output.size > 0
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_nonzero_initial_phase_changes_initial_sampling_position(implementation) -> None:
    default = implementation(**_kwargs(), initial_phase=0.0)
    shifted = implementation(**_kwargs(), initial_phase=0.5)

    assert shifted[0] != default[0]
    assert default[0] == pytest.approx(8.0 + 0.0j)
    assert shifted[0] == pytest.approx(16.0 + 0.0j)


@pytest.mark.parametrize("functional", [F.clock_drift, F.clock_jitter])
def test_clock_functionals_propagate_initial_phase(monkeypatch, functional) -> None:
    captured = {}

    def fake_sampling_clock_impairments(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(F, "_sampling_clock_impairments", fake_sampling_clock_impairments)
    data = np.ones(8, dtype=np.complex64)

    functional(data, initial_phase=0.375, rng=np.random.default_rng(3))

    assert captured["initial_phase"] == 0.375


@pytest.mark.parametrize(
    ("transform", "functional_name"),
    [
        (ClockDrift(drift_ppm=(1.0, 1.0), initial_phase=(0.25, 0.25)), "clock_drift"),
        (ClockJitter(jitter_ppm=(1.0, 1.0), initial_phase=(0.25, 0.25)), "clock_jitter"),
    ],
)
def test_clock_transform_classes_propagate_initial_phase(
    monkeypatch,
    transform,
    functional_name: str,
) -> None:
    captured = {}

    def fake_functional(**kwargs):
        captured.update(kwargs)
        return kwargs["data"]

    monkeypatch.setattr(
        f"torchsig.transforms.transforms.F.{functional_name}",
        fake_functional,
    )
    signal = SimpleNamespace(data=np.ones(8, dtype=np.complex64))

    transform.__apply__(signal)

    assert captured["initial_phase"] == 0.25


@pytest.mark.parametrize("transform_class", [ClockDrift, ClockJitter])
def test_clock_transform_classes_draw_initial_phase_from_configured_range(
    transform_class,
) -> None:
    transform = transform_class(initial_phase=(0.2, 0.4), seed=7)

    initial_phase = transform.initial_phase_distribution()

    assert 0.2 <= initial_phase < 0.4


@pytest.mark.parametrize("transform_class", [ClockDrift, ClockJitter])
def test_clock_transform_classes_preserve_scalar_initial_phase(transform_class) -> None:
    transform = transform_class(initial_phase=0.3, seed=7)

    assert transform.initial_phase_distribution() == 0.3


@pytest.mark.parametrize("transform_class", [ClockDrift, ClockJitter])
@pytest.mark.parametrize("initial_phase", [-0.1, 1.0, np.inf, (-0.1, 0.5), (0.5, 1.0)])
def test_clock_transform_classes_reject_invalid_initial_phase(
    transform_class,
    initial_phase: float | tuple[float, float],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"initial_phase must be finite and in the interval \[0, 1\)",
    ):
        transform_class(initial_phase=initial_phase)
