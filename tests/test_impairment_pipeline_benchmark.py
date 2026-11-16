"""Tests for the optional impairment pipeline benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "optional" / "benchmark_impairment_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("impairment_pipeline_benchmark", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
benchmark_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark_module
_SPEC.loader.exec_module(benchmark_module)

EXPECTED_IMPAIRMENTS = {
    "add_slope",
    "carrier_frequency_drift",
    "carrier_phase_noise",
    "carrier_phase_offset",
    "channel_swap",
    "clock_drift",
    "clock_jitter",
    "coarse_gain_change",
    "digital_agc",
    "fading",
    "intermodulation_products",
    "iq_imbalance",
    "nonlinear_amplifier",
    "passband_ripple",
    "quantize",
    "random_drop_samples",
    "spectral_inversion",
    "time_reversal",
}


def test_deterministic_iq_is_stable_and_indexed() -> None:
    first = benchmark_module.deterministic_iq(3, 64, 17)
    repeated = benchmark_module.deterministic_iq(3, 64, 17)
    different = benchmark_module.deterministic_iq(4, 64, 17)

    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.complex64
    assert not np.array_equal(first, different)


def test_full_impairment_pipeline_is_deterministic_and_complete() -> None:
    iq = benchmark_module.deterministic_iq(3, 256, 17)

    first = benchmark_module.apply_impairment_pipeline(iq, 3)
    repeated = benchmark_module.apply_impairment_pipeline(iq, 3)

    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.complex64
    assert set(benchmark_module.IMPAIRMENTS) == EXPECTED_IMPAIRMENTS


def test_break_even_epochs_reports_savings_only() -> None:
    assert benchmark_module.break_even_epochs(10.0, 100.0, 200.0, 1000) == pytest.approx(2.0)
    assert benchmark_module.break_even_epochs(10.0, 200.0, 100.0, 1000) is None


def test_run_benchmark_reports_all_pipeline_categories(tmp_path: Path) -> None:
    payload = benchmark_module.run_benchmark(samples=3, iq_length=64, fft_size=16, seed=7, batch_size=2, workers=0, scratch=tmp_path)

    assert payload["schema_version"] == 1
    assert payload["environment"]["python"]
    assert payload["configuration"] == {
        "samples": 3,
        "iq_length": 64,
        "fft_size": 16,
        "seed": 7,
        "batch_size": 2,
        "workers": 0,
        "dtype": "complex64",
        "impairments": list(benchmark_module.IMPAIRMENTS),
    }
    categories = {(record["workload"], record["category"]) for record in payload["stages"]}
    for workload in ("narrowband", "wideband"):
        assert {(workload, category) for category in ("generation", "impairment", "representation", "storage", "model_input", "end_to_end", "epoch")} <= categories
        assert payload["summaries"][workload]["materialization_seconds"] >= 0.0
    impairment_stages = {record["stage"] for record in payload["stages"] if record["category"] == "impairment"}
    assert impairment_stages == set(benchmark_module.IMPAIRMENTS)
    assert impairment_stages == EXPECTED_IMPAIRMENTS


def test_cli_writes_reviewable_json(tmp_path: Path) -> None:
    output = tmp_path / "results.json"

    assert benchmark_module.main(["--samples", "2", "--iq-length", "64", "--fft-size", "16", "--seed", "9", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["configuration"]["seed"] == 9
    assert len(payload["stages"]) > 0


@pytest.mark.parametrize(
    ("samples", "iq_length", "fft_size", "message"),
    [(0, 64, 16, "positive"), (1, 63, 16, "integer multiple")],
)
def test_run_benchmark_rejects_invalid_dimensions(tmp_path: Path, samples: int, iq_length: int, fft_size: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_module.run_benchmark(
            samples=samples,
            iq_length=iq_length,
            fft_size=fft_size,
            seed=1,
            batch_size=1,
            workers=0,
            scratch=tmp_path,
        )
