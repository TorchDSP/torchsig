"""Standalone stage, throughput, and memory benchmark for fading.

Run from the repository root with::

    python benchmarks/optional/benchmark_fading.py

The benchmark prints JSON and writes no artifact unless ``--output`` is used.
Inputs and RNG seeds are deterministic and constructed outside timed calls.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy import signal as sp
from scipy.interpolate import interp1d

import torchsig
from torchsig.transforms.functional import _fading_reference, fading

if TYPE_CHECKING:
    from collections.abc import Callable

_INTERPOLATION_FACTOR = 100


def deterministic_iq(length: int) -> np.ndarray:
    """Return deterministic complex64 benchmark input."""
    rng = np.random.default_rng(2027)
    return (rng.standard_normal(length) + 1j * rng.standard_normal(length)).astype(np.complex64)


def _reference_stages(data: np.ndarray, coherence_bandwidth: float, profile: np.ndarray, seed: int) -> dict[str, float]:
    """Time the distinct stages of the pre-483 implementation once."""
    stages: dict[str, float] = {}
    started = time.perf_counter()
    num_taps = int(np.ceil(1.0 / coherence_bandwidth))
    rng = np.random.default_rng(seed)
    initial_taps = rng.standard_normal(num_taps) + 1j * rng.standard_normal(num_taps)
    stages["tap_generation_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    old_time = np.linspace(0, 1.0, num_taps)
    new_time = np.linspace(0, 1.0, _INTERPOLATION_FACTOR * num_taps)
    taps = interp1d(old_time, initial_taps.real)(new_time) + 1j * interp1d(old_time, initial_taps.imag)(new_time)
    taps *= np.sqrt(np.interp(new_time, np.linspace(0, 1.0, len(profile)), profile))
    stages["interpolation_setup_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    output = sp.upfirdn(taps, data, up=_INTERPOLATION_FACTOR, down=_INTERPOLATION_FACTOR)[-len(data) :]
    stages["filtering_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    output *= np.linalg.norm(data) / np.linalg.norm(output)
    stages["renormalization_seconds"] = time.perf_counter() - started
    return stages


def _optimized_stages(data: np.ndarray, coherence_bandwidth: float, profile: np.ndarray, seed: int) -> dict[str, float]:
    """Time the distinct stages of the optimized implementation once."""
    stages: dict[str, float] = {}
    started = time.perf_counter()
    num_taps = int(np.ceil(1.0 / coherence_bandwidth))
    rng = np.random.default_rng(seed)
    initial_taps = rng.standard_normal(num_taps) + 1j * rng.standard_normal(num_taps)
    stages["tap_generation_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    old_time = np.linspace(0, 1.0, num_taps)
    effective_time = np.arange(num_taps, dtype=np.float64) * _INTERPOLATION_FACTOR / (_INTERPOLATION_FACTOR * num_taps - 1)
    taps = np.interp(effective_time, old_time, initial_taps.real) + 1j * np.interp(effective_time, old_time, initial_taps.imag)
    taps *= np.sqrt(np.interp(effective_time, np.linspace(0, 1.0, len(profile)), profile))
    stages["interpolation_setup_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    output = np.convolve(data, taps, mode="full")[-len(data) :]
    stages["filtering_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    output *= np.linalg.norm(data) / np.linalg.norm(output)
    stages["renormalization_seconds"] = time.perf_counter() - started
    return stages


def _measure(call: Callable[[], np.ndarray], repeats: int) -> dict[str, float | int]:
    """Measure first-call, warm median, and traced peak memory."""
    started = time.perf_counter()
    call()
    first_call = time.perf_counter() - started

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)

    tracemalloc.start()
    call()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "first_call_seconds": first_call,
        "warm_median_seconds": statistics.median(timings),
        "peak_traced_bytes": peak_bytes,
    }


def benchmark_length(length: int, repeats: int) -> dict[str, Any]:
    """Benchmark reference and optimized fading for one input length."""
    data = deterministic_iq(length)
    profile = np.full(16, 1.0 / 16, dtype=np.float32)
    coherence_bandwidth = 0.1
    seed = 1732

    def reference_call() -> np.ndarray:
        return _fading_reference(data, coherence_bandwidth, profile, np.random.default_rng(seed))

    def optimized_call() -> np.ndarray:
        return fading(data, coherence_bandwidth, profile, np.random.default_rng(seed))

    reference_output = reference_call()
    optimized_output = optimized_call()
    if not np.array_equal(reference_output, optimized_output):
        raise AssertionError("optimized fading does not exactly match the reference")

    reference = _measure(reference_call, repeats)
    optimized = _measure(optimized_call, repeats)
    reference["stages"] = _reference_stages(data, coherence_bandwidth, profile, seed)
    optimized["stages"] = _optimized_stages(data, coherence_bandwidth, profile, seed)
    return {
        "length": length,
        "coherence_bandwidth": coherence_bandwidth,
        "power_delay_profile_length": len(profile),
        "reference": reference,
        "optimized": optimized,
        "warm_speedup": reference["warm_median_seconds"] / optimized["warm_median_seconds"],
        "peak_memory_ratio": optimized["peak_traced_bytes"] / reference["peak_traced_bytes"],
    }


def main() -> None:
    """Run the standalone benchmark and print or save JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 4096, 65536])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torchsig": torchsig.__version__,
        },
        "repeats": args.repeats,
        "workloads": [benchmark_length(length, args.repeats) for length in args.lengths],
    }
    serialized = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
