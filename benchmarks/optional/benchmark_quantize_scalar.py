"""Compare legacy, optimized-kernel, and transform-level scalar quantization."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.transforms.functional import quantize
from torchsig.transforms.transforms import Quantize
from torchsig.utils.dsp import TorchSigComplexDataType, TorchSigRealDataType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any


def legacy_quantize(data: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """Run the pre-464 scalar quantization algorithm."""
    levels = np.arange(-(2**num_bits) // 2, (2**num_bits) // 2) / ((2**num_bits) // 2)
    distance = levels[1] - levels[0]
    thresholds = levels + distance / 2
    maximum = max(np.max(np.abs(data.real)), np.max(np.abs(data.imag)))
    if maximum == 0:
        return np.zeros(data.shape, dtype=TorchSigComplexDataType)
    real = np.zeros(len(data), dtype=TorchSigRealDataType)
    imag = np.zeros(len(data), dtype=TorchSigRealDataType)
    scaled = data / maximum
    for component, output in ((scaled.real, real), (scaled.imag, imag)):
        negative = np.where(component <= thresholds[0])[0]
        positive = np.where(component >= thresholds[-1])[0]
        output[negative] = levels[0]
        output[positive] = levels[-1]
        remaining = np.setdiff1d(np.arange(len(data)), negative)
        remaining = np.setdiff1d(remaining, positive)
        output[remaining] = levels[np.digitize(component[remaining], thresholds)]
    return ((real + 1j * imag) * maximum).astype(TorchSigComplexDataType)


def _measure(operation: Callable[[], Any], *, warmups: int, repeats: int) -> tuple[float, int]:
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    try:
        operation()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return statistics.median(samples), peak_bytes


def run_benchmark(*, lengths: Sequence[int], warmups: int = 2, repeats: int = 7) -> list[dict[str, Any]]:
    """Measure legacy, optimized-kernel, and public-transform execution."""
    rng = np.random.default_rng(464)
    transform = Quantize(num_bits=(8, 8), ref_level_adjustment_db=(0.0, 0.0), rounding_mode=["floor"])
    results = []
    for length in lengths:
        data = (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(np.complex64)
        signal = Signal(data=data)
        legacy_seconds, legacy_peak = _measure(lambda value=data: legacy_quantize(value), warmups=warmups, repeats=repeats)
        kernel_seconds, kernel_peak = _measure(lambda value=data: quantize(value, 8, 0.0, "floor"), warmups=warmups, repeats=repeats)
        transform_seconds, transform_peak = _measure(lambda value=signal: transform(value.copy(preserve_parent=False)), warmups=warmups, repeats=repeats)
        np.testing.assert_array_equal(legacy_quantize(data), quantize(data, 8, 0.0, "floor"))
        results.append(
            {
                "length": length,
                "legacy_median_seconds": legacy_seconds,
                "kernel_median_seconds": kernel_seconds,
                "transform_median_seconds": transform_seconds,
                "speedup": legacy_seconds / kernel_seconds,
                "legacy_peak_bytes": legacy_peak,
                "kernel_peak_bytes": kernel_peak,
                "transform_peak_bytes": transform_peak,
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and optionally write its JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 16384, 65536])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    results = run_benchmark(lengths=args.lengths, warmups=args.warmups, repeats=args.repeats)
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
