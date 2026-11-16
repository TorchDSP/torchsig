"""Profile representative impairment-experiment pipeline stages.

The benchmark is deliberately standalone and writes results only when
``--output`` is provided. It uses deterministic synthetic IQ so repeated runs
measure the same workload while separating generation, individual impairment,
representation, materialization, and model-input loading costs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torchsig.transforms import functional

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class StageResult:
    """Timing result for one independently measured pipeline stage."""

    workload: str
    category: str
    stage: str
    cold_seconds: float
    warm_seconds: float
    samples: int
    samples_per_second: float


def deterministic_iq(index: int, length: int, seed: int) -> np.ndarray:
    """Return deterministic complex64 IQ for one synthetic sample."""
    generator = np.random.default_rng(np.random.SeedSequence([seed, index]))
    time_axis = np.arange(length, dtype=np.float32)
    tone = np.exp(2j * np.pi * ((index % 31) + 1) * time_axis / length)
    noise = generator.standard_normal(length) + 1j * generator.standard_normal(length)
    return (tone + 0.05 * noise).astype(np.complex64)


def impairment_add_slope(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply the AddSlope impairment."""
    return functional.add_slope(iq)


def impairment_carrier_frequency_drift(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply CarrierFrequencyDrift with a deterministic random stream."""
    return functional.carrier_frequency_drift(iq, drift_ppm=2.0, rng=np.random.default_rng(2718 + index))


def impairment_carrier_phase_noise(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply CarrierPhaseNoise with a deterministic random stream."""
    return functional.carrier_phase_noise(iq, phase_noise_degrees=1.0, rng=np.random.default_rng(3141 + index))


def impairment_carrier_phase_offset(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply CarrierPhaseOffset."""
    return functional.phase_offset(iq, phase=np.pi / 4)


def impairment_channel_swap(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply ChannelSwap."""
    return functional.channel_swap(iq)


def impairment_clock_drift(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply ClockDrift with a deterministic random stream."""
    return functional.clock_drift(iq, drift_ppm=10.0, rng=np.random.default_rng(1618 + index))


def impairment_clock_jitter(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply ClockJitter with a deterministic random stream."""
    return functional.clock_jitter(iq, jitter_ppm=10.0, rng=np.random.default_rng(1414 + index))


def impairment_coarse_gain_change(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply CoarseGainChange."""
    return functional.coarse_gain_change(iq, gain_change_db=-2.5, start_idx=max(1, len(iq) // 2))


def impairment_digital_agc(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply DigitalAGC."""
    return functional.digital_agc(iq)


def impairment_fading(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply Fading with a deterministic random stream."""
    power_delay_profile = np.full(16, 1.0 / 16, dtype=np.float32)
    return functional.fading(iq, coherence_bandwidth=0.1, power_delay_profile=power_delay_profile, rng=np.random.default_rng(1732 + index))


def impairment_intermodulation_products(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply IntermodulationProducts."""
    return functional.intermodulation_products(iq)


def impairment_iq_imbalance(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply IQImbalance."""
    return functional.iq_imbalance(iq, amplitude_imbalance=1.0, phase_imbalance=0.1, dc_offset_db=-40.0, dc_offset_phase_rads=0.0)


def impairment_nonlinear_amplifier(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply NonlinearAmplifier."""
    return functional.nonlinear_amplifier(iq)


def impairment_passband_ripple(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply PassbandRipple with a deterministic random stream."""
    return functional.passband_ripple(iq, num_taps=65, max_ripple_db=1.0, rng=np.random.default_rng(2236 + index))


def impairment_quantize(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply Quantize."""
    return functional.quantize(iq, num_bits=8, ref_level_adjustment_db=0.0, rounding_mode="floor")


def impairment_random_drop_samples(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply RandomDropSamples with deterministic locations and fill."""
    start = max(1, len(iq) // 3)
    size = max(1, min(len(iq) - start - 1, len(iq) // 100))
    return functional.drop_samples(iq.copy(), np.array([start]), np.array([size]), "ffill")


def impairment_spectral_inversion(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply SpectralInversion."""
    return functional.spectral_inversion(iq)


def impairment_time_reversal(iq: np.ndarray, _index: int) -> np.ndarray:
    """Apply TimeReversal."""
    return functional.time_reversal(iq)


IMPAIRMENTS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "add_slope": impairment_add_slope,
    "carrier_frequency_drift": impairment_carrier_frequency_drift,
    "carrier_phase_noise": impairment_carrier_phase_noise,
    "carrier_phase_offset": impairment_carrier_phase_offset,
    "channel_swap": impairment_channel_swap,
    "clock_drift": impairment_clock_drift,
    "clock_jitter": impairment_clock_jitter,
    "coarse_gain_change": impairment_coarse_gain_change,
    "digital_agc": impairment_digital_agc,
    "fading": impairment_fading,
    "intermodulation_products": impairment_intermodulation_products,
    "iq_imbalance": impairment_iq_imbalance,
    "nonlinear_amplifier": impairment_nonlinear_amplifier,
    "passband_ripple": impairment_passband_ripple,
    "quantize": impairment_quantize,
    "random_drop_samples": impairment_random_drop_samples,
    "spectral_inversion": impairment_spectral_inversion,
    "time_reversal": impairment_time_reversal,
}


def apply_impairment_pipeline(iq: np.ndarray, index: int) -> np.ndarray:
    """Apply every unique impairment used by the default Impairments tree."""
    impaired = iq
    for impairment in IMPAIRMENTS.values():
        impaired = impairment(impaired, index)
    return impaired


def represent_iq(iq: np.ndarray) -> np.ndarray:
    """Convert complex IQ to a two-channel float representation."""
    return np.stack((iq.real, iq.imag)).astype(np.float32)


def represent_spectrogram(iq: np.ndarray, fft_size: int) -> np.ndarray:
    """Return a deterministic magnitude spectrogram."""
    usable = iq[: len(iq) - (len(iq) % fft_size)]
    frames = usable.reshape(-1, fft_size)
    return np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)).astype(np.float32)


def measure_stage(
    workload: str,
    category: str,
    stage: str,
    operation: Callable[[int], Any],
    *,
    samples: int,
) -> StageResult:
    """Measure one cold call followed by a warm sample loop."""
    start = time.perf_counter()
    operation(0)
    cold_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for index in range(samples):
        operation(index)
    warm_seconds = time.perf_counter() - start
    rate = samples / warm_seconds if warm_seconds > 0.0 else float("inf")
    return StageResult(workload, category, stage, cold_seconds, warm_seconds, samples, rate)


def break_even_epochs(
    materialization_seconds: float,
    online_samples_per_second: float,
    materialized_samples_per_second: float,
    samples: int,
) -> float | None:
    """Return epochs needed to recover materialization cost, if possible."""
    online_epoch = samples / online_samples_per_second
    materialized_epoch = samples / materialized_samples_per_second
    savings = online_epoch - materialized_epoch
    return None if savings <= 0.0 else materialization_seconds / savings


def environment_metadata() -> dict[str, Any]:
    """Return reviewable runtime metadata for benchmark comparison."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "logical_cpu_count": os.cpu_count(),
    }


class PipelineDataset(Dataset[torch.Tensor]):
    """Generate deterministic, impaired model inputs on demand."""

    def __init__(self, samples: int, iq_length: int, seed: int, workload: str, fft_size: int) -> None:
        self.samples = samples
        self.iq_length = iq_length
        self.seed = seed
        self.workload = workload
        self.fft_size = fft_size

    def __len__(self) -> int:
        """Return the number of samples in an epoch."""
        return self.samples

    def __getitem__(self, index: int) -> torch.Tensor:
        """Generate one deterministic model input."""
        iq = deterministic_iq(index, self.iq_length, self.seed)
        impaired = apply_impairment_pipeline(iq, index)
        represented = represent_iq(impaired) if self.workload == "narrowband" else represent_spectrogram(impaired, self.fft_size)
        return torch.from_numpy(represented)


class MaterializedDataset(Dataset[torch.Tensor]):
    """Load model inputs from a memory-mapped NumPy array."""

    def __init__(self, path: Path) -> None:
        self.array = np.load(path, mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        """Return the number of materialized samples."""
        return len(self.array)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load one model input into writable memory."""
        return torch.from_numpy(np.array(self.array[index], copy=True))


def measure_epoch(
    workload: str,
    stage: str,
    dataset: Dataset[torch.Tensor],
    *,
    batch_size: int,
    workers: int,
) -> StageResult:
    """Measure cold and warm complete DataLoader epochs."""

    def run_epoch() -> None:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        for _batch in loader:
            pass

    start = time.perf_counter()
    run_epoch()
    cold_seconds = time.perf_counter() - start
    start = time.perf_counter()
    run_epoch()
    warm_seconds = time.perf_counter() - start
    samples = len(dataset)
    rate = samples / warm_seconds if warm_seconds > 0.0 else float("inf")
    return StageResult(workload, "epoch", stage, cold_seconds, warm_seconds, samples, rate)


def profile_workload(
    workload: str,
    *,
    samples: int,
    iq_length: int,
    fft_size: int,
    seed: int,
    batch_size: int,
    workers: int,
    scratch: Path,
) -> tuple[list[StageResult], dict[str, float | None]]:
    """Profile one IQ or spectrogram impairment workload."""
    if workload not in {"narrowband", "wideband"}:
        raise ValueError(f"unknown workload {workload!r}")

    def represent(iq: np.ndarray) -> np.ndarray:
        return represent_iq(iq) if workload == "narrowband" else represent_spectrogram(iq, fft_size)

    def online(index: int) -> np.ndarray:
        iq = deterministic_iq(index, iq_length, seed)
        return represent(apply_impairment_pipeline(iq, index))

    clean = [deterministic_iq(index, iq_length, seed) for index in range(samples)]

    results = [
        measure_stage(workload, "generation", "clean_iq", lambda index: deterministic_iq(index, iq_length, seed), samples=samples),
    ]
    for name, impairment in IMPAIRMENTS.items():
        results.append(measure_stage(workload, "impairment", name, lambda index, fn=impairment: fn(clean[index % samples], index), samples=samples))
    results.append(measure_stage(workload, "representation", workload, lambda index: represent(clean[index % samples]), samples=samples))

    online_result = measure_stage(workload, "end_to_end", "online", online, samples=samples)
    results.append(online_result)

    materialized_path = scratch / f"{workload}.npy"
    start = time.perf_counter()
    materialized = np.stack([online(index) for index in range(samples)])
    np.save(materialized_path, materialized, allow_pickle=False)
    materialization_seconds = time.perf_counter() - start
    loaded = np.load(materialized_path, mmap_mode="r", allow_pickle=False)
    materialized_result = measure_stage(workload, "storage", "materialized_load", lambda index: np.asarray(loaded[index % samples]), samples=samples)
    results.append(materialized_result)
    results.append(measure_stage(workload, "model_input", "tensor_conversion", lambda index: torch.from_numpy(np.array(loaded[index % samples], copy=True)), samples=samples))
    online_epoch = measure_epoch(
        workload,
        "online",
        PipelineDataset(samples, iq_length, seed, workload, fft_size),
        batch_size=batch_size,
        workers=workers,
    )
    materialized_epoch = measure_epoch(
        workload,
        "materialized",
        MaterializedDataset(materialized_path),
        batch_size=batch_size,
        workers=workers,
    )
    results.extend((online_epoch, materialized_epoch))

    summary = {
        "materialization_seconds": materialization_seconds,
        "online_samples_per_second": online_epoch.samples_per_second,
        "materialized_samples_per_second": materialized_epoch.samples_per_second,
        "break_even_epochs": break_even_epochs(
            materialization_seconds,
            online_epoch.samples_per_second,
            materialized_epoch.samples_per_second,
            samples,
        ),
    }
    return results, summary


def run_benchmark(*, samples: int, iq_length: int, fft_size: int, seed: int, batch_size: int, workers: int, scratch: Path) -> dict[str, Any]:
    """Run deterministic narrowband and wideband benchmark matrices."""
    if samples < 1 or iq_length < 1 or fft_size < 1 or batch_size < 1 or workers < 0:
        raise ValueError("samples, iq_length, fft_size, and batch_size must be positive; workers cannot be negative")
    if iq_length < fft_size or iq_length % fft_size:
        raise ValueError("iq_length must be an integer multiple of fft_size")

    records: list[StageResult] = []
    summaries = {}
    for workload in ("narrowband", "wideband"):
        workload_results, summary = profile_workload(
            workload,
            samples=samples,
            iq_length=iq_length,
            fft_size=fft_size,
            seed=seed,
            batch_size=batch_size,
            workers=workers,
            scratch=scratch,
        )
        records.extend(workload_results)
        summaries[workload] = summary
    return {
        "schema_version": 1,
        "environment": environment_metadata(),
        "configuration": {
            "samples": samples,
            "iq_length": iq_length,
            "fft_size": fft_size,
            "seed": seed,
            "batch_size": batch_size,
            "workers": workers,
            "dtype": "complex64",
            "impairments": list(IMPAIRMENTS),
        },
        "stages": [asdict(record) for record in records],
        "summaries": summaries,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--iq-length", type=int, default=4096)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and optionally write structured JSON results."""
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="torchsig-impairment-benchmark-") as directory:
        payload = run_benchmark(
            samples=args.samples,
            iq_length=args.iq_length,
            fft_size=args.fft_size,
            seed=args.seed,
            batch_size=args.batch_size,
            workers=args.workers,
            scratch=Path(directory),
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
