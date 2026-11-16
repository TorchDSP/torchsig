"""Benchmark online versus materialized deterministic Stage-2 training.

The repository does not contain the application-specific Stage-2 pipeline, so
this script ships a representative deterministic fallback and accepts import
hooks for the production transform and model factories. Run ``--help`` for the
adapter contract and full benchmark matrix options.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import multiprocessing
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data._utils.collate import default_collate

from torchsig.datasets import StructuredHDF5Dataset, materialize_structured_dataset
from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import MetadataAttributeError
from torchsig.utils.file_handlers import HDF5Reader, HomogeneousHDF5Reader, HomogeneousHDF5Writer, PackedHDF5Reader

Access = Literal["shuffled", "sequential"]
Layout = Literal["full", "separate"]
Pipeline = Literal["online", "materialized", "device_only"]
Workload = Literal["combined", "iq", "spectrogram", "detection"]


def _import_object(spec: str) -> Any:
    """Import ``module:attribute`` without modifying the application package."""
    try:
        module_name, attribute = spec.split(":", 1)
    except ValueError as error:
        raise ValueError(f"Import hook must use module:attribute syntax: {spec!r}") from error
    return getattr(importlib.import_module(module_name), attribute)


def _reader_class(root: Path) -> type:
    """Select the existing TorchSig Signal reader from HDF5 attributes."""
    with h5py.File(root / "data.h5", "r") as handle:
        format_name = handle.attrs.get("format")
    if format_name == "torchsig-packed":
        return PackedHDF5Reader
    if format_name == "torchsig-homogeneous":
        return HomogeneousHDF5Reader
    return HDF5Reader


def _synthetic_manifest(samples: int, iq_length: int, seed: int) -> dict[str, Any]:
    return {"format": "torchsig-stage2-synthetic-v1", "samples": samples, "iq_length": iq_length, "seed": seed}


def _generate_synthetic_source(root: Path, samples: int, iq_length: int, seed: int, batch_size: int) -> float:
    """Generate a deterministic static homogeneous TorchSig source dataset."""
    manifest = _synthetic_manifest(samples, iq_length, seed)
    manifest_path = root / "synthetic_manifest.json"
    if manifest_path.exists() and (root / "data.h5").exists():
        try:
            if json.loads(manifest_path.read_text()) == manifest:
                print(f"Reusing synthetic source: {root}")
                return 0.0
        except (OSError, json.JSONDecodeError):
            pass

    start = time.perf_counter()
    with HomogeneousHDF5Writer(root, compression=None, shuffle=False, fletcher32=False, chunk_samples=1) as writer:
        for batch_index, batch_start in enumerate(range(0, samples, batch_size)):
            batch = []
            for index in range(batch_start, min(batch_start + batch_size, samples)):
                generator = np.random.default_rng(seed + index)
                time_axis = np.arange(iq_length, dtype=np.float32)
                tone = np.exp(2j * np.pi * ((index % 127) + 1) * time_axis / iq_length)
                noise = generator.standard_normal(iq_length) + 1j * generator.standard_normal(iq_length)
                iq = (tone + 0.1 * noise).astype(np.complex64)
                batch.append(Signal(data=iq, class_index=index % 64, family_index=index % 8, sample_index=index))
            writer.write(batch_index, batch)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    elapsed = time.perf_counter() - start
    print(f"Generated {samples} synthetic IQ samples at {root} in {elapsed:.2f} seconds")
    return elapsed


def _metadata_int(signal: Signal, names: Sequence[str], default: int) -> int:
    for name in names:
        try:
            value = signal[name]
        except MetadataAttributeError:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def fallback_stage2_transform(signal: Signal, index: int, workload: Workload = "combined") -> dict[str, Any]:
    """Build deterministic typed Stage-2-like features from one Signal."""
    iq = np.asarray(signal.data).reshape(-1).astype(np.complex64, copy=False)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    normalized = iq / max(rms, np.finfo(np.float32).eps)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(normalized))).astype(np.float32)
    spectrum = np.log1p(spectrum)
    spectral = np.interp(np.linspace(0, len(spectrum) - 1, 128), np.arange(len(spectrum)), spectrum).astype(np.float32)
    amplitude = np.abs(normalized).astype(np.float32)
    histogram = np.histogram(amplitude, bins=32, range=(0.0, max(float(amplitude.max()), 1.0)), density=True)[0].astype(np.float32)
    scalar = np.array(
        [
            rms,
            float(amplitude.mean()),
            float(amplitude.std()),
            float(spectrum.mean()),
            float(spectrum.std()),
            float(np.max(amplitude)),
        ],
        dtype=np.float32,
    )
    class_index = _metadata_int(signal, ("class_index",), index % 64)
    family_index = _metadata_int(signal, ("family_index", "modulation_family_index"), class_index % 8)
    combined = {
        "features": {
            "iq": np.stack((normalized.real, normalized.imag)).astype(np.float32),
            "spectral": spectral,
            "scalar": scalar,
            "amplitude_histogram": histogram,
        },
        "labels": {
            "class_index": np.int64(class_index),
            "family_index": np.int64(family_index),
        },
    }
    if workload == "combined":
        return combined
    if workload == "iq":
        return {"features": combined["features"]["iq"], "labels": np.int64(class_index)}
    image = np.resize(combined["features"]["iq"], (2, 64, 64)).astype(np.float32)
    if workload == "spectrogram":
        return {"features": image, "labels": np.int64(class_index)}
    if workload == "detection":
        return {
            "features": image,
            "labels": {
                "class_index": np.int64(class_index),
                "boxes": np.array(
                    [[0.10, 0.15, 0.35, 0.40], [0.45, 0.30, 0.80, 0.75], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
                "classes": np.array([class_index, family_index, -1, -1], dtype=np.int64),
                "valid": np.array([True, True, False, False]),
            },
        }
    raise ValueError(f"Unknown workload: {workload}")


class OnlineStage2Dataset(Dataset):
    """Apply the selected deterministic Stage-2 transform to static Signals."""

    def __init__(self, root: Path, transform_spec: str | None, limit: int | None, workload: Workload = "combined") -> None:
        self.source = StaticTorchSigDataset(root=str(root), file_handler_class=_reader_class(root), transforms=[], target_labels=None)
        self.length = min(len(self.source), limit) if limit is not None else len(self.source)
        if transform_spec is None:
            self.transform = lambda signal, index: fallback_stage2_transform(signal, index, workload)
            self.transform_name = f"built-in fallback ({workload})"
        else:
            builder = _import_object(transform_spec)
            built = builder()
            if isinstance(built, Sequence) and not isinstance(built, (str, bytes)):
                transforms = tuple(built)

                def apply_transforms(signal: Signal, index: int) -> Any:
                    value: Any = signal
                    for transform in transforms:
                        value = transform(value)
                    return value

                self.transform = apply_transforms
            elif callable(built):

                def apply_transform(signal: Signal, _index: int) -> Any:
                    return built(signal)

                self.transform = apply_transform
            else:
                raise TypeError("Stage-2 transform factory must return a callable or sequence of callables")
            self.transform_name = transform_spec

    def __len__(self) -> int:
        """Return the selected source sample count."""
        return self.length

    def __getitem__(self, index: int) -> Any:
        """Read and transform one static TorchSig sample."""
        return self.transform(self.source[index], index)


def _numeric_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _numeric_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _numeric_leaves(item)]
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return [tensor]


def _features_and_labels(batch: Any) -> tuple[Any, torch.Tensor]:
    """Extract feature containers and class labels from fallback or tuple output."""
    if isinstance(batch, Mapping) and "features" in batch and "labels" in batch:
        labels = batch["labels"]
        if isinstance(labels, Mapping):
            labels = labels["class_index"]
        return batch["features"], torch.as_tensor(labels, dtype=torch.long)
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], torch.as_tensor(batch[1], dtype=torch.long)
    raise TypeError("Stage-2 samples must be {'features', 'labels'} mappings or (features, labels) pairs")


def _flatten_features(features: Any, device: torch.device) -> torch.Tensor:
    leaves = [leaf.to(device=device, dtype=torch.float32, non_blocking=True) for leaf in _numeric_leaves(features)]
    batch_size = len(leaves[0])
    return torch.cat([leaf.reshape(batch_size, -1) for leaf in leaves], dim=1)


class FallbackStage2Model(nn.Module):
    """Small classifier used only when no production model factory is given."""

    def __init__(self, input_features: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_features, 512), nn.GELU(), nn.Linear(512, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _build_model(model_spec: str | None, input_features: int, classes: int) -> nn.Module:
    if model_spec is None:
        return FallbackStage2Model(input_features, classes)
    factory = _import_object(model_spec)
    try:
        return factory(input_features=input_features, classes=classes)
    except TypeError:
        return factory()


def _split_indices(length: int, seed: int) -> dict[str, list[int]]:
    train_length = int(length * 0.8)
    validation_length = int(length * 0.1)
    test_length = length - train_length - validation_length
    splits = random_split(range(length), [train_length, validation_length, test_length], generator=torch.Generator().manual_seed(seed))
    return {name: list(split.indices) for name, split in zip(("train", "validation", "test"), splits, strict=True)}


def estimate_storage_bytes(sample: Any, samples: int, copies: int) -> int:
    """Estimate uncompressed leaf storage before materialization."""
    return sum(np.asarray(leaf).nbytes for leaf in _plain_leaves(sample)) * samples * copies


def _plain_leaves(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _plain_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _plain_leaves(item)]
    return [value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value]


def _assert_equivalent(expected: Any, actual: Any, path: str = "$") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise AssertionError(f"Structure mismatch at {path}")
        for key in expected:
            _assert_equivalent(expected[key], actual[key], f"{path}[{key!r}]")
        return
    if isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(expected) != len(actual):
            raise AssertionError(f"Structure mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(left, right, f"{path}[{index}]")
        return
    left, right = np.asarray(expected), np.asarray(actual)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(f"Shape or dtype mismatch at {path}: {left.shape}/{left.dtype} != {right.shape}/{right.dtype}")
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(left.dtype, np.complexfloating):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=path)
    else:
        np.testing.assert_array_equal(left, right, err_msg=path)


def _assert_model_equivalent(online_samples: list[Any], materialized_samples: list[Any], model_spec: str | None, classes: int, seed: int) -> None:
    """Compare evaluation outputs for corresponding online/materialized batches."""
    online_batch = default_collate(online_samples)
    materialized_batch = default_collate(materialized_samples)
    online_features, _ = _features_and_labels(online_batch)
    materialized_features, _ = _features_and_labels(materialized_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_flat = _flatten_features(online_features, device)
    materialized_flat = _flatten_features(materialized_features, device)
    torch.manual_seed(seed)
    model = _build_model(model_spec, online_flat.shape[1], classes).to(device).eval()
    with torch.no_grad():
        expected = model(online_flat)
        actual = model(materialized_flat)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _manifest_matches(root: Path, expected: Mapping[str, Any]) -> bool:
    manifest = root / "benchmark_manifest.json"
    if not manifest.exists() or not (root / "data.h5").exists():
        return False
    try:
        actual = json.loads(manifest.read_text())
        actual.pop("materialization_seconds", None)
    except (OSError, json.JSONDecodeError):
        return False
    else:
        return actual == expected


def _materialize(
    source: Dataset,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    compression: str | None,
    chunk_samples: int,
    reuse: bool,
) -> tuple[float, int]:
    if reuse and _manifest_matches(root, manifest):
        stored_manifest = json.loads((root / "benchmark_manifest.json").read_text())
        return float(stored_manifest.get("materialization_seconds", 0.0)), (root / "data.h5").stat().st_size
    start = time.perf_counter()
    result = materialize_structured_dataset(
        source,
        root,
        batch_size=batch_size,
        progress=True,
        overwrite=True,
        validation="sampled",
        validation_samples=min(32, len(source)),
        writer_kwargs={"compression": compression, "shuffle": compression is not None, "fletcher32": False, "chunk_samples": chunk_samples},
    )
    result.close()
    elapsed = time.perf_counter() - start
    manifest["materialization_seconds"] = elapsed
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return elapsed, (root / "data.h5").stat().st_size


@dataclass(frozen=True)
class Case:
    """Serializable configuration for one fresh-process measurement."""

    pipeline: Pipeline
    access: Access
    workers: int
    repetition: int
    dataset_root: str
    materialized_root: str | None
    indices: tuple[int, ...]
    transform_spec: str | None
    model_spec: str | None
    samples: int
    batch_size: int
    warmup_steps: int
    benchmark_steps: int
    seed: int
    precision: str
    classes: int
    compression: str
    chunk_samples: int
    layout: Layout
    materialization_seconds: float
    file_size: int
    workload: Workload = "combined"
    datamodule_spec: str | None = None


def _dataset_for_case(case: Case) -> Dataset:
    if case.pipeline == "online":
        base: Dataset = OnlineStage2Dataset(Path(case.dataset_root), case.transform_spec, case.samples, case.workload)
    else:
        base = StructuredHDF5Dataset(case.materialized_root)
    return Subset(base, list(case.indices)) if case.indices else base


def _loader_for_case(case: Case) -> DataLoader:
    if case.datamodule_spec is not None and case.pipeline != "device_only":
        factory = _import_object(case.datamodule_spec)
        data_module = factory(
            pipeline=case.pipeline,
            source_root=Path(case.dataset_root),
            structured_root=Path(case.materialized_root) if case.materialized_root else None,
            batch_size=case.batch_size,
            num_workers=case.workers,
            shuffle=case.access == "shuffled",
            seed=case.seed,
        )
        data_module.setup("fit")
        loader = data_module.train_dataloader()
        if not isinstance(loader, DataLoader):
            raise TypeError("DataModule factory train_dataloader() must return a DataLoader")
        return loader
    worker_options = {"prefetch_factor": 1, "multiprocessing_context": "fork"} if case.workers else {}
    return DataLoader(
        _dataset_for_case(case),
        batch_size=case.batch_size,
        shuffle=case.access == "shuffled",
        num_workers=case.workers,
        persistent_workers=case.workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **worker_options,
    )


def _shutdown_loader(loader: DataLoader, iterator: Any) -> None:
    """Stop persistent workers before the isolated child interpreter exits."""
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
    if getattr(loader, "_iterator", None) is iterator:
        loader._iterator = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, precision: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _measure_case(case: Case) -> dict[str, Any]:
    torch.manual_seed(case.seed + case.repetition)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _loader_for_case(case)
    startup_start = time.perf_counter()
    iterator = iter(loader)
    try:
        first_batch = next(iterator)
        startup_seconds = time.perf_counter() - startup_start

        def next_batch() -> Any:
            nonlocal iterator
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(loader)
                return next(iterator)

        features, labels = _features_and_labels(first_batch)
        flat = _flatten_features(features, device)
        labels = labels.to(device, non_blocking=True)
        model = _build_model(case.model_spec, flat.shape[1], case.classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and case.precision == "fp16")

        def training_step(batch: Any) -> None:
            batch_features, batch_labels = _features_and_labels(batch)
            inputs = _flatten_features(batch_features, device)
            targets = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, case.precision):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        repeated_batch = first_batch
        step = (lambda: training_step(repeated_batch)) if case.pipeline == "device_only" else (lambda: training_step(next_batch()))
        for _ in range(case.warmup_steps):
            step()
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cpu_start = time.process_time()
        start = time.perf_counter()
        for _ in range(case.benchmark_steps):
            step()
        _sync(device)
        elapsed = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start

        for _ in range(case.warmup_steps):
            _features_and_labels(next_batch())
        start_data = time.perf_counter()
        for _ in range(case.benchmark_steps):
            _features_and_labels(next_batch())
        data_elapsed = time.perf_counter() - start_data
        measured_samples = case.benchmark_steps * case.batch_size
        result = {
            **asdict(case),
            "device": str(device),
            "startup_seconds": startup_seconds,
            "measured_steps": case.benchmark_steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / case.benchmark_steps,
            "steps_per_second": case.benchmark_steps / elapsed,
            "samples_per_second": measured_samples / elapsed,
            "dataloader_samples_per_second": measured_samples / data_elapsed,
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_utilization_percent": 100.0 * cpu_seconds / elapsed,
            "peak_host_memory_bytes": _peak_host_memory_bytes(),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "warm_filesystem_cache": True,
            "epoch_seconds": len(case.indices) / (measured_samples / elapsed) if case.indices else case.samples / (measured_samples / elapsed),
        }
    finally:
        _shutdown_loader(loader, iterator)
    result["peak_host_memory_bytes"] = max(
        result["peak_host_memory_bytes"],
        _peak_host_memory_bytes(resource.RUSAGE_CHILDREN),
    )
    return result


def _peak_host_memory_bytes(who: int = resource.RUSAGE_SELF) -> int:
    """Return peak resident memory for the benchmark process."""
    peak = resource.getrusage(who).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _case_process(case_payload: dict[str, Any], output_path: str) -> None:
    try:
        result = _measure_case(Case(**case_payload))
        Path(output_path).write_text(json.dumps({"result": result}))
    except Exception as error:
        Path(output_path).write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        raise


def _run_fresh(case: Case, scratch: Path) -> dict[str, Any]:
    output = scratch / f"case-{os.getpid()}-{time.time_ns()}.json"
    launched_at = time.perf_counter()
    process = multiprocessing.get_context("spawn").Process(target=_case_process, args=(asdict(case), str(output)))
    process.start()
    process.join()
    if not output.exists():
        raise RuntimeError(f"Benchmark child exited with code {process.exitcode} without a result")
    payload = json.loads(output.read_text())
    output.unlink()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    if process.exitcode != 0:
        raise RuntimeError(f"Benchmark child exited with code {process.exitcode}; its result is not trusted")
    payload["result"]["process_total_seconds"] = time.perf_counter() - launched_at
    return payload["result"]


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repetitions and calculate variability and matching speedups."""
    keys = ("pipeline", "access", "workers", "compression", "chunk_samples", "layout")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(tuple(result[key] for key in keys), []).append(result)
    aggregates = []
    for key, values in groups.items():
        rates = [value["samples_per_second"] for value in values]
        row = dict(zip(keys, key, strict=True))
        row.update(
            repetitions=len(values),
            median_samples_per_second=statistics.median(rates),
            stdev_samples_per_second=statistics.stdev(rates) if len(rates) > 1 else 0.0,
            median_dataloader_samples_per_second=statistics.median(value["dataloader_samples_per_second"] for value in values),
            median_startup_seconds=statistics.median(value["startup_seconds"] for value in values),
            median_epoch_seconds=statistics.median(value["epoch_seconds"] for value in values),
            median_process_cpu_utilization_percent=statistics.median(value.get("process_cpu_utilization_percent", 0.0) for value in values),
            peak_host_memory_bytes=max(value.get("peak_host_memory_bytes", 0) for value in values),
            peak_gpu_memory_bytes=max(value.get("peak_gpu_memory_bytes", 0) for value in values),
            materialization_seconds=values[0]["materialization_seconds"],
            file_size=values[0]["file_size"],
        )
        aggregates.append(row)
    baselines = {(row["access"], row["workers"]): row for row in aggregates if row["pipeline"] == "online"}
    for row in aggregates:
        baseline = baselines.get((row["access"], row["workers"]))
        row["speedup_vs_online"] = row["median_samples_per_second"] / baseline["median_samples_per_second"] if baseline else None
        saved = baseline["median_epoch_seconds"] - row["median_epoch_seconds"] if baseline and row["pipeline"] == "materialized" else 0.0
        row["break_even_epochs"] = row["materialization_seconds"] / saved if saved > 0 else None
    return aggregates


def recommend(aggregates: list[dict[str, Any]], minimum_speedup: float = 1.5) -> list[str]:
    """Produce conservative conclusions from shuffled end-to-end medians."""
    shuffled = [row for row in aggregates if row["pipeline"] == "materialized" and row["access"] == "shuffled"]
    if not shuffled:
        return ["No shuffled materialized measurements completed."]
    best_rate = max(row["median_samples_per_second"] for row in shuffled)
    candidates = [row for row in shuffled if row["median_samples_per_second"] >= best_rate * 0.95]
    candidates.sort(key=lambda row: (row["compression"] != "none", row["chunk_samples"], row["workers"]))
    best = candidates[0]
    ceiling_rows = [row for row in aggregates if row["pipeline"] == "device_only"]
    ceiling = max((row["median_samples_per_second"] for row in ceiling_rows), default=math.nan)
    stable = best["repetitions"] > 1 and best["stdev_samples_per_second"] <= best_rate * 0.05
    same_storage = [row for row in shuffled if row["compression"] == best["compression"] and row["chunk_samples"] == best["chunk_samples"] and row["workers"] == best["workers"]]
    full = next((row for row in same_storage if row["layout"] == "full"), None)
    separate = next((row for row in same_storage if row["layout"] == "separate"), None)
    if not full or not separate:
        separation = "Physical split separation was not measured."
    elif separate["median_samples_per_second"] > full["median_samples_per_second"] * 1.05:
        separation = "Physically separate splits are recommended."
    else:
        separation = "Physical split separation did not improve throughput by more than 5%."
    stability = "not established" if best["repetitions"] < 2 else "yes" if stable else "no"
    speedup = best["speedup_vs_online"]
    target = f"Throughput target ({minimum_speedup:.2f}x shuffled speedup): {'met' if speedup is not None and speedup >= minimum_speedup else 'not met'}."
    return [
        target,
        f"Recommended shuffled layout: compression={best['compression']}, chunk_samples={best['chunk_samples']}, workers={best['workers']}, layout={best['layout']}.",
        f"Shuffled end-to-end speedup versus online: {speedup:.2f}x." if speedup is not None else "No matching online baseline completed.",
        f"It reaches {100 * best_rate / ceiling:.1f}% of the device-only ceiling." if math.isfinite(ceiling) else "No device-only ceiling completed.",
        f"Additional storage: {best['file_size'] / 1024**3:.2f} GiB.",
        f"Break-even: {best['break_even_epochs']:.2f} epochs." if best["break_even_epochs"] is not None else "No materialization break-even exists.",
        f"Repetition stability (<=5% standard deviation): {stability}.",
        separation,
    ]


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact end-to-end comparison table."""
    header = "pipeline access workers compression chunk layout samples/s speedup break-even"
    print(header)
    for row in sorted(rows, key=lambda item: (item["pipeline"], item["access"], item["workers"], item["compression"], item["chunk_samples"], item["layout"])):
        speedup = "-" if row["speedup_vs_online"] is None else f"{row['speedup_vs_online']:.2f}x"
        break_even = "never" if row["break_even_epochs"] is None else f"{row['break_even_epochs']:.2f}"
        print(
            f"{row['pipeline']:12} {row['access']:10} {row['workers']:7} {row['compression']:11} {row['chunk_samples']:5} {row['layout']:8} {row['median_samples_per_second']:9.1f} {speedup:7} {break_even}"
        )


def _csv_value(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (dict, list, tuple)) else value


def _write_results(
    path: Path,
    results: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    recommendation: list[str],
    failures: list[dict[str, Any]] | None = None,
    environment: Mapping[str, Any] | None = None,
    benchmark_config: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "environment": environment or {},
                "benchmark_config": benchmark_config or {},
                "measurements": results,
                "aggregates": aggregates,
                "recommendation": recommendation,
                "failures": failures or [],
            },
            indent=2,
        )
    )
    csv_path = path.with_suffix(".csv")
    fields = sorted({key for row in results for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in results)


def _case_key(value: Case | Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the configuration identity used to resume an interrupted run."""
    fields = (
        "pipeline",
        "access",
        "workers",
        "repetition",
        "dataset_root",
        "materialized_root",
        "transform_spec",
        "model_spec",
        "samples",
        "batch_size",
        "warmup_steps",
        "benchmark_steps",
        "seed",
        "precision",
        "classes",
        "compression",
        "chunk_samples",
        "layout",
        "workload",
        "datamodule_spec",
    )
    return tuple(getattr(value, field) if isinstance(value, Case) else value[field] for field in fields)


def _checkpoint(path: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    """Atomically save completed measurements after every isolated case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps({"measurements": results, "failures": failures}, indent=2))
    temporary.replace(path)


def _parse_csv(value: str, converter: Callable[[str], Any]) -> tuple[Any, ...]:
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def _environment_metadata() -> dict[str, Any]:
    """Capture enough software and hardware context to review benchmark results."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Production hooks: --stage2-transforms module:build_stage2_transforms and --model-factory module:build_model.",
    )
    parser.add_argument("--dataset-root", type=Path, help="Existing static TorchSig dataset; omit to generate deterministic fake IQ data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--synthetic-iq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Compatibility alias used when --worker-counts is omitted")
    parser.add_argument("--worker-counts", default=None, help="Comma-separated matrix, for example 0,4,16,32")
    parser.add_argument("--chunk-sizes", default="1,4,8,32")
    parser.add_argument("--compressions", default="none,lzf")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--benchmark-steps", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=83763046)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--results", type=Path, default=Path("stage2_materialization_results.json"))
    parser.add_argument("--reuse-materialized", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the large-storage confirmation")
    parser.add_argument("--large-storage-gib", type=float, default=50.0)
    parser.add_argument("--stage2-transforms", help="Import path module:build_stage2_transforms")
    parser.add_argument("--model-factory", help="Import path module:build_model")
    parser.add_argument(
        "--datamodule-factory",
        help="Import path module:build_datamodule; exercises setup('fit') and train_dataloader() for online/materialized cases",
    )
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--split-layouts", default="full,separate", help="Comma-separated full,separate")
    parser.add_argument("--workload", choices=("combined", "iq", "spectrogram", "detection"), default="combined")
    parser.add_argument("--minimum-speedup", type=float, default=1.5, help="Predeclared shuffled throughput success target")
    return parser.parse_args()


def main() -> None:
    """Materialize the requested matrix and run isolated measurements."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = args.output_dir / "synthetic-source"
        _generate_synthetic_source(dataset_root, args.samples, args.synthetic_iq_length, args.seed, args.batch_size)
    source = OnlineStage2Dataset(dataset_root, args.stage2_transforms, args.samples, args.workload)
    if not len(source):
        raise ValueError("The source dataset is empty")
    sample_count = len(source)
    chunk_sizes = _parse_csv(args.chunk_sizes, int)
    compressions = _parse_csv(args.compressions, str)
    workers = _parse_csv(args.worker_counts, int) if args.worker_counts else (args.num_workers,)
    layouts = _parse_csv(args.split_layouts, str)
    if args.samples < 1 or args.batch_size < 1 or args.warmup_steps < 0 or args.benchmark_steps < 1 or args.repetitions < 1:
        raise ValueError("samples, batch size, benchmark steps, and repetitions must be positive; warmup steps must be non-negative")
    if args.minimum_speedup <= 0:
        raise ValueError("minimum speedup must be positive")
    if any(worker < 0 for worker in workers) or any(chunk < 1 for chunk in chunk_sizes):
        raise ValueError("worker counts must be non-negative and chunk sizes must be positive")
    if not compressions or any(compression not in {"none", "lzf"} for compression in compressions):
        raise ValueError("compressions must contain only none and lzf")
    if not layouts or any(layout not in {"full", "separate"} for layout in layouts):
        raise ValueError("split layouts must contain only full and separate")
    copies = len(chunk_sizes) * len(compressions) * len(layouts)
    estimated = estimate_storage_bytes(source[0], sample_count, copies)
    print(f"Transform pipeline: {source.transform_name}")
    print(f"Estimated uncompressed materialized storage across the matrix: {estimated / 1024**3:.2f} GiB")
    if estimated > args.large_storage_gib * 1024**3 and not args.yes and (not sys.stdin.isatty() or input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}):
        raise SystemExit("Materialization cancelled; pass --yes to override")

    splits = _split_indices(sample_count, args.seed)
    train_indices = splits["train"]
    if len(train_indices) < args.batch_size:
        raise ValueError("The training split must contain at least one full batch")
    materialized: dict[tuple[str, int, str], tuple[Path, float, int, tuple[int, ...]]] = {}
    for compression_name in compressions:
        compression = None if compression_name == "none" else compression_name
        for chunk_size in chunk_sizes:
            for layout in layouts:
                root = args.output_dir / f"{layout}-{compression_name}-chunk{chunk_size}"
                materialization_source: Dataset = source if layout == "full" else Subset(source, train_indices)
                manifest = {
                    "dataset_root": str(dataset_root.resolve()),
                    "samples": sample_count,
                    "compression": compression_name,
                    "chunk_samples": chunk_size,
                    "layout": layout,
                    "seed": args.seed,
                    "transform": source.transform_name,
                    "workload": args.workload,
                }
                elapsed, size = _materialize(materialization_source, root, manifest, args.batch_size, compression, chunk_size, args.reuse_materialized)
                indices = tuple(train_indices) if layout == "full" else tuple(range(len(train_indices)))
                materialized[(compression_name, chunk_size, layout)] = (root, elapsed, size, indices)

    rng = np.random.default_rng(args.seed)
    model_check_done = False
    for (_, _, layout), (root, _, _, stored_indices) in materialized.items():
        stored_length = len(stored_indices)
        local_checks = sorted({0, stored_length // 2, stored_length - 1, *(int(value) for value in rng.choice(stored_length, size=min(5, stored_length), replace=False))})
        reference = StructuredHDF5Dataset(root)
        try:
            for local_index in local_checks:
                source_index = stored_indices[local_index] if layout == "full" else train_indices[local_index]
                _assert_equivalent(source[source_index], reference[local_index if layout == "separate" else source_index])
            if not model_check_done:
                batch_local = list(range(min(args.batch_size, stored_length)))
                online_samples = [source[stored_indices[index] if layout == "full" else train_indices[index]] for index in batch_local]
                materialized_samples = [reference[index if layout == "separate" else stored_indices[index]] for index in batch_local]
                _assert_model_equivalent(online_samples, materialized_samples, args.model_factory, args.classes, args.seed)
                model_check_done = True
        finally:
            reference.close()
    print("Sample and model-output equivalence checks passed.")

    device_root, _, _, device_indices = next(
        (value for key, value in materialized.items() if key[2] == "full"),
        next(iter(materialized.values())),
    )

    cases: list[Case] = []
    accesses: tuple[Access, ...] = ("shuffled", "sequential")
    for repetition in range(args.repetitions):
        for worker_count in workers:
            for access in accesses:
                cases.append(
                    Case(
                        pipeline="online",
                        access=access,
                        workers=worker_count,
                        repetition=repetition,
                        dataset_root=str(dataset_root),
                        materialized_root=None,
                        indices=tuple(train_indices),
                        transform_spec=args.stage2_transforms,
                        model_spec=args.model_factory,
                        samples=sample_count,
                        batch_size=args.batch_size,
                        warmup_steps=args.warmup_steps,
                        benchmark_steps=args.benchmark_steps,
                        seed=args.seed,
                        precision=args.precision,
                        classes=args.classes,
                        compression="n/a",
                        chunk_samples=0,
                        layout="full",
                        materialization_seconds=0.0,
                        file_size=0,
                        workload=args.workload,
                        datamodule_spec=args.datamodule_factory,
                    )
                )
                for (compression_name, chunk_size, layout), (root, elapsed, size, indices) in materialized.items():
                    cases.append(
                        Case(
                            pipeline="materialized",
                            access=access,
                            workers=worker_count,
                            repetition=repetition,
                            dataset_root=str(dataset_root),
                            materialized_root=str(root),
                            indices=indices,
                            transform_spec=args.stage2_transforms,
                            model_spec=args.model_factory,
                            samples=sample_count,
                            batch_size=args.batch_size,
                            warmup_steps=args.warmup_steps,
                            benchmark_steps=args.benchmark_steps,
                            seed=args.seed,
                            precision=args.precision,
                            classes=args.classes,
                            compression=compression_name,
                            chunk_samples=chunk_size,
                            layout=layout,
                            materialization_seconds=elapsed,
                            file_size=size,
                            workload=args.workload,
                            datamodule_spec=args.datamodule_factory,
                        )
                    )
        cases.append(
            Case(
                pipeline="device_only",
                access="shuffled",
                workers=0,
                repetition=repetition,
                dataset_root=str(dataset_root),
                materialized_root=str(device_root),
                indices=device_indices,
                transform_spec=args.stage2_transforms,
                model_spec=args.model_factory,
                samples=sample_count,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
                seed=args.seed,
                precision=args.precision,
                classes=args.classes,
                compression="memory",
                chunk_samples=0,
                layout="full",
                materialization_seconds=0.0,
                file_size=0,
                workload=args.workload,
                datamodule_spec=None,
            )
        )

    rng.shuffle(cases)
    partial_path = args.results.with_suffix(".partial.json")
    results: list[dict[str, Any]] = []
    if partial_path.exists():
        try:
            saved = json.loads(partial_path.read_text()).get("measurements", [])
            valid_keys = {_case_key(case) for case in cases}
            results = [result for result in saved if _case_key(result) in valid_keys]
            print(f"Resuming {len(results)} completed cases from {partial_path}")
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
            print(f"Ignoring unreadable checkpoint {partial_path}: {error}")
    completed_keys = {_case_key(result) for result in results}
    failures: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        if _case_key(case) in completed_keys:
            continue
        print(f"[{number}/{len(cases)}] {case.pipeline} {case.access} workers={case.workers} compression={case.compression} chunk={case.chunk_samples} layout={case.layout} rep={case.repetition}")
        try:
            result = _run_fresh(case, args.output_dir)
        except Exception as error:  # noqa: BLE001 - isolate failures between benchmark cases
            failure = {**asdict(case), "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            print(f"Case failed: {failure['error']}", file=sys.stderr)
        else:
            results.append(result)
            completed_keys.add(_case_key(case))
        _checkpoint(partial_path, results, failures)
    aggregates = aggregate_results(results)
    recommendation = recommend(aggregates, args.minimum_speedup)
    _write_results(
        args.results,
        results,
        aggregates,
        recommendation,
        failures,
        environment=_environment_metadata(),
        benchmark_config={
            "workload": args.workload,
            "minimum_speedup": args.minimum_speedup,
            "repetitions": args.repetitions,
            "worker_counts": workers,
            "split_layouts": layouts,
            "transform_factory": args.stage2_transforms,
            "model_factory": args.model_factory,
            "datamodule_factory": args.datamodule_factory,
        },
    )
    _print_table(aggregates)
    for line in recommendation:
        print(line)
    print(f"JSON: {args.results}")
    print(f"CSV: {args.results.with_suffix('.csv')}")
    if failures:
        print(f"{len(failures)} cases failed; rerun the same command to retry them. Checkpoint: {partial_path}")
    else:
        partial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
