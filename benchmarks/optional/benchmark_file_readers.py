"""Pytest benchmarks for TorchSig HDF5, WAV, SigMF, and OGG readers.

The session fixture creates equivalent temporary datasets before timing. Run:

    pytest benchmarks/optional/benchmark_file_readers.py --benchmark-only

Override defaults with TORCHSIG_BENCHMARK_RECORDS,
TORCHSIG_BENCHMARK_IQ_SAMPLES, TORCHSIG_BENCHMARK_ELEMENTS_PER_FILE,
TORCHSIG_BENCHMARK_READS, and TORCHSIG_BENCHMARK_AUDIO_CACHE_SIZES. Results
are warm-cache measurements. Pytest-benchmark reports records per second; the
extra fields report requested bytes and process peak memory. Compare runs with
``TORCHSIG_BENCHMARK_IQ_SAMPLES`` set to representative small and large record
sizes. A greater than 10% throughput regression against the cache-disabled
baseline should be investigated.
"""

from __future__ import annotations

import csv
import json
import os
import resource
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

from torchsig.utils.file_handlers.hdf5 import HDF5Reader
from torchsig.utils.file_handlers.ogg import OGGReader
from torchsig.utils.file_handlers.sigmf import SigMFReader
from torchsig.utils.file_handlers.wav import WAVReader

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

READER_TYPES = {
    "hdf5": HDF5Reader,
    "wav": WAVReader,
    "sigmf": SigMFReader,
    "ogg": OGGReader,
}


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _cache_sizes() -> tuple[int, ...]:
    """Return configured nonnegative cache sizes for audio benchmarks."""
    values = tuple(int(value) for value in os.environ.get("TORCHSIG_BENCHMARK_AUDIO_CACHE_SIZES", "0,1,8").split(","))
    if not values or any(value < 0 for value in values):
        raise ValueError("TORCHSIG_BENCHMARK_AUDIO_CACHE_SIZES must contain nonnegative integers")
    return values


def _write_sidecars(
    root: Path,
    records: int,
    num_files: int,
    elements_per_file: int,
    num_iq_samples: int,
    sample_rate: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Headerless is intentional: WAVReader and SigMFReader currently count
    # every nonempty CSV line themselves, including a header if present.
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for idx in range(records):
            writer.writerow((idx, "BPSK", 0, sample_rate))

    info = {
        "sidecar_schema_version": "1.0",
        "size": records,
        "class_list": ["BPSK"],
        "sample_rate": sample_rate,
        "num_files": num_files,
        "elements_per_file": elements_per_file,
        "num_iq_samples": num_iq_samples,
        "datatype": "cf32_le",
    }
    (root / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _write_hdf5(root: Path, data: np.ndarray, sample_rate: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "data.h5", "w") as h5_file:
        data_group = h5_file.create_group("data")
        metadata_group = h5_file.create_group("metadata")
        index_group = h5_file.create_group("index")
        h5_file.create_group("component_signals")
        for idx, record in enumerate(data):
            key = str(idx)
            data_group.create_dataset(key, data=record)
            index_group.create_dataset(key, data=np.bytes_(key))
            metadata = metadata_group.create_group(key)
            metadata.create_dataset("index", data=idx)
            metadata.create_dataset("label", data=np.bytes_("BPSK"))
            metadata.create_dataset("modcod", data=0)
            metadata.create_dataset("sample_rate", data=sample_rate)


def _write_wav(root: Path, files: Sequence[np.ndarray], sample_rate: int) -> None:
    for file_idx, records in enumerate(files):
        values = records.reshape(-1)
        stereo = np.column_stack((values.real, values.imag)).astype(np.float32)
        sf.write(root / f"data_{file_idx:04d}.wav", stereo, sample_rate, format="WAV", subtype="FLOAT")


def _write_sigmf(root: Path, files: Sequence[np.ndarray], sample_rate: int) -> None:
    for file_idx, records in enumerate(files):
        stem = root / f"data_{file_idx:04d}"
        records.astype("<c8", copy=False).tofile(stem.with_suffix(".sigmf-data"))
        metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": sample_rate,
                "core:version": "1.0.0",
            },
            "captures": [{"core:sample_start": 0}],
            "annotations": [],
        }
        stem.with_suffix(".sigmf-meta").write_text(json.dumps(metadata), encoding="utf-8")


def _write_ogg(root: Path, files: Sequence[np.ndarray], sample_rate: int) -> None:
    if "OGG" not in sf.available_formats():
        pytest.skip("libsndfile does not support OGG in this environment")
    for file_idx, records in enumerate(files):
        values = records.reshape(-1)
        stereo = np.column_stack((values.real, values.imag)).astype(np.float32)
        # libsndfile's portable OGG writer is Vorbis. This benchmark measures
        # decode throughput rather than round-trip sample fidelity.
        sf.write(root / f"data_{file_idx:04d}.ogg", stereo, sample_rate, format="OGG", subtype="VORBIS")


@pytest.fixture(scope="session")
def file_reader_datasets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create equivalent datasets that pytest cleans up after the session."""
    records = _positive_env_int("TORCHSIG_BENCHMARK_RECORDS", 128)
    num_iq_samples = _positive_env_int("TORCHSIG_BENCHMARK_IQ_SAMPLES", 16384)
    elements_per_file = _positive_env_int("TORCHSIG_BENCHMARK_ELEMENTS_PER_FILE", 16)
    # Vorbis accepts standard audio rates; this value is metadata for the IQ
    # records and does not affect the number of benchmarked samples.
    sample_rate = _positive_env_int("TORCHSIG_BENCHMARK_SAMPLE_RATE", 48_000)
    if records % elements_per_file:
        raise ValueError("TORCHSIG_BENCHMARK_RECORDS must be divisible by TORCHSIG_BENCHMARK_ELEMENTS_PER_FILE")

    root = tmp_path_factory.mktemp("file_reader_benchmark")
    roots = {name: root / name for name in READER_TYPES}
    num_files = records // elements_per_file
    rng = np.random.default_rng(0)
    data = (rng.uniform(-0.8, 0.8, size=(records, num_iq_samples)) + 1j * rng.uniform(-0.8, 0.8, size=(records, num_iq_samples))).astype(np.complex64)
    files = tuple(np.split(data, num_files))

    for name in ("wav", "sigmf", "ogg"):
        _write_sidecars(roots[name], records, num_files, elements_per_file, num_iq_samples, sample_rate)
    _write_hdf5(roots["hdf5"], data, sample_rate)
    _write_wav(roots["wav"], files, sample_rate)
    _write_sigmf(roots["sigmf"], files, sample_rate)
    _write_ogg(roots["ogg"], files, sample_rate)
    return roots


def _close(reader: Any) -> None:
    for method_name in ("teardown", "close"):
        method = getattr(reader, method_name, None)
        if callable(method):
            method()
            return


def _construct_and_close(reader_type: type, root: Path) -> int:
    reader = reader_type(root)
    try:
        return len(reader)
    finally:
        _close(reader)


def _read_records(reader: Any, indices: Sequence[int]) -> tuple[int, float]:
    payload_bytes = 0
    checksum = 0.0
    for idx in indices:
        signal = reader.read(idx)
        payload_bytes += signal.data.nbytes
        if signal.data.size:
            checksum += float(signal.data[0].real)
    return payload_bytes, checksum


def _read_audio_records(reader: Any, indices: Sequence[int]) -> tuple[int, float]:
    """Read records and report logical bytes requested from audio storage."""
    checksum = 0.0
    requested_bytes = 0
    for idx in indices:
        record = reader.record_layout[idx]
        signal = reader.read(idx)
        requested_bytes += record.num_frames * 2 * np.dtype(np.float32).itemsize
        if signal.data.size:
            checksum += float(signal.data[0].real)
    return requested_bytes, checksum


class _AudioReadDataset(Dataset):
    """Adapt a file reader and fixed index order for DataLoader benchmarks."""

    def __init__(self, reader: Any, indices: Sequence[int]) -> None:
        self.reader = reader
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.reader.read(self.indices[index]).data


def _consume_loader(loader: DataLoader) -> float:
    """Consume one DataLoader epoch and return a small anti-optimization checksum."""
    checksum = 0.0
    for batch in loader:
        if batch.numel():
            checksum += float(batch[0, 0].real)
    return checksum


def _read_audio_records_whole(reader: Any, indices: Sequence[int]) -> tuple[int, float]:
    """Reproduce the pre-MR3 whole-file implementation for comparison."""
    storage_bytes = 0
    checksum = 0.0
    for idx in indices:
        record = reader.record_layout[idx]
        pcm, _ = sf.read(record.path, dtype="float32", always_2d=True)
        storage_bytes += np.asarray(pcm).nbytes
        stereo = np.asarray(pcm)[record.start_frame : record.start_frame + record.num_frames]
        if stereo.size:
            checksum += float(stereo[0, 0])
    return storage_bytes, checksum


@pytest.mark.parametrize("reader_name", READER_TYPES)
def test_reader_initialization(
    benchmark: Callable,
    file_reader_datasets: dict[str, Path],
    reader_name: str,
) -> None:
    """Measure construction, metadata parsing, length lookup, and cleanup."""
    reader_type = READER_TYPES[reader_name]
    size = benchmark(_construct_and_close, reader_type, file_reader_datasets[reader_name])
    assert size > 0


@pytest.mark.parametrize("access_pattern", ["sequential", "random"])
@pytest.mark.parametrize("reader_name", READER_TYPES)
def test_reader_throughput(
    benchmark: Callable,
    file_reader_datasets: dict[str, Path],
    reader_name: str,
    access_pattern: str,
) -> None:
    """Measure warm-cache element-read throughput."""
    reader = READER_TYPES[reader_name](file_reader_datasets[reader_name])
    try:
        count = min(_positive_env_int("TORCHSIG_BENCHMARK_READS", 128), len(reader))
        indices = tuple(range(count)) if access_pattern == "sequential" else tuple(int(idx) for idx in np.random.default_rng(1).integers(0, len(reader), size=count))

        _read_records(reader, indices)  # untimed warm-up
        payload_bytes, checksum = benchmark(_read_records, reader, indices)
        benchmark.extra_info["records_per_iteration"] = count
        benchmark.extra_info["payload_mib_per_iteration"] = payload_bytes / (1024**2)
        assert np.isfinite(checksum)
    finally:
        _close(reader)


@pytest.mark.parametrize("cache_size", _cache_sizes())
@pytest.mark.parametrize("access_pattern", ["sequential", "shuffled"])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("reader_name", ["wav", "ogg"])
def test_audio_partial_read_cache_throughput(
    benchmark: Callable,
    file_reader_datasets: dict[str, Path],
    reader_name: str,
    access_pattern: str,
    workers: int,
    cache_size: int,
) -> None:
    """Compare bounded partial reads across access patterns and cache sizes."""
    reader = READER_TYPES[reader_name](
        file_reader_datasets[reader_name],
        audio_handle_cache_size=cache_size,
    )
    try:
        count = min(_positive_env_int("TORCHSIG_BENCHMARK_READS", 128), len(reader))
        indices = list(range(count))
        if access_pattern == "shuffled":
            np.random.default_rng(1).shuffle(indices)
        requested_bytes = sum(reader.record_layout[idx].num_frames * 2 * np.dtype(np.float32).itemsize for idx in indices)
        loader_kwargs = (
            {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
            }
            if workers
            else {}
        )
        loader = DataLoader(_AudioReadDataset(reader, indices), batch_size=1, num_workers=workers, **loader_kwargs)
        checksum = benchmark(_consume_loader, loader)
        benchmark.extra_info["records_per_iteration"] = count
        benchmark.extra_info["requested_mib_per_iteration"] = requested_bytes / (1024**2)
        usage = resource.getrusage(resource.RUSAGE_SELF if workers == 0 else resource.RUSAGE_CHILDREN)
        benchmark.extra_info["peak_worker_mib"] = usage.ru_maxrss / 1024
        benchmark.extra_info["cache_size"] = cache_size
        benchmark.extra_info["workers"] = workers
        assert np.isfinite(checksum)
    finally:
        _close(reader)


@pytest.mark.parametrize("access_pattern", ["sequential", "shuffled"])
@pytest.mark.parametrize("reader_name", ["wav", "ogg"])
def test_audio_whole_file_baseline(
    benchmark: Callable,
    file_reader_datasets: dict[str, Path],
    reader_name: str,
    access_pattern: str,
) -> None:
    """Measure the pre-MR3 whole-file read path as a comparison baseline."""
    reader = READER_TYPES[reader_name](file_reader_datasets[reader_name], audio_handle_cache_size=0)
    try:
        count = min(_positive_env_int("TORCHSIG_BENCHMARK_READS", 128), len(reader))
        indices = list(range(count))
        if access_pattern == "shuffled":
            np.random.default_rng(1).shuffle(indices)
        storage_bytes, checksum = benchmark(_read_audio_records_whole, reader, indices)
        benchmark.extra_info["records_per_iteration"] = count
        benchmark.extra_info["storage_mib_per_iteration"] = storage_bytes / (1024**2)
        assert np.isfinite(checksum)
    finally:
        _close(reader)
