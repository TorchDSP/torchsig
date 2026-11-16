"""Tests for the unified raw HDF5 reader benchmark."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "optional" / "benchmark_hdf5_readers_unified.py"
_SPEC = importlib.util.spec_from_file_location("unified_hdf5_benchmark", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
benchmark_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark_module
_SPEC.loader.exec_module(benchmark_module)


@pytest.mark.parametrize("workload", ["iq", "spectrogram"])
def test_workloads_are_deterministic_and_typed(workload: str) -> None:
    first = benchmark_module.generate_workload(workload, 4, 16, 8, 5)
    second = benchmark_module.generate_workload(workload, 4, 16, 8, 5)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == (np.complex64 if workload == "iq" else np.float32)
    assert first.shape == ((4, 16) if workload == "iq" else (4, 8, 8))


@pytest.mark.parametrize("format_name", benchmark_module.FORMATS)
def test_equivalent_datasets_and_read_operations(tmp_path: Path, format_name: str) -> None:
    data = benchmark_module.generate_workload("iq", 8, 16, 8, 3)
    root = tmp_path / format_name
    chunk_samples = 2 if format_name in {"homogeneous", "structured"} else None
    size = benchmark_module.write_dataset(format_name, root, data, "none", chunk_samples, 3)
    configuration = benchmark_module.DatasetConfiguration(format_name, "iq", "none", chunk_samples, str(root), size)
    benchmark_module.validate_dataset(configuration, data)
    reader = benchmark_module.READER_CLASSES[format_name](root)
    try:
        for operation in benchmark_module.OPERATIONS:
            indices = benchmark_module._indices(operation, len(data), 6, 11)
            checksum = benchmark_module.read_operation(reader, format_name, operation, indices, 2)
            assert np.isfinite(checksum)
    finally:
        benchmark_module._close(reader)


def test_invalid_matrix_values_fail_clearly() -> None:
    with pytest.raises(ValueError, match="formats must contain only"):
        benchmark_module._csv_values("structured,unknown", set(benchmark_module.FORMATS), "formats")


def test_cli_writes_json_csv_and_capabilities(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--output-dir",
            str(tmp_path / "output"),
            "--results",
            str(results),
            "--samples",
            "6",
            "--reads",
            "4",
            "--iq-length",
            "16",
            "--batch-size",
            "2",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--workloads",
            "iq",
            "--compressions",
            "none",
            "--chunk-sizes",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(results.read_text())
    assert len(payload["measurements"]) == 16
    assert payload["format_controls"]["standard"]["chunk_samples"] is False
    assert payload["format_controls"]["structured"]["chunk_samples"] is True
    assert {row["format"] for row in payload["measurements"]} == set(benchmark_module.FORMATS)
    shuffled = [row for row in payload["measurements"] if row["operation"] == "shuffled_batch"]
    assert all(row["native_shuffled_batch"] == (row["format"] == "structured") for row in shuffled)
    assert results.with_suffix(".csv").exists()
