import json
from pathlib import Path

import numpy as np
import pytest

from torchsig.utils.file_handlers.sigmf import SigMFReader


def _write_sigmf_pair(
    root: Path,
    stem: str,
    data: np.ndarray,
    datatype: str = "cf32_le",
) -> None:
    data.astype(np.complex64).tofile(root / f"{stem}.sigmf-data")

    meta = {
        "global": {
            "core:datatype": datatype,
            "core:sample_rate": 1_000_000,
            "core:version": "1.0.0",
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }

    with open(root / f"{stem}.sigmf-meta", "w") as f:
        json.dump(meta, f)


def _write_metadata_csv(root: Path, n_rows: int) -> None:
    labels = ["BPSK", "QPSK", "Noise"]

    with open(root / "metadata.csv", "w") as f:
        for idx in range(n_rows):
            label = labels[idx % len(labels)]
            modcod = idx  # must be an integer
            f.write(f"{idx},{label},{modcod},1000000\n")


def test_sigmf_reader_reads_single_file(tmp_path):
    num_iq_samples = 4
    elements_per_file = 2

    data = np.array(
        [
            1 + 1j,
            2 + 2j,
            3 + 3j,
            4 + 4j,
            5 + 5j,
            6 + 6j,
            7 + 7j,
            8 + 8j,
        ],
        dtype=np.complex64,
    )

    _write_sigmf_pair(tmp_path, "recording_000", data)
    _write_metadata_csv(tmp_path, 2)

    reader = SigMFReader(tmp_path)
    reader.num_iq_samples = num_iq_samples
    reader.elements_per_file = elements_per_file

    signal = reader.read(1)

    assert signal.data.dtype == np.complex64
    np.testing.assert_array_equal(signal.data, data[4:8])
    assert signal.component_signals == []


def test_sigmf_reader_infers_layout_from_csv_and_file(tmp_path):
    data = np.arange(12, dtype=np.float32).astype(np.complex64)

    _write_sigmf_pair(tmp_path, "recording_000", data)
    _write_metadata_csv(tmp_path, 3)

    reader = SigMFReader(tmp_path)

    assert reader.elements_per_file == 3
    assert reader.num_iq_samples == 4
    assert len(reader) == 3


def test_sigmf_reader_reads_across_multiple_files(tmp_path):
    data_0 = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64)
    data_1 = np.array([5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j], dtype=np.complex64)

    _write_sigmf_pair(tmp_path, "recording_000", data_0)
    _write_sigmf_pair(tmp_path, "recording_001", data_1)
    _write_metadata_csv(tmp_path, 4)

    reader = SigMFReader(tmp_path)
    reader.num_iq_samples = 2
    reader.elements_per_file = 2

    np.testing.assert_array_equal(reader.read(0).data, data_0[:2])
    np.testing.assert_array_equal(reader.read(1).data, data_0[2:])
    np.testing.assert_array_equal(reader.read(2).data, data_1[:2])
    np.testing.assert_array_equal(reader.read(3).data, data_1[2:])


def test_sigmf_reader_raises_when_no_data_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .sigmf-data files found"):
        SigMFReader(tmp_path)


def test_sigmf_reader_raises_when_meta_file_missing(tmp_path):
    data = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    data.tofile(tmp_path / "recording_000.sigmf-data")
    _write_metadata_csv(tmp_path, 1)

    with pytest.raises(FileNotFoundError, match="Missing SigMF metadata file"):
        SigMFReader(tmp_path)


def test_sigmf_reader_raises_for_invalid_index(tmp_path):
    data = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)

    _write_sigmf_pair(tmp_path, "recording_000", data)
    _write_metadata_csv(tmp_path, 1)

    reader = SigMFReader(tmp_path)

    with pytest.raises(IndexError):
        reader.read(-1)

    with pytest.raises(IndexError):
        reader.read(1)


def test_sigmf_reader_raises_for_unsupported_datatype(tmp_path):
    data = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    data.tofile(tmp_path / "recording_000.sigmf-data")

    meta = {
        "global": {
            "core:datatype": "unsupported_dtype",
        }
    }

    with open(tmp_path / "recording_000.sigmf-meta", "w") as f:
        json.dump(meta, f)

    _write_metadata_csv(tmp_path, 1)

    with pytest.raises(ValueError, match="Unsupported SigMF datatype"):
        SigMFReader(tmp_path)


def test_sigmf_reader_converts_ci16_to_complex64(tmp_path):
    raw = np.array(
        [(1, 2), (3, 4), (5, 6), (7, 8)],
        dtype=np.dtype([("i", "<i2"), ("q", "<i2")]),
    )

    raw.tofile(tmp_path / "recording_000.sigmf-data")

    meta = {
        "global": {
            "core:datatype": "ci16_le",
            "core:sample_rate": 1_000_000,
        }
    }

    with open(tmp_path / "recording_000.sigmf-meta", "w") as f:
        json.dump(meta, f)

    _write_metadata_csv(tmp_path, 2)

    reader = SigMFReader(tmp_path)

    assert reader.num_iq_samples == 2
    assert reader.elements_per_file == 2

    signal = reader.read(1)

    expected = np.array([5 + 6j, 7 + 8j], dtype=np.complex64)
    assert signal.data.dtype == np.complex64
    np.testing.assert_array_equal(signal.data, expected)
