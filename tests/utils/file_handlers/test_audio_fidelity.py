"""Fidelity-policy tests shared by WAV and OGG IQ readers."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

from torchsig.utils.file_handlers import OGGReader, WAVReader

if TYPE_CHECKING:
    from pathlib import Path

READER_CASES = [(WAVReader, ".wav"), (OGGReader, ".ogg")]
FIELDS = [
    "index",
    "label",
    "modcod",
    "sample_rate",
    "file_path",
    "start_frame",
    "num_frames",
    "expected_sample_rate",
    "channel_count",
]


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> None:
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, path: str, rate: int, frames: int) -> dict[str, object]:
    return {
        "index": index,
        "label": "BPSK",
        "modcod": 0,
        "sample_rate": rate,
        "file_path": path,
        "start_frame": 0,
        "num_frames": frames,
        "expected_sample_rate": rate,
        "channel_count": 2,
    }


def _write_audio(path: Path, data: np.ndarray, rate: int = 48_000) -> None:
    sf.write(path, data, rate, format="WAV", subtype="FLOAT")


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
@pytest.mark.parametrize("channels", [1, 3])
def test_rejects_non_stereo_audio(tmp_path: Path, reader_class, suffix: str, channels: int) -> None:
    path = "data" + suffix
    _write_audio(tmp_path / path, np.zeros((4, channels), dtype=np.float32))
    _write_manifest(tmp_path, [_row(0, path, 48_000, 4)])

    with pytest.raises(ValueError, match="exactly 2 channels"):
        reader_class(tmp_path)


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
def test_accepts_mixed_declared_rates_without_dataset_rate(tmp_path: Path, reader_class, suffix: str) -> None:
    rows = []
    for index, rate in enumerate((32_000, 48_000)):
        path = f"data_{index}{suffix}"
        _write_audio(tmp_path / path, np.zeros((2, 2), dtype=np.float32), rate)
        rows.append(_row(index, path, rate, 2))
    _write_manifest(tmp_path, rows)

    assert len(reader_class(tmp_path)) == 2


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
def test_rejects_record_and_dataset_sample_rate_conflicts(tmp_path: Path, reader_class, suffix: str) -> None:
    path = "data" + suffix
    _write_audio(tmp_path / path, np.zeros((2, 2), dtype=np.float32), 48_000)
    row = _row(0, path, 44_100, 2)
    _write_manifest(tmp_path, [row])

    with pytest.raises(ValueError, match="record sample_rate"):
        reader_class(tmp_path)

    row["sample_rate"] = 48_000
    row["expected_sample_rate"] = 48_000
    _write_manifest(tmp_path, [row])
    (tmp_path / "info.json").write_text(json.dumps({"sample_rate": 44_100}), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset sample_rate"):
        reader_class(tmp_path)


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
@pytest.mark.parametrize("mode", ["rms", "peak"])
def test_joint_normalization_preserves_complex_ratios_and_records_scale(tmp_path: Path, reader_class, suffix: str, mode: str) -> None:
    path = "data" + suffix
    complex_data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    _write_audio(tmp_path / path, np.column_stack((complex_data.real, complex_data.imag)))
    _write_manifest(tmp_path, [_row(0, path, 48_000, 2)])

    default = reader_class(tmp_path).read(0)
    normalized = reader_class(tmp_path, normalization=mode, record_normalization_scale=True).read(0)

    np.testing.assert_array_equal(default.data, complex_data)
    np.testing.assert_allclose(normalized.data[1] / normalized.data[0], complex_data[1] / complex_data[0])
    target = np.sqrt(np.mean(np.abs(normalized.data) ** 2)) if mode == "rms" else np.max(np.abs(normalized.data))
    assert target == pytest.approx(1.0)
    assert normalized.metadata["normalization_scale"] > 0


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
def test_silence_normalization_is_deterministic(tmp_path: Path, reader_class, suffix: str) -> None:
    path = "data" + suffix
    _write_audio(tmp_path / path, np.zeros((3, 2), dtype=np.float32))
    _write_manifest(tmp_path, [_row(0, path, 48_000, 3)])

    signal = reader_class(tmp_path, normalization="rms", record_normalization_scale=True).read(0)

    np.testing.assert_array_equal(signal.data, np.zeros(3, dtype=np.complex64))
    assert signal.metadata["normalization_scale"] == 1.0


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
@pytest.mark.parametrize(
    ("returned", "error"),
    [
        (np.zeros((1, 2), dtype=np.float32), "returned shape"),
        (np.array([[np.nan, 0], [0, 0]], dtype=np.float32), "non-finite"),
    ],
)
def test_read_rejects_truncation_and_nonfinite_values(tmp_path: Path, monkeypatch, reader_class, suffix: str, returned: np.ndarray, error: str) -> None:
    path = "data" + suffix
    _write_audio(tmp_path / path, np.zeros((2, 2), dtype=np.float32))
    _write_manifest(tmp_path, [_row(0, path, 48_000, 2)])
    reader = reader_class(tmp_path)
    monkeypatch.setattr(
        reader._audio_handles,
        "read",
        lambda _path, _start, _frames: returned,
    )

    with pytest.raises(ValueError, match=error):
        reader.read(0)


def test_ogg_rejects_lossy_codec(tmp_path: Path) -> None:
    if "VORBIS" not in sf.available_subtypes("OGG"):
        pytest.skip("libsndfile has no OGG/Vorbis support")
    path = "lossy.ogg"
    sf.write(tmp_path / path, np.zeros((8, 2), dtype=np.float32), 48_000, format="OGG", subtype="VORBIS")
    _write_manifest(tmp_path, [_row(0, path, 48_000, 8)])

    with pytest.raises(ValueError, match="Unsupported lossy OGG codec"):
        OGGReader(tmp_path)


@pytest.mark.parametrize("normalization", ["unknown", 1])
def test_rejects_invalid_normalization(tmp_path: Path, normalization: object) -> None:
    with pytest.raises((TypeError, ValueError), match="normalization"):
        WAVReader(tmp_path, normalization=normalization)  # type: ignore[arg-type]
