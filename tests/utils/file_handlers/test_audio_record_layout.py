"""Shared manifest tests for WAV and OGG IQ readers."""

from __future__ import annotations

import csv
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from torchsig.utils.file_handlers import OGGReader, WAVReader

ReaderCase = tuple[type[WAVReader] | type[OGGReader], str]
READER_CASES: list[ReaderCase] = [(WAVReader, ".wav"), (OGGReader, ".ogg")]
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


_WORKER_READER: WAVReader | None = None


def _read_worker_index(index: int) -> tuple[int, np.ndarray]:
    """Read one index from the reader inherited by a fork worker."""
    if _WORKER_READER is None:
        raise RuntimeError("worker reader is not initialized")
    return index, _WORKER_READER.read(index).data


def _write_audio(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.column_stack((values.real, values.imag)).astype(np.float32)
    sf.write(path, stereo, 48_000, format="WAV", subtype="FLOAT")


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> None:
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, path: str, start: object, length: object) -> dict[str, object]:
    return {
        "index": index,
        "label": "BPSK",
        "modcod": 0,
        "sample_rate": 48_000,
        "file_path": path,
        "start_frame": start,
        "num_frames": length,
        "expected_sample_rate": 48_000,
        "channel_count": 2,
    }


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
def test_manifest_reads_variable_lengths_and_nested_paths(
    tmp_path: Path,
    reader_class: type[WAVReader | OGGReader],
    suffix: str,
) -> None:
    first_path = Path("nested/first" + suffix)
    second_path = Path("second" + suffix)
    first = np.arange(10, dtype=np.float32) + 1j * np.arange(20, 30, dtype=np.float32)
    second = np.arange(100, 106, dtype=np.float32) + 1j * np.arange(200, 206, dtype=np.float32)
    _write_audio(tmp_path / first_path, first)
    _write_audio(tmp_path / second_path, second)
    _write_manifest(
        tmp_path,
        [
            _row(0, first_path.as_posix(), 0, 2),
            _row(1, first_path.as_posix(), 3, 4),
            _row(2, first_path.as_posix(), 8, 2),
            _row(3, second_path.as_posix(), 1, 5),
        ],
    )

    reader = reader_class(tmp_path)

    assert len(reader) == 4
    assert [record.num_frames for record in reader.record_layout.records] == [2, 4, 2, 5]
    np.testing.assert_array_equal(reader.read(0).data, first[0:2])
    np.testing.assert_array_equal(reader.read(1).data, first[3:7])
    np.testing.assert_array_equal(reader.read(2).data, first[8:10])
    np.testing.assert_array_equal(reader.read(3).data, second[1:6])


def test_wav_and_ogg_build_equivalent_manifests(tmp_path: Path) -> None:
    layouts = []
    for reader_class, suffix in READER_CASES:
        root = tmp_path / suffix.removeprefix(".")
        root.mkdir()
        audio_path = Path("records/data" + suffix)
        _write_audio(root / audio_path, np.arange(6) + 1j * np.arange(6))
        _write_manifest(root, [_row(0, audio_path.as_posix(), 0, 2), _row(1, audio_path.as_posix(), 3, 3)])
        reader = reader_class(root)
        layouts.append([(record.start_frame, record.num_frames) for record in reader.record_layout.records])

    assert layouts[0] == layouts[1] == [(0, 2), (3, 3)]


def test_shuffled_multiworker_reads_reopen_process_local_handles(tmp_path: Path) -> None:
    """Forked DataLoader workers return correct records after parent access."""
    values = np.arange(24, dtype=np.float32) + 1j * np.arange(100, 124, dtype=np.float32)
    _write_audio(tmp_path / "records.wav", values)
    rows = [_row(index, "records.wav", index * 3, 3) for index in range(8)]
    _write_manifest(tmp_path, rows)
    reader = WAVReader(tmp_path, audio_handle_cache_size=1)
    np.testing.assert_array_equal(reader.read(0).data, values[:3])
    reader.teardown()

    global _WORKER_READER  # noqa: PLW0603 - fork workers intentionally inherit this reader
    _WORKER_READER = reader
    indices = list(np.random.default_rng(0).permutation(len(reader)))
    with multiprocessing.get_context("fork").Pool(2) as pool:
        observed = dict(pool.map(_read_worker_index, indices))

    assert set(observed) == set(range(8))
    for index, data in observed.items():
        np.testing.assert_array_equal(data, values[index * 3 : index * 3 + 3])
    reader.teardown()
    _WORKER_READER = None


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([_row(0, "../outside.wav", 0, 1)], "escapes dataset root"),
        ([_row(0, "/absolute.wav", 0, 1)], "path must be relative"),
        ([_row(0, "data.wav", -1, 1)], "negative start_frame"),
        ([_row(0, "data.wav", 0, -1)], "negative num_frames"),
        ([_row(0, "data.wav", 4, 2)], "exceeds"),
        ([_row(0, "data.wav", 0, 3), _row(1, "data.wav", 2, 2)], "overlap"),
    ],
)
def test_manifest_rejects_invalid_segments(
    tmp_path: Path,
    reader_class: type[WAVReader | OGGReader],
    suffix: str,
    rows: list[dict[str, object]],
    error: str,
) -> None:
    actual_name = "data" + suffix
    for row in rows:
        if row["file_path"] == "data.wav":
            row["file_path"] = actual_name
    _write_audio(tmp_path / actual_name, np.arange(5) + 1j * np.arange(5))
    _write_manifest(tmp_path, rows)

    with pytest.raises(ValueError, match=error):
        reader_class(tmp_path)


@pytest.mark.parametrize(("reader_class", "suffix"), READER_CASES)
def test_manifest_requires_every_record_to_resolve(
    tmp_path: Path,
    reader_class: type[WAVReader | OGGReader],
    suffix: str,
) -> None:
    _write_audio(tmp_path / ("data" + suffix), np.arange(2) + 1j * np.arange(2))
    _write_manifest(tmp_path, [_row(0, "missing" + suffix, 0, 1)])

    with pytest.raises(ValueError, match="does not resolve"):
        reader_class(tmp_path)
