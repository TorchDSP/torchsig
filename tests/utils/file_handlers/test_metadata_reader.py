from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from torchsig.utils.file_handlers import MetadataIndexError, MetadataReader

if TYPE_CHECKING:
    from pathlib import Path

# ----------------------------------------------------------------------
# Malformed ``class_list`` values that should trigger the fallback.
# Each entry is ``(bad_value, short description)``.
# ----------------------------------------------------------------------
malformed_cases = [
    (["BPSK", 1, "Noise"], "contains a non-string element"),
    ("BPSK,QPSK,Noise", "is a plain string instead of a list"),
    (123, "is a number"),
    (None, "is None"),
    ("{'BPSK', 'QPSK', 'Noise'}", "looks like a set but is a string"),
    (["BPSK", None, "Noise"], "contains None"),
]


# ----------------------------------------------------------------------
# Helper: write a tiny ``metadata.csv`` (no header -- the reader supplies its own).
# ----------------------------------------------------------------------
def write_csv(root: Path, rows: list[tuple[int, str, int, float]]) -> None:
    """Write a ``metadata.csv`` file under *root*.

    The CSV is written **without a header** because :class:`MetadataReader`
    supplies its own field names.  Each ``row`` must be a 4-tuple matching
    the declared order: ``index, label, modcod, sample_rate``.
    """
    csv_path = root / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


# ----------------------------------------------------------------------
# Helper to build a well-formed reference row.
# ----------------------------------------------------------------------
def good_row():
    """Return a correctly-shaped tuple that follows the CSV schema."""
    return (0, "BPSK", 0, 192_000.0)  # index, label, modcod, sample_rate


def write_info_json(root: Path, payload: dict) -> None:
    """Write an ``info.json`` file under *root*.

    ``payload`` is serialised with :func:`json.dump` using UTF-8 encoding.
    """
    (root / "info.json").write_text(json.dumps(payload), encoding="utf-8")


# ----------------------------------------------------------------------
# Fixture: a minimal dataset directory with three rows and an optional JSON.
# ----------------------------------------------------------------------
@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Return a temporary directory that already contains a ``metadata.csv`` with
    three rows:

    * two rows have known labels (``BPSK`` and ``QPSK``);
    * the third row contains an unknown label (``UNKNOWN``).

    No ``info.json`` is written - individual tests decide whether they need it.
    """
    rows = [
        (0, "BPSK", 0, 192_000.0),  # known label, index 0
        (1, "QPSK", 1, 192_000.0),  # known label, index 1
        (2, "UNKNOWN", 2, 250_000.0),  # unknown label, index 2
    ]
    write_csv(tmp_path, rows)
    return tmp_path


def test_load_row_basic(dataset_dir: Path):
    """Three well-formed rows must be parsed into the expected dictionary."""
    class_list = ["BPSK", "QPSK", "Noise"]
    reader = MetadataReader(dataset_dir)

    # ---- row 0 ----------------------------------------------------
    meta0 = reader.load_row(0, class_list)
    assert meta0["index"] == 0
    assert meta0["label"] == "BPSK"
    assert meta0["modcod"] == 0
    assert pytest.approx(meta0["sample_rate"], rel=1e-9) == 192_000.0
    assert meta0["class_name"] == "bpsk"
    assert meta0["class_index"] == 0  # BPSK is first in the list
    assert meta0["num_signals_max"] == 1

    # ---- row 1 ----------------------------------------------------
    meta1 = reader.load_row(1, class_list)
    assert meta1["label"] == "QPSK"
    assert meta1["class_name"] == "qpsk"
    assert meta1["class_index"] == 1  # QPSK is second in the list

    # ---- row 2 (unknown label) ---------------------------------
    meta2 = reader.load_row(2, class_list)
    assert meta2["label"] == "UNKNOWN"
    assert meta2["class_name"] == "unknown"
    # Unknown label → class_index = -1
    assert meta2["class_index"] == -1


def test_load_row_uses_instance_class_list(dataset_dir: Path):
    """``MetadataReader`` reads the default class list from ``info.json`` (or the
    hard-coded fallback).  When the caller does **not** provide ``class_list``,
    the method must fall back to ``self.class_list``.
    """
    reader = MetadataReader(dataset_dir)
    # The internal fallback is ["BPSK", "QPSK", "Noise"].
    meta = reader.load_row(0)  # no explicit class_list argument
    assert meta["class_index"] == 0  # BPSK → index 0 in the fallback list


def test_root_accepts_str_and_path(dataset_dir: Path):
    class_list = ["BPSK", "QPSK", "Noise"]
    # As a Path object.
    reader_path = MetadataReader(dataset_dir)
    meta_path = reader_path.load_row(0, class_list)
    # As a plain string.
    reader_str = MetadataReader(str(dataset_dir))
    meta_str = reader_str.load_row(0, class_list)
    assert meta_path == meta_str


def test_load_row_out_of_range(dataset_dir: Path):
    reader = MetadataReader(dataset_dir)
    with pytest.raises(
        MetadataIndexError,
        match=r"Metadata idx 10 is out of bounds \(file has fewer rows\)",
    ):
        reader.load_row(10, ["BPSK", "QPSK", "Noise"])


@pytest.mark.parametrize(
    ("bad_row", "expected_msg"),
    [
        # non-numeric index
        (["not-an-int", "BPSK", 0, 192_000.0], "Cannot convert 'index'='not-an-int' to int"),
        # non-numeric modcod
        ([0, "BPSK", "bad-modcod", 192_000.0], "Cannot convert 'modcod'='bad-modcod' to int"),
        # non-numeric sample_rate
        ([0, "BPSK", 0, "bad-rate"], "Cannot convert 'sample_rate'='bad-rate' to float"),
    ],
)
def test_load_row_type_conversion_errors(tmp_path: Path, bad_row, expected_msg):
    """Each malformed field should raise a clear ``ValueError``."""
    # Write a CSV containing a single malformed row.
    csv_path = tmp_path / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(bad_row)

    reader = MetadataReader(tmp_path)
    with pytest.raises(ValueError, match=expected_msg):
        reader.load_row(0)


def test_load_json_success(tmp_path: Path):
    payload = {
        "size": 123,
        "class_list": ["BPSK", "QPSK", "Noise", "CUSTOM"],
        "extra": "data",
    }
    (tmp_path / "info.json").write_text(json.dumps(payload), encoding="utf-8")
    # ``load_json`` is a pure helper. It should simply return the dict.
    reader = MetadataReader(tmp_path)  # constructor also calls it
    assert reader.dataset_metadata == payload
    assert reader.dataset_size == 123
    assert reader.class_list == payload["class_list"]


def test_load_json_missing_is_swallowed(tmp_path: Path):
    """If ``info.json`` does not exist, the constructor must not raise.
    ``self.metadata`` should be an empty dict and the default attribute values
    should be used.
    """
    # Ensure there is no info.json.
    assert not (tmp_path / "info.json").exists()
    reader = MetadataReader(tmp_path)
    assert reader.dataset_metadata == {}
    assert reader.dataset_size == 0
    assert reader.class_list == ["BPSK", "QPSK", "Noise"]


def test_load_json_malformed_is_swallowed(tmp_path: Path):
    """A corrupted JSON file should not crash the constructor; defaults are used."""
    (tmp_path / "info.json").write_text("{ not valid JSON !!!", encoding="utf-8")
    reader = MetadataReader(tmp_path)
    assert reader.dataset_metadata == {}  # JSON parsing failed
    assert reader.dataset_size == 0
    assert reader.class_list == ["BPSK", "QPSK", "Noise"]


def test_repr_contains_root_and_size(dataset_dir: Path):
    """The representation should mention the class name, root path and advertised size."""
    reader = MetadataReader(dataset_dir)
    rep = repr(reader)
    assert "MetadataReader" in rep
    assert str(dataset_dir) in rep
    # Size comes from the (optional) JSON; default value is 0.
    assert "size=0" in rep


@pytest.mark.parametrize(
    ("size_value", "expected"),
    [
        (123, 123),  # already an int → stays the same
        ("456", 456),  # numeric string → converted to int
        (78.9, 78),  # float → int() truncates toward zero
        (None, 0),  # None → TypeError → fallback 0
        ("not-a-number", 0),  # non-numeric string → ValueError → fallback 0
        ([], 0),  # list → TypeError → fallback 0
    ],
    ids=[
        "int",
        "numeric-str",
        "float-trunc",
        "none",
        "bad-str",
        "list-type",
    ],
)
def test_dataset_size_parsing(tmp_path: Path, size_value, expected):
    """``MetadataReader._populate_from_json`` must coerce the ``size`` field to an
    ``int`` when possible; otherwise it must fall back to ``0``.
    """
    write_info_json(tmp_path, {"size": size_value})

    reader = MetadataReader(tmp_path)

    assert reader.dataset_size == expected


# ----------------------------------------------------------------------
# Build a list of IDs from the description strings for the malformed class_list tests.
# ----------------------------------------------------------------------
case_ids = [desc for _, desc in malformed_cases]


@pytest.mark.parametrize(
    ("bad_class_list", "description"),
    malformed_cases,
    ids=case_ids,  # <-- explicit list of IDs (not a callable)
)
def test_class_list_fallback(tmp_path: Path, bad_class_list, description):
    """When ``info.json`` contains a ``class_list`` value that is *not* a list of
    strings, ``MetadataReader`` must fall back to the default list
    ``["BPSK", "QPSK", "Noise"]``.
    """
    write_info_json(tmp_path, {"class_list": bad_class_list})

    reader = MetadataReader(tmp_path)

    assert reader.class_list == ["BPSK", "QPSK", "Noise"], f"Failed for case: {description}. Expected fallback default, got {reader.class_list!r}"


# ----------------------------------------------------------------------
# Test scenarios for missing required columns.
# Each entry is:
#   description, rows_to_write, expected_missing_set
# ----------------------------------------------------------------------
missing_cases = [
    (
        "missing sample_rate column",
        [(0, "BPSK", 0)],  # index, label, modcod
        {"sample_rate"},
    ),
    (
        "missing sample_rate column (row has index, label, sample_rate)",
        [(0, "BPSK", 192_000.0)],  # index, label, sample_rate
        {"sample_rate"},
    ),
    (
        "empty row",
        [()],  # zero-length tuple
        {"index", "label", "modcod", "sample_rate"},
    ),
    (
        "empty label value (handled later)",
        [(0, "", 0, 192_000.0)],  # label is an empty string
        set(),  # no actual missing keys
    ),
]


@pytest.mark.parametrize(
    ("description", "rows", "expected_missing"),
    [
        ("missing sample_rate column", [(0, "BPSK", 0)], {"sample_rate"}),
        ("missing sample_rate column (different order)", [(0, "BPSK", 192_000.0)], {"sample_rate"}),
        ("empty values but all keys present", [("", "", "", "")], set()),
    ],
    ids=lambda d: d,  # use the description string as the pytest ID
)
def test_load_row_missing_columns(tmp_path: Path, description, rows, expected_missing):
    """Verify that the “missing-column” guard raises a ``ValueError`` when a row
    lacks one or more required fields, and that it reports the exact column
    names.
    """
    write_csv(tmp_path, rows)
    reader = MetadataReader(tmp_path)

    if expected_missing:
        # Expect a missing-column error.
        with pytest.raises(ValueError, match=r"is missing required columns") as excinfo:
            reader.load_row(0)
        msg = str(excinfo.value)
        for col in expected_missing:
            assert col in msg, f"Missing column {col!r} not mentioned in error: {msg}"
        assert "Row 0 of" in msg
        assert str(tmp_path / "metadata.csv") in msg
    else:
        with pytest.raises(ValueError, match=r"is missing required columns"):
            reader.load_row(0)


def test_load_row_empty_value_raises_value_error(tmp_path: Path):
    """When a required column is present but its content is empty (e.g. an empty
    label string), ``load_row`` should eventually raise a ``ValueError`` during
    the conversion / class-index step.
    """
    write_csv(tmp_path, [(0, "", 0, 192_000.0)])  # empty label
    reader = MetadataReader(tmp_path)

    with pytest.raises(ValueError, match=r"is missing required columns"):
        reader.load_row(0)


def test_load_row_no_rows(tmp_path: Path):
    """An empty ``metadata.csv`` (zero lines) must cause ``load_row`` to raise
    ``MetadataIndexError`` because there is nothing to read.
    """
    # Create an *empty* CSV file -- no rows at all.
    (tmp_path / "metadata.csv").touch()
    reader = MetadataReader(tmp_path)
    with pytest.raises(MetadataIndexError) as excinfo:
        reader.load_row(0)
    msg = str(excinfo.value)
    assert "Metadata idx 0 is out of bounds" in msg
    assert "metadata.csv" in msg
