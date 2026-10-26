"""Shared sidecar metadata support for file-based TorchSIG datasets.

The :class:`MetadataReader` is the common helper used by file handlers whose
waveform payload and descriptive metadata are stored separately. It reads two
sidecars from the dataset root:

* ``metadata.csv`` contains one record per dataset element. New files use a
  versioned, header-based schema; valid legacy headerless files remain
  supported.
* ``info.json`` optionally supplies dataset-wide values such as sample count,
  sample rate, file layout, and class ordering.

CSV rows are parsed and validated once when the reader is constructed. Keeping
the resulting table in memory makes arbitrary and shuffled row access O(1), at
the cost of memory proportional to the number and width of metadata records.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from .base_handler import FileReader

__all__ = ["SIDECAR_SCHEMA_VERSION", "MetadataIndexError", "MetadataReader"]

SIDECAR_SCHEMA_VERSION = "1.0"
"""Current header-based ``metadata.csv`` sidecar schema version."""

_REQUIRED_COLUMNS = ("index", "label", "modcod", "sample_rate")
_LEGACY_OPTIONAL_COLUMNS = ("snr_db", "seed")
_DEFAULT_CLASS_LIST = ["BPSK", "QPSK", "Noise"]


class MetadataIndexError(IndexError):
    """Raised when a caller requests a metadata row that does not exist."""


class MetadataReader(FileReader):
    """Read validated dataset metadata from CSV and optional JSON sidecars.

    Header-based CSV files resolve fields by name, preserve additional columns,
    and may reorder the required columns. Legacy headerless files with four
    columns remain supported; the historical six-column generator output is
    also accepted. Metadata rows are parsed once during construction and kept
    in memory, making shuffled row lookup O(1) without retaining waveform data.

    ``info.json`` is supplementary. When ``metadata.csv`` exists, its record
    count determines dataset size and a conflicting JSON ``size`` is rejected.
    Class names are inferred in first-seen order when JSON does not provide a
    valid ``class_list``.

    Parameters
    ----------
    root : str | Path
        Directory containing ``metadata.csv`` and, optionally, ``info.json``.
        A string is converted to a :class:`~pathlib.Path` by the base reader.

    Attributes
    ----------
    root : Path
        Dataset directory normalized by :class:`FileReader`.
    dataset_metadata : dict[str, object]
        Parsed ``info.json`` object, or an empty dictionary when the optional
        file is absent, malformed, or does not contain a JSON object.
    metadata_path : Path
        Path to the dataset's ``metadata.csv`` sidecar.
    metadata_has_header : bool
        Whether the CSV was recognized as the header-based schema.
    metadata_fieldnames : tuple[str, ...]
        Normalized CSV field names, including preserved extension columns.
    dataset_size : int
        Number of parsed CSV records when ``metadata.csv`` exists; otherwise
        the valid nonnegative size declared by ``info.json``, or zero.
    class_list : list[str]
        JSON class ordering when valid, otherwise labels inferred in first-seen
        CSV order. The historical BPSK/QPSK/Noise default is used only when
        neither source supplies classes.
    sample_rate, num_files, elements_per_file, num_iq_samples : int
        Optional nonnegative dataset-wide values read from ``info.json``;
        missing or invalid values default to zero so concrete readers may
        infer them from their waveform files.

    Raises
    ------
    ValueError
        If the CSV schema or a record is invalid, indices are negative or
        duplicated, the declared schema version is unsupported, or the JSON
        size contradicts the CSV record count.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        try:
            self.dataset_metadata = self.load_json()
        except (TypeError, ValueError):
            self.dataset_metadata = {}

        self.metadata_path = self.root / "metadata.csv"
        self.metadata_has_header = False
        self.metadata_fieldnames: tuple[str, ...] = ()
        self._metadata_rows = self._load_metadata_rows() if self.metadata_path.exists() else []
        self._populate_metadata_attributes()

    def __repr__(self) -> str:
        """Return a concise representation for debugging and logging.

        The representation includes the concrete reader class, normalized
        dataset root, and reconciled dataset size.
        """
        return f"{self.__class__.__name__}(root={self.root!s}, size={self.dataset_size})"

    @staticmethod
    def _optional_nonnegative_int(value: Any, name: str) -> int | None:
        """Convert an optional JSON value to a nonnegative integer."""
        if value is None or isinstance(value, bool):
            return None
        try:
            converted = int(value)
        except (TypeError, ValueError):
            return None
        if converted < 0:
            raise ValueError(f"info.json field {name!r} must be nonnegative")
        return converted

    def _populate_metadata_attributes(self) -> None:
        """Reconcile optional JSON attributes with the CSV record index."""
        declared_version = self.dataset_metadata.get("sidecar_schema_version")
        if declared_version is not None and str(declared_version) != SIDECAR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported sidecar schema version {declared_version!r}; expected {SIDECAR_SCHEMA_VERSION!r}")

        declared_size = self._optional_nonnegative_int(self.dataset_metadata.get("size"), "size")
        if self.metadata_path.exists():
            csv_size = len(self._metadata_rows)
            if declared_size is not None and declared_size != csv_size:
                raise ValueError(f"info.json reports {declared_size} elements, but metadata.csv contains {csv_size} rows")
            self.dataset_size = csv_size
        else:
            self.dataset_size = declared_size or 0

        for name in ("sample_rate", "num_files", "elements_per_file", "num_iq_samples"):
            setattr(self, name, self._optional_nonnegative_int(self.dataset_metadata.get(name), name) or 0)

        raw_class_list = self.dataset_metadata.get("class_list")
        if isinstance(raw_class_list, list) and all(isinstance(item, str) for item in raw_class_list):
            self.class_list = list(raw_class_list)
            return

        inferred = list(dict.fromkeys(str(row["label"]) for row in self._metadata_rows))
        self.class_list = inferred or list(_DEFAULT_CLASS_LIST)

    @staticmethod
    def _is_header(row: list[str]) -> bool:
        """Return whether a first CSV row is recognizably a header."""
        normalized = {value.strip() for value in row}
        return bool(normalized.intersection(_REQUIRED_COLUMNS))

    @staticmethod
    def _validate_header(header: list[str], csv_path: Path) -> tuple[str, ...]:
        """Validate and normalize a header row."""
        normalized = tuple(value.strip() for value in header)
        if any(not value for value in normalized):
            raise ValueError(f"{csv_path} contains an empty column name")
        duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
        if duplicates:
            raise ValueError(f"{csv_path} contains duplicate columns: {duplicates}")
        missing = sorted(set(_REQUIRED_COLUMNS).difference(normalized))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        return normalized

    def _load_metadata_rows(self) -> list[dict[str, Any]]:
        """Parse and validate all CSV rows into an O(1) lookup table."""
        with self.metadata_path.open("r", encoding="utf-8", newline="") as csv_file:
            raw_rows = [row for row in csv.reader(csv_file) if row]

        if not raw_rows:
            return []

        first_row = raw_rows[0]
        self.metadata_has_header = self._is_header(first_row)
        if self.metadata_has_header:
            fieldnames = self._validate_header(first_row, self.metadata_path)
            data_rows = raw_rows[1:]
        else:
            legacy_column_count = len(_REQUIRED_COLUMNS)
            if len(first_row) < legacy_column_count:
                missing = list(_REQUIRED_COLUMNS[len(first_row) :])
                raise ValueError(f"Row 0 of {self.metadata_path} is missing required columns: {missing}")
            if len(first_row) not in {legacy_column_count, legacy_column_count + len(_LEGACY_OPTIONAL_COLUMNS)}:
                raise ValueError(f"Legacy {self.metadata_path} rows must contain 4 or 6 columns; row 0 contains {len(first_row)}")
            fieldnames = _REQUIRED_COLUMNS + _LEGACY_OPTIONAL_COLUMNS[: len(first_row) - legacy_column_count]
            data_rows = raw_rows

        self.metadata_fieldnames = tuple(fieldnames)
        records: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for row_index, values in enumerate(data_rows):
            if len(values) != len(fieldnames):
                raise ValueError(f"Row {row_index} of {self.metadata_path} has {len(values)} values; expected {len(fieldnames)} from its schema")
            raw_record = dict(zip(fieldnames, values, strict=True))
            missing = sorted(name for name in _REQUIRED_COLUMNS if raw_record[name] == "")
            if missing:
                raise ValueError(f"Row {row_index} of {self.metadata_path} is missing required columns: {missing}")
            record = self._parse_record(raw_record)
            index = record["index"]
            if index < 0:
                raise ValueError(f"Row {row_index} of {self.metadata_path} has negative index {index}")
            if index in seen_indices:
                raise ValueError(f"Row {row_index} of {self.metadata_path} has duplicate index {index}")
            seen_indices.add(index)
            records.append(record)
        return records

    @staticmethod
    def _parse_record(raw_record: dict[str, str]) -> dict[str, Any]:
        """Convert required fields while retaining additional CSV columns."""
        record: dict[str, Any] = dict(raw_record)
        try:
            record["index"] = int(raw_record["index"])
        except ValueError as exc:
            raise ValueError(f"Cannot convert 'index'='{raw_record['index']}' to int") from exc
        try:
            record["modcod"] = int(raw_record["modcod"])
        except ValueError as exc:
            raise ValueError(f"Cannot convert 'modcod'='{raw_record['modcod']}' to int") from exc
        try:
            record["sample_rate"] = float(raw_record["sample_rate"])
        except ValueError as exc:
            raise ValueError(f"Cannot convert 'sample_rate'='{raw_record['sample_rate']}' to float") from exc
        record["label"] = raw_record["label"]
        return record

    def load_row(self, idx: int, class_list: list[str] | None = None) -> dict[str, object]:
        """Return parsed metadata for the zero-based row position ``idx``.

        Additional header-based columns are returned as strings. Required
        numeric fields are converted to their documented numeric types.

        Parameters
        ----------
        idx : int
            Zero-based row position in ``metadata.csv``. This is the record's
            table position and need not equal the value of its ``index`` field.
        class_list : list[str] | None, optional
            Ordered class names used to derive ``class_index``. If omitted,
            the instance's reconciled :attr:`class_list` is used. A label not
            present in that list receives a class index of ``-1``.

        Returns
        -------
        dict[str, object]
            A fresh dictionary containing the CSV fields. ``index`` and
            ``modcod`` are integers, ``sample_rate`` is a float, and additional
            header columns remain strings. Derived fields include lower-case
            ``class_name``, numeric ``class_index``, and ``num_signals_max=1``.

        Raises
        ------
        MetadataIndexError
            If ``idx`` is negative or outside the parsed metadata table.
        """
        if idx < 0 or idx >= len(self._metadata_rows):
            raise MetadataIndexError(f"Metadata idx {idx} is out of bounds (file has fewer rows) - {self.metadata_path}")

        classes = self.class_list if class_list is None else class_list
        record = dict(self._metadata_rows[idx])
        label = str(record["label"])
        record["class_name"] = label.lower()
        record["num_signals_max"] = 1
        try:
            record["class_index"] = classes.index(label)
        except ValueError:
            record["class_index"] = -1
        return record

    def load_json(self) -> dict[str, object]:
        """Read and return optional dataset-level JSON metadata.

        Returns
        -------
        dict[str, object]
            The decoded JSON object without schema-specific coercion.

        Raises
        ------
        TypeError
            If the decoded JSON's top-level value is not an object.
        ValueError
            If ``info.json`` is missing or cannot be decoded as JSON.
            Construction treats these cases as absent optional metadata;
            direct callers can inspect the error.
        """
        meta_path = self.root / "info.json"
        try:
            with meta_path.open(encoding="utf-8") as json_file:
                payload = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read {meta_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"Cannot read {meta_path}: expected a JSON object")
        return payload
