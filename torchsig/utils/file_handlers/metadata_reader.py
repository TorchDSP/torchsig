"""Metadata loader for TorchSig.

The :class:`MetadataReader` is the tiny helper that every TorchSig file-handler
relies on.  It offers two read-only utilities that never pull the whole CSV
into memory:

* :meth:`MetadataReader.load_row` - fetches **one** row from ``metadata.csv``,
  parses the fields and returns a fully-typed dictionary.
* :meth:`MetadataReader.load_json` - reads the optional ``info.json`` that lives
  next to the CSV and returns its parsed JSON payload.

Both helpers accept either a plain ``str`` or a :class:`Path` as the
dataset root, raise explicit, custom exceptions on error, and use
``itertools.islice`` so that only the requested line is read from disk.
"""

from __future__ import annotations

import csv
import itertools
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from .base_handler import FileReader

__all__ = ["MetadataIndexError", "MetadataReader"]


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------
class MetadataIndexError(IndexError):
    """Raised when a caller asks for a CSV row that does not exist."""


# ----------------------------------------------------------------------
# Core reader
# ----------------------------------------------------------------------
class MetadataReader(FileReader):
    """Minimal-overhead reader for the dataset-wide metadata files that accompany
    TorchSig recordings.

    Parameters
    ----------
    root : str | Path
        Directory that contains ``metadata.csv`` and, optionally,
        ``info.json``.  If a string is supplied it is converted to a
        :class:`~Path`.

    Attributes
    ----------
    root : Path
        Normalised (absolute) path to the dataset folder.
    metadata : dict
        The parsed contents of ``info.json`` - empty when the file is missing
        or cannot be decoded.
    dataset_size : int
        The *declared* number of samples, taken from the ``size`` entry in
        ``info.json``.  If the entry is missing or invalid the value defaults to
        ``0``.
    class_list : list[str]
        Ordered list of class names.  It is taken from the ``class_list`` entry
        in ``info.json`` when present; otherwise the default list
        ``["BPSK", "QPSK", "Noise"]`` is used.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)

        # Load the optional JSON metadata.  Any problem (missing file,
        # malformed JSON, …) results in an empty dict -- the rest of the class
        # can continue to operate with sensible defaults.
        try:
            self.dataset_metadata = self.load_json()
        except ValueError:
            self.dataset_metadata = {}

        # Populate the convenience attributes from the JSON payload.
        self._populate_from_json()

    def __repr__(self) -> str:
        """Return a concise representation of the metadata reader.

        The string shows the class name, the resolved ``root`` and the
        advertised ``dataset_size`` (taken from ``info.json`` if present,
        otherwise ``0``).  This is useful for debugging and logging.
        """
        return f"{self.__class__.__name__}(root={self.root!s}, size={self.dataset_size})"

    # ------------------------------------------------------------------
    # Private helper -- turn the JSON payload into public attributes
    # ------------------------------------------------------------------
    def _populate_from_json(self) -> None:
        """Populate attributes from ``self.dataset_metadata``."""

        def _try_get_int(name: str) -> int:
            try:
                return int(self.dataset_metadata.get(name, 0))
            except (TypeError, ValueError):
                return 0

        self.dataset_size = _try_get_int("size")
        self.sample_rate: int = _try_get_int("sample_rate")
        self.num_files: int = _try_get_int("num_files")
        self.elements_per_file: int = _try_get_int("elements_per_file")
        self.num_iq_samples: int = _try_get_int("num_iq_samples")

        _default = ["BPSK", "QPSK", "Noise"]
        raw = self.dataset_metadata.get("class_list", _default)
        if isinstance(raw, list) and all(isinstance(i, str) for i in raw):
            self.class_list = raw
        else:
            self.class_list = _default

    def load_row(
        self,
        idx: int,
        class_list: list[str] | None = None,
    ) -> dict[str, object]:
        """Return a dictionary with the parsed metadata for row ``idx`` of
        ``metadata.csv``.

        Parameters
        ----------
        idx : int
            Zero-based row index to read.
        class_list : list[str] | None, optional
            Ordered list of class names.  If omitted the instance's
            ``self.class_list`` is used.  Labels that are not present in the list
            receive a ``class_index`` of ``-1``.

        Returns
        -------
        dict
            Dictionary containing the raw CSV fields plus a few derived entries:
            ``index``, ``label``, ``modcod``, ``sample_rate``,
            ``class_name`` (lower-cased label), ``class_index`` and
            ``num_signals_max`` (always ``1``).

        Raises
        ------
        MetadataIndexError
            If ``idx`` is larger than the number of rows in the CSV file.
        ValueError
            If a required column is missing/empty or cannot be cast to the
            expected type.
        """
        csv_path = self.root / "metadata.csv"

        # Use the supplied class list or fall back to the instance attribute.
        if class_list is None:
            class_list = self.class_list

        # --------------------------------------------------------------
        # 1️⃣  Read only the requested line -- we never load the whole CSV.
        # --------------------------------------------------------------
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(
                f,
                fieldnames=["index", "label", "modcod", "sample_rate"],
            )
            row = next(itertools.islice(reader, idx, idx + 1), None)

        # --------------------------------------------------------------
        # 2️⃣  Guard against an out-of-range request.
        # --------------------------------------------------------------
        if row is None:
            raise MetadataIndexError(f"Metadata idx {idx} is out of bounds (file has fewer rows) - {csv_path}")

        # --------------------------------------------------------------
        # 3️⃣  Verify that every required column is present and non-empty.
        # --------------------------------------------------------------
        required = {"index", "label", "modcod", "sample_rate"}
        if any(row[col] in (None, "") for col in required):
            raise ValueError(f"Row {idx} of {csv_path} is missing required columns: {[col for col in required if row[col] in (None, '')]!r}")

        # --------------------------------------------------------------
        # 4️⃣  Convert the raw strings to their proper Python types,
        #     preserving the original traceback on failure.
        # --------------------------------------------------------------
        try:
            index = int(row["index"])
        except ValueError as exc:
            raise ValueError(f"Cannot convert 'index'='{row['index']}' to int") from exc

        label = row["label"]

        try:
            modcod = int(row["modcod"])
        except ValueError as exc:
            raise ValueError(f"Cannot convert 'modcod'='{row['modcod']}' to int") from exc

        try:
            sample_rate = float(row["sample_rate"])
        except ValueError as exc:
            raise ValueError(f"Cannot convert 'sample_rate'='{row['sample_rate']}' to float") from exc

        # --------------------------------------------------------------
        # 5️⃣  Build the result dictionary and add the derived helper fields.
        # --------------------------------------------------------------
        record: dict[str, object] = {
            "index": index,
            "label": label,
            "modcod": modcod,
            "sample_rate": sample_rate,
            "class_name": label.lower(),
            "num_signals_max": 1,
        }

        # --------------------------------------------------------------
        # 6️⃣  Resolve the numeric class index (-1 if the label is unknown).
        # --------------------------------------------------------------
        try:
            record["class_index"] = class_list.index(label)
        except ValueError:
            record["class_index"] = -1

        return record

    def load_json(self) -> dict[str, object]:
        """Load the optional ``info.json`` file that lives next to ``metadata.csv``.

        Returns
        -------
        dict
            The parsed JSON payload.

        Raises
        ------
        ValueError
            If the file does not exist or cannot be decoded as JSON.
        """
        meta_path = self.root / "info.json"
        try:
            # The JSON files shipped with TorchSig are UTF-8 encoded.
            with meta_path.open(encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read {meta_path}: {exc}") from exc
