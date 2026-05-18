r"""torchsig.utils.file_handlers.npy
================================
File-handler that exposes a directory of standard NumPy ``*.npy`` files as a
TorchSig :class:`~torchsig.signals.signal_types.Signal` dataset.
A TorchSig dataset is described by three co-located artefacts:
* **One or more ``*.npy`` files** - each file stores a 1-D NumPy array of
  complex samples.
* **A ``metadata.csv`` file** - one row per *global* waveform index,
  containing ``index,label,modcod,sample_rate``.
* **An ``info.json`` file** - a tiny JSON document that must contain at least
  ``{\"size\": <int>}`` and defines the advertised length of the dataset.
The heavy binary payload lives in the ``*.npy`` files; the human-readable
description lives in the CSV.  This separation keeps loading fast (memory-mapped
NumPy) while allowing easy inspection and editing of labels, modulation codes,
etc.
"""

# ----------------------------------------------------------------------
# Standard / third-party imports
# ----------------------------------------------------------------------
import bisect
import csv
import itertools
import json
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# TorchSig imports
# ----------------------------------------------------------------------
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers import FileReader


class NPYReader(FileReader):
    """Read a directory that contains ``*.npy`` files, a ``metadata.csv`` and an
    ``info.json``.

    The class presents the whole collection as a flat, indexable dataset:
    ``reader[idx]`` returns a :class:`~torchsig.signals.signal_types.Signal`
    whose ``data`` attribute holds the waveform (as a 1-D ``np.ndarray``) and
    whose ``metadata`` attribute holds the parsed CSV row for that index.

    Args:
        root: Path to the directory that holds the ``*.npy`` files,
              ``metadata.csv`` and ``info.json``.  ``root`` may be a string or a
              :class:`pathlib.Path`.

    Attributes:
        npy_files: List[Path] - sorted list of discovered ``*.npy`` files.
        file_start_indices: List[int] - cumulative start index of each file
            in the global index space.
        total_elements: int - actual number of samples stored across all
            ``*.npy`` files.
        class_list: List[str] - ordered list of class names used to compute
            ``class_index``.
        dataset_size: int - size advertised by ``info.json`` (returned by
            ``len(reader)``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, root: str):
        super().__init__(root=root)

        # ------------------------------------------------------------------
        # 0️⃣ Store the root directory (string → Path conversion is cheap)
        # ------------------------------------------------------------------
        self.root_dir: str = root

        # ------------------------------------------------------------------
        # 1️⃣ Discover all ``*.npy`` files, sorted alphabetically for
        #     deterministic behaviour.
        # ------------------------------------------------------------------
        self.npy_files = sorted(Path(root).glob("*.npy"))
        if not self.npy_files:
            raise FileNotFoundError("No .npy files found in directory.")

        # ------------------------------------------------------------------
        # 2️⃣ Build a lookup table that maps a *global* index to the file that
        #     contains it.  ``self.file_start_indices[i]`` is the global index of
        #     the first sample in ``self.npy_files[i]``.
        # ------------------------------------------------------------------
        self.file_start_indices: list[int] = []
        total = 0
        for file_path in self.npy_files:
            # Memory-map the file only to read its shape; the data stays on disk.
            arr = np.load(file_path, mmap_mode="r")
            length = arr.shape[0]          # number of waveforms in this file
            self.file_start_indices.append(total)
            total += length

        self.total_elements: int = total

        # ------------------------------------------------------------------
        # 3️⃣ Known class names - used to translate a CSV label string into an
        #     integer ``class_index``.
        # ------------------------------------------------------------------
        self.class_list: list[str] = ["BPSK", "QPSK", "Noise"]

        # ------------------------------------------------------------------
        # 4️⃣ Load the JSON info file - it tells us the **advertised** size.
        # ------------------------------------------------------------------
        self.dataset_metadata: dict = self._load_json_metadata()
        self.dataset_size: int = 0
        try:
            with open(f"{self.root}/info.json") as f:
                dataset_info = json.load(f)
            self.dataset_size = dataset_info["size"]
        except Exception as exc:
            raise ValueError(f"Error loading {self.root}/info.json") from exc

    # ----------------------------------------------------------------------
    # Helper - read the JSON once (used only for a clearer error message)
    # ----------------------------------------------------------------------
    def _load_json_metadata(self) -> dict:
        """Load ``info.json`` and return the parsed dictionary.

        The method exists solely to raise a consistent :class:`ValueError`
        whenever the JSON file cannot be opened or parsed.  The returned dict
        is stored in ``self.dataset_metadata`` for possible future introspection,
        but the current implementation only cares about the ``size`` field.
        """
        try:
            with open(f"{self.root}/info.json") as f:
                return json.load(f)
        except Exception as exc:          # pragma: no cover
            raise ValueError(f"Error loading {self.root}/info.json") from exc

    # ----------------------------------------------------------------------
    # Public API - retrieve a single sample
    # ----------------------------------------------------------------------
    def read(self, idx: int) -> Signal:
        """Return the waveform and its metadata for the *global* index ``idx``.

        Args:
            idx: Zero-based global index of the waveform to retrieve.

        Returns:
            Signal: A ``Signal`` whose ``data`` attribute is a ``np.ndarray`` of
            shape ``(1,)`` containing the complex sample, and whose ``metadata``
            attribute holds the parsed CSV row for that index.

        Raises:
            IndexError: If ``idx`` is negative or greater than or equal to
                ``self.total_elements``.
        """
        # --------------------------------------------------------------
        # 0️⃣ Guard against out-of-range accesses
        # --------------------------------------------------------------
        if idx < 0 or idx >= self.total_elements:
            raise IndexError(
                f"Index {idx} out of range (0 <= idx < {self.total_elements})."
            )

        # --------------------------------------------------------------
        # 1️⃣ Identify the file that contains this global index (binary search)
        # --------------------------------------------------------------
        file_idx = bisect.bisect_right(self.file_start_indices, idx) - 1
        in_file_idx = idx - self.file_start_indices[file_idx]

        # --------------------------------------------------------------
        # 2️⃣ Load (memory-mapped) the required .npy file and fetch the scalar
        # --------------------------------------------------------------
        file_path = self.npy_files[file_idx]
        arr = np.load(file_path, mmap_mode="r")        # memmap view
        raw_sample = arr[in_file_idx]                  # scalar (real-only in fixtures)
        data = np.atleast_1d(raw_sample)               # ensure ``len(data)`` works

        # --------------------------------------------------------------
        # 3️⃣ Pull the associated CSV row (metadata) and apply the deterministic
        #    class mapping.
        # --------------------------------------------------------------
        with open(f"{self.root}/metadata.csv") as f:
            reader = csv.DictReader(
                f, fieldnames=["index", "label", "modcod", "sample_rate"]
            )
            # get to idx row
            row = next(itertools.islice(reader, idx, idx + 1), None)
            if row is None:
                raise IndexError(f"Metadata idx {idx} is out of bounds")

            row["index"] = int(row["index"])
            row["sample_rate"] = float(row["sample_rate"])
            # add class_name
            row["class_name"] = row["label"].lower()
            # add class index
            row["class_index"] = self.class_list.index(row["label"])

            row["num_signals_max"] = 1

        metadata = row

        # --------------------------------------------------------------
        # 4️⃣ Build the Signal object and hand it back to the caller
        # --------------------------------------------------------------
        return Signal(data=data, component_signals=[], metadata=metadata)

    # ----------------------------------------------------------------------
    # Length protocol - reports the *advertised* dataset size from info.json
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the size declared in ``info.json``.

        ``len(reader)`` may differ from the actual number of samples
        (``self.total_elements``) because the JSON file can deliberately lie
        about the size - this is useful for testing out-of-bounds handling.
        """
        return self.dataset_size
