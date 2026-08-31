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

The heavy binary payload lives in the ``*.npy`` files; the human readable
description lives in the CSV.  This separation keeps loading fast (memory mapped
NumPy) while allowing easy inspection and editing of labels, modulation codes,
etc.
"""

# ----------------------------------------------------------------------
# Standard / third-party imports
# ----------------------------------------------------------------------
import bisect
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# TorchSig imports
# ----------------------------------------------------------------------
from torchsig.signals.signal_types import Signal

from .metadata_reader import MetadataReader

__all__ = ["NPYReader"]


class NPYReader(MetadataReader):
    """Read a directory that contains ``*.npy`` files, a ``metadata.csv`` and an
    ``info.json``.

    The class presents the whole collection as a flat, indexable dataset:
    ``reader[idx]`` returns a :class:`~torchsig.signals.signal_types.Signal`
    whose ``data`` attribute holds the waveform (as a 1-D ``np.ndarray``) and
    whose ``metadata`` attribute holds the parsed CSV row for that index.

    Args:
        root: Path to the directory that holds the ``*.npy`` files,
              ``metadata.csv`` and ``info.json``.  ``root`` may be a string or a
              :class:`Path`.

    Attributes:
        npy_files: List[Path] - sorted list of discovered ``*.npy`` files.
        file_start_indices: List[int] - cumulative start index of each file
            in the global index space.
        total_elements: int - actual number of samples stored across all
            ``*.npy`` files.
        class_list: List[str] - ordered list of class names used to compute
            ``class_index``.
        dataset_size: int - sample size advertised by ``info.json`` (returned by
            ``len(reader)``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)

        # ------------------------------------------------------------------
        # 1️⃣ Discover all ``*.npy`` files, sorted alphabetically for
        #     deterministic behaviour.
        # ------------------------------------------------------------------
        self.npy_files = sorted(self.root.glob("*.npy"))
        if not self.npy_files:
            raise FileNotFoundError("No .npy files found in directory.")

        # --------------------------------------------------------------
        # 2️⃣  Build a lookup table that maps a *global* index to the file that
        #     contains it.  ``self.file_start_indices[i]`` is the global
        #     offset of the first waveform in ``self.npy_files[i]``.
        # --------------------------------------------------------------
        self.file_start_indices: list[int] = []
        total = 0
        for file_path in self.npy_files:
            # Memory-map the file only to read its shape; the data stays on disk.
            arr = np.load(file_path, mmap_mode="r")
            length = arr.shape[0]  # number of waveforms in this file
            self.file_start_indices.append(total)
            total += length

        self.total_elements: int = total

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
            raise IndexError(f"Index {idx} out of range (0 ≤ idx < {self.total_elements})")

        # --------------------------------------------------------------
        # 1️⃣ Identify the file that contains this global index (binary search)
        # --------------------------------------------------------------
        file_idx = bisect.bisect_right(self.file_start_indices, idx) - 1
        in_file_idx = idx - self.file_start_indices[file_idx]

        # --------------------------------------------------------------
        # 2️⃣  Load the required .npy file (memory-mapped) and fetch the sample
        # --------------------------------------------------------------
        file_path = self.npy_files[file_idx]
        arr = np.load(file_path, mmap_mode="r")  # memmap view
        raw_sample = arr[in_file_idx]  # scalar (real-only in fixtures)
        data = np.atleast_1d(raw_sample)  # ensure ``len(data)`` works

        # --------------------------------------------------------------
        # 3️⃣ Pull the associated CSV row (metadata) and apply the deterministic
        #    class mapping.
        # --------------------------------------------------------------
        metadata = self.load_row(idx, self.class_list)

        # --------------------------------------------------------------
        # 4️⃣ Build the Signal object and hand it back to the caller
        # --------------------------------------------------------------
        return Signal(data=data, component_signals=[], metadata=metadata)

    # ----------------------------------------------------------------------
    # Length protocol -- reports the *advertised* dataset size from info.json
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the size declared in ``info.json``."""
        return self.dataset_size
