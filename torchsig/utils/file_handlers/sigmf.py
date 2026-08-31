"""File-handler that reads a directory of SigMF IQ recordings.

Each SigMF recording stores ``elements_per_file`` whole IQ records; every
record contains ``num_iq_samples`` complex samples.
"""

import bisect
import json
from pathlib import Path

import numpy as np

from torchsig.signals.signal_types import Signal

from .metadata_reader import MetadataReader

__all__ = ["SIGMFReader"]


class SigMFReader(MetadataReader):
    """Read a directory that contains a TorchSig-compatible SigMF dataset.

    Required files
    --------------
    * ``*.sigmf-data`` - binary IQ sample files.
    * ``*.sigmf-meta`` - SigMF metadata files associated with the data files.
    * ``metadata.csv`` - one row per element, not per IQ sample.
    * Optional ``info.json`` - dataset-level TorchSig metadata such as
      ``num_iq_samples``, ``elements_per_file``, dataset size, class list,
      sample rate, and dtype.

    Public API
    ----------
    ``reader.read(i)`` returns the i-th IQ record as a
    ``torchsig.signals.Signal`` whose ``data`` attribute is a ``complex64``
    array of shape ``(num_iq_samples,)``.
    """

    DEFAULT_DTYPE = "cf32_le"

    SIGMF_DTYPE_MAP = {
        "cf32_le": np.dtype("<c8"),
        "cf32_be": np.dtype(">c8"),
        "ci16_le": np.dtype([("i", "<i2"), ("q", "<i2")]),
        "ci16_be": np.dtype([("i", ">i2"), ("q", ">i2")]),
        "ci8": np.dtype([("i", "i1"), ("q", "i1")]),
        "cu8": np.dtype([("i", "u1"), ("q", "u1")]),
    }

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.root = Path(self.root).resolve()

        self.data_files = sorted(self.root.rglob("*.sigmf-data"), key=lambda p: str(p))
        if not self.data_files:
            raise FileNotFoundError(f"No .sigmf-data files found in {self.root}")

        self.meta_files = {path.with_suffix(".sigmf-data"): path for path in self.root.rglob("*.sigmf-meta")}

        csv_path = self.root / "metadata.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                rows = [line for line in f if line.strip()]
            self.dataset_size = len(rows)
            if hasattr(self, "metadata_rows"):
                self.metadata_rows = rows

        self.sigmf_metadata = [self._load_sigmf_meta(path) for path in self.data_files]
        self.sample_dtype = self._resolve_dtype()
        self.numpy_dtype = self._numpy_dtype(self.sample_dtype)

        if self.num_iq_samples == 0 or self.elements_per_file == 0:
            first_samples = self._num_complex_samples(self.data_files[0])

            if self.elements_per_file == 0:
                if self.dataset_size > 0 and self.dataset_size % len(self.data_files) == 0:
                    self.elements_per_file = self.dataset_size // len(self.data_files)
                else:
                    self.elements_per_file = 1

            if self.num_iq_samples == 0:
                if first_samples % self.elements_per_file != 0:
                    raise ValueError("Cannot infer num_iq_samples: first SigMF data file does not contain a whole number of elements.")
                self.num_iq_samples = first_samples // self.elements_per_file

        self.file_start_indices: list[int] = []
        cum = 0

        for data_file in self.data_files:
            samples = self._num_complex_samples(data_file)
            if samples % self.num_iq_samples != 0:
                raise ValueError(f"{data_file} contains {samples} complex samples, which is not a multiple of num_iq_samples={self.num_iq_samples}.")

            elements_in_file = samples // self.num_iq_samples
            if elements_in_file != self.elements_per_file:
                raise ValueError(f"{data_file} contains {elements_in_file} elements, expected {self.elements_per_file}.")

            self.file_start_indices.append(cum)
            cum += elements_in_file

        self.total_elements = cum
        self.dataset_size = max(self.dataset_size, self.total_elements)

    def read(self, idx: int) -> Signal:
        """Read one dataset element by global element index."""
        if idx < 0 or idx >= self.dataset_size:
            raise IndexError(f"index {idx} out of range")

        file_idx = bisect.bisect_right(self.file_start_indices, idx) - 1
        element_offset = idx - self.file_start_indices[file_idx]
        data_path = self.data_files[file_idx]

        start_sample = element_offset * self.num_iq_samples
        raw = np.fromfile(
            data_path,
            dtype=self.numpy_dtype,
            count=self.num_iq_samples,
            offset=start_sample * self.numpy_dtype.itemsize,
        )

        complex_vec = self._to_complex64(raw)
        metadata = self.load_row(idx, self.class_list)

        return Signal(
            data=complex_vec,
            component_signals=[],
            metadata=metadata,
        )

    def __len__(self) -> int:
        """Number of elements in the dataset."""
        return self.dataset_size

    def _load_sigmf_meta(self, data_path: Path) -> dict:
        meta_path = data_path.with_suffix(".sigmf-meta")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing SigMF metadata file for {data_path}")

        with open(meta_path) as f:
            return json.load(f)

    def _resolve_dtype(self) -> str:
        for meta in self.sigmf_metadata:
            datatype = meta.get("global", {}).get("core:datatype")
            if datatype:
                return datatype

        return getattr(self, "datatype", self.DEFAULT_DTYPE)

    def _numpy_dtype(self, sigmf_dtype: str) -> np.dtype:
        try:
            return self.SIGMF_DTYPE_MAP[sigmf_dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported SigMF datatype: {sigmf_dtype}") from exc

    def _num_complex_samples(self, data_path: Path) -> int:
        size_bytes = data_path.stat().st_size
        if size_bytes % self.numpy_dtype.itemsize != 0:
            raise ValueError(f"{data_path} size is not divisible by dtype size {self.numpy_dtype.itemsize}.")

        return size_bytes // self.numpy_dtype.itemsize

    def _to_complex64(self, raw: np.ndarray) -> np.ndarray:
        if raw.dtype.fields is None:
            return raw.astype(np.complex64, copy=False)

        return (raw["i"].astype(np.float32) + 1j * raw["q"].astype(np.float32)).astype(np.complex64)
