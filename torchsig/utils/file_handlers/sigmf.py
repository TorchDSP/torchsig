"""File handlers for SigMF .sigmf-data / .sigmf-meta file pairs."""

# Built-In
import json
from pathlib import Path
from typing import Optional

# Third Party
import numpy as np

# TorchSig
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.base_handler import BaseFileHandler, FileReader, FileWriter

_SIGMF_DTYPE_MAP: dict = {
    "cf32_le": (np.float32, "<"),
    "cf32_be": (np.float32, ">"),
    "cf64_le": (np.float64, "<"),
    "cf64_be": (np.float64, ">"),
    "ci32_le": (np.int32,   "<"),
    "ci32_be": (np.int32,   ">"),
    "ci16_le": (np.int16,   "<"),
    "ci16_be": (np.int16,   ">"),
    "ci8":     (np.int8,    "<"),
    "ci8_le":  (np.int8,    "<"),
    "cu8":     (np.uint8,   "<"),
    "cu8_le":  (np.uint8,   "<"),
}

_SIGMF_SCALE_MAP: dict = {
    np.float32: 1.0,
    np.float64: 1.0,
    np.int32:   1.0 / 2_147_483_648.0,
    np.int16:   1.0 / 32_768.0,
    np.int8:    1.0 / 128.0,
    np.uint8:   1.0 / 128.0,
}


def _parse_sigmf_meta(path: Path) -> dict:
    """Parse a SigMF metadata sidecar file into a dict.

    Args:
        path (Path): Path to the .sigmf-meta JSON file.

    Returns:
        dict: Parsed SigMF metadata.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"SigMF meta file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_meta_for_data(data_path: Path) -> Path:
    """Locate the .sigmf-meta sidecar for a .sigmf-data file.

    Tries two naming conventions:

    1. ``foo.sigmf-data`` → ``foo.sigmf-meta``
    2. ``foo.sigmf-data`` → ``foo.sigmf-data.sigmf-meta``

    Args:
        data_path (Path): Path to the ``.sigmf-data`` file.

    Returns:
        Path: Path to the discovered ``.sigmf-meta`` file.

    Raises:
        FileNotFoundError: If neither naming convention yields an existing file.
    """
    candidate_a = data_path.with_suffix("").with_suffix(".sigmf-meta")
    candidate_b = Path(str(data_path) + ".sigmf-meta")
    for candidate in (candidate_a, candidate_b):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No .sigmf-meta sidecar found for: {data_path}\n"
        f"Tried: {candidate_a}, {candidate_b}"
    )


def _find_capture_freq(captures: list, sample_start: int) -> Optional[float]:
    """Return the center frequency in effect at *sample_start*.

    Walks the captures list (sorted by ``core:sample_start``) and returns the
    ``core:frequency`` value from the latest capture whose start index is <=
    *sample_start*.

    Args:
        captures (list): List of capture segment dicts from the SigMF metadata.
        sample_start (int): Sample offset within the data file.

    Returns:
        Optional[float]: Center frequency in Hz, or ``None`` if captures is
        empty or no entry carries a ``core:frequency`` field.
    """
    if not captures:
        return None
    freq = None
    for capture in sorted(captures, key=lambda c: c.get("core:sample_start", 0)):
        if capture.get("core:sample_start", 0) <= sample_start:
            if "core:frequency" in capture:
                freq = float(capture["core:frequency"])
    return freq


class SigMFReader(FileReader):
    """Read SigMF file pairs as a TorchSig dataset.

    Root directory must contain one or more ``.sigmf-data`` files with
    accompanying ``.sigmf-meta`` sidecar files. Each annotation in the
    ``.sigmf-meta`` describes one signal sample; all annotations from all
    files are concatenated in sorted file-name order.

    Args:
        root (str): Directory containing ``.sigmf-data`` / ``.sigmf-meta`` pairs.

    Raises:
        FileNotFoundError: If no ``.sigmf-data`` files are found in *root*.
        ValueError: If a ``.sigmf-meta`` contains an unsupported datatype.
        NotImplementedError: If a file declares ``core:num_channels > 1``.
    """

    def __init__(self, root: str, **kwargs):
        """Initialize SigMFReader by scanning *root* for SigMF file pairs.

        Args:
            root (str): Directory path containing ``.sigmf-data`` files.
            **kwargs: Forwarded to :class:`FileReader`.
        """
        super().__init__(root=root, **kwargs)

        data_files = sorted(Path(root).glob("*.sigmf-data"))
        if not data_files:
            raise FileNotFoundError(
                f"No .sigmf-data files found in directory: {root}"
            )

        self._index: list = []
        class_set: set = set()

        for data_path in data_files:
            meta_path = _find_meta_for_data(data_path)
            meta = _parse_sigmf_meta(meta_path)

            num_channels = meta.get("global", {}).get("core:num_channels", 1)
            if num_channels > 1:
                raise NotImplementedError(
                    "SigMFReader does not support multi-channel SigMF files "
                    f"(core:num_channels > 1) in file: {data_path}"
                )

            datatype = meta.get("global", {}).get("core:datatype", "")
            if datatype not in _SIGMF_DTYPE_MAP:
                raise ValueError(
                    f"Unsupported SigMF datatype {datatype!r} in {meta_path}. "
                    f"Supported types: {list(_SIGMF_DTYPE_MAP.keys())}"
                )

            for annotation in meta.get("annotations", []):
                label = annotation.get("core:label", "") or ""
                class_name = label.lower() if label else "unknown"
                class_set.add(class_name)
                self._index.append((data_path, meta, annotation))

        self._class_list: list = sorted(class_set)

    def __len__(self) -> int:
        """Return the total number of annotated signal samples.

        Returns:
            int: Number of annotations across all SigMF files in the root.
        """
        return len(self._index)

    def read(self, idx: int) -> Signal:
        """Load and return the signal sample at index *idx*.

        Reads only the byte range corresponding to the annotation from the
        ``.sigmf-data`` file (seek + read, not full-file memmap).

        Args:
            idx (int): Zero-based sample index.

        Returns:
            Signal: TorchSig Signal with complex64 IQ data and metadata.

        Raises:
            IndexError: If *idx* is out of range.
            ValueError: If the datatype is unsupported (also caught at init).
        """
        if idx < 0 or idx >= len(self._index):
            raise IndexError(
                f"Index {idx} out of range (dataset length: {len(self._index)})"
            )

        data_path, meta, annotation = self._index[idx]

        sample_start = annotation.get("core:sample_start", 0)
        sample_count = annotation.get("core:sample_count", 0)

        datatype = meta["global"]["core:datatype"]
        elem_type, endian = _SIGMF_DTYPE_MAP[datatype]
        scale = _SIGMF_SCALE_MAP[elem_type]

        bytes_per_sample = np.dtype(elem_type).itemsize * 2  # I + Q interleaved
        byte_offset = sample_start * bytes_per_sample
        byte_count = sample_count * bytes_per_sample

        with open(data_path, "rb") as fobj:
            fobj.seek(byte_offset)
            buf = fobj.read(byte_count)

        raw = np.frombuffer(buf, dtype=np.dtype(elem_type).newbyteorder(endian))

        # Unsigned 8-bit: re-centre around zero before scaling
        if elem_type == np.uint8:
            raw = raw.astype(np.float32) - 128.0
            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) * scale
        else:
            raw_f = raw.astype(np.float32) * scale
            iq = (raw_f[0::2] + 1j * raw_f[1::2]).astype(np.complex64)

        # --- Build metadata dict ---
        sample_rate = float(meta["global"].get("core:sample_rate", 0.0))
        captures = meta.get("captures", [])
        center_freq = _find_capture_freq(captures, sample_start)

        label = annotation.get("core:label", "") or ""
        class_name = label.lower() if label else "unknown"
        class_index = (
            self._class_list.index(class_name)
            if class_name in self._class_list
            else -1
        )

        metadata = {
            "sample_rate": sample_rate,
            "class_name": class_name,
            "class_index": class_index,
            "num_iq_samples": sample_count,
            "num_signals_max": 1,
        }

        if center_freq is not None:
            metadata["center_freq"] = center_freq

        # Pass through non-core annotation keys and extra core fields verbatim
        _skip_keys = {"core:sample_start", "core:sample_count", "core:label"}
        for ann_key, ann_val in annotation.items():
            if ann_key not in _skip_keys:
                metadata[ann_key] = ann_val

        return Signal(data=iq, component_signals=[], **metadata)


class SigMFFileHandler(BaseFileHandler):
    """File handler for SigMF datasets (read-only).

    Usage:
        >>> handler = SigMFFileHandler.create_handler(mode="r", root="/path/to/sigmf/")
    """

    reader_class = SigMFReader
    writer_class = FileWriter

    @staticmethod
    def create_handler(mode: str, root: str, **kwargs):
        """Create a SigMF reader (write mode is not supported).

        Args:
            mode (str): Must be ``"r"``; write mode raises NotImplementedError.
            root (str): Directory containing ``.sigmf-data`` / ``.sigmf-meta`` pairs.
            **kwargs: Forwarded to :class:`SigMFReader`.

        Returns:
            SigMFReader: A reader for the given root directory.

        Raises:
            NotImplementedError: If *mode* is not ``"r"``.
        """
        if mode == "r":
            return SigMFFileHandler.reader_class(root=root, **kwargs)
        raise NotImplementedError(
            f"SigMFFileHandler does not support mode {mode!r}; only 'r' is supported."
        )
