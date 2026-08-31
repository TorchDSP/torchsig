"""Simplified file handler for GeoDataset using .yaml/.dat pairs.

This module provides file handlers for writing and reading TorchSigGeoDataset
objects to/from .yaml (metadata) and .dat (IQ data) file pairs.

The format:
- .yaml: YAML file containing receiver metadata (lat, lon, alt, etc.)
- .dat: Binary file containing interleaved I/Q samples

For integer data types (int16, int32), IQ samples in [-1, 1] are scaled to
the symmetric range [-INT_MAX, INT_MAX].

Example usage:
    >>> from torchsig.geo.utils.file_handler import GeoDatasetWriter, GeoDatasetReader
    >>> from torchsig.geo.datasets import TorchSigGeoDataset
    >>>
    >>> # Write dataset to files
    >>> writer = GeoDatasetWriter(root="./output", data_type="float32")
    >>> with writer:
    ...     writer.write(0, signal)
    >>>
    >>> # Read back
    >>> reader = GeoDatasetReader(root="./output")
    >>> signal = reader.read(0)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.base_handler import BaseFileHandler, FileReader, FileWriter
from torchsig.utils.yaml import custom_representer

# Default field mapping: remap rx_* fields to simpler names
DEFAULT_FIELD_MAPPING: dict[str, str] = {
    "rx_id": "id",
    "rx_lat": "lat",
    "rx_lon": "lon",
    "rx_alt": "alt",
    "rx_vel_east": "vel_east",
    "rx_vel_north": "vel_north",
    "rx_vel_up": "vel_up",
}

# Mapping from human-readable data type names to numpy dtypes
DATA_TYPE_TO_DTYPE = {
    "short": np.dtype(np.int16),
    "int16": np.dtype(np.int16),
    "int": np.dtype(np.int32),
    "int32": np.dtype(np.int32),
    "float": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "float64": np.dtype(np.float64),
}

# Integer scaling factors: map float [-1, 1] to full integer range
INT_SCALING_FACTOR = {
    np.dtype(np.int16): np.iinfo(np.int16).max,
    np.dtype(np.int32): np.iinfo(np.int32).max,
}

__all__ = ["GeoDatasetFileHandler", "GeoDatasetReader", "GeoDatasetWriter"]


class GeoDatasetWriter(FileWriter):
    """Writes TorchSigGeoDataset signals to .yaml and .dat file pairs.

    Each signal is written as two files:
    - {index}.yaml: YAML metadata file
    - {index}.dat: Binary IQ data file

    The .dat file contains interleaved I/Q samples in the specified data type.
    For integer types (int16, int32), float IQ values in [-1, 1] are scaled
    to the symmetric range [-INT_MAX, INT_MAX].

    Note:
        For integer data types, input IQ values must be in [-1, 1] range.
        Values outside this range will raise a ValueError. Use
        torchsig.transforms.Normalize to normalize data before writing.

    Args:
        root: Directory to write files
        data_type: Data type for .dat file ('float32', 'short', 'int16', etc.)
        field_mapping: Dict mapping internal metadata keys to output YAML keys.
            Both writer and reader accept the same mapping (internal -> file direction).
            The reader automatically reverses this mapping when reading.
        allowlist: If provided, only metadata keys in this list will be written
        blocklist: If provided, metadata keys in this list will be excluded
        **kwargs: Additional arguments passed to FileWriter

    Example:
        >>> writer = GeoDatasetWriter(root="./output", data_type="float32", field_mapping={"rx_lat": "lat", "rx_lon": "lon"})
        >>> with writer:
        ...     writer.write(0, signal)
    """

    def __init__(
        self,
        root: str,
        data_type: str = "float32",
        field_mapping: dict[str, str] | None = None,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
        **kwargs,
    ):
        file_handler_overwrite = kwargs.pop(
            "file_handler_overwrite",
            None,
        )
        self._overwrite = bool(kwargs.get("overwrite", False) if file_handler_overwrite is None else file_handler_overwrite)

        super().__init__(root=root, **kwargs)

        # Validate and store data type
        data_type_lower = data_type.lower()
        if data_type_lower not in DATA_TYPE_TO_DTYPE:
            raise ValueError(f"Unsupported data_type: {data_type}. Supported types: {list(DATA_TYPE_TO_DTYPE.keys())}")
        self.dtype = DATA_TYPE_TO_DTYPE[data_type_lower]
        self.data_type_str = data_type_lower  # Store for metadata

        # Combine default and user field mappings
        self.field_mapping = {**DEFAULT_FIELD_MAPPING, **(field_mapping or {})}
        mapped_fields = list(self.field_mapping.values())
        duplicate_fields = sorted(field for field in set(mapped_fields) if mapped_fields.count(field) > 1)
        if duplicate_fields:
            raise ValueError(f"Field mapping collision: multiple metadata fields map to {duplicate_fields}")
        self.allowlist = allowlist
        self.blocklist = blocklist
        self._counter = 0

    def __enter__(self) -> GeoDatasetWriter:
        """Enter the writer context after handling an existing dataset."""
        existing_files = [path for pattern in ("*.yaml", "*.dat") for path in self.root.glob(pattern) if path.stem.isdigit()]

        if existing_files and not self._overwrite:
            raise FileExistsError(f"GeoDatasetWriter output directory is not empty: {self.root}")

        if self._overwrite:
            for path in existing_files:
                path.unlink()

        return super().__enter__()

    def _setup(self) -> None:
        """Create output directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _normalize_value(self, value: Any) -> Any:
        """Convert value to YAML-serializable format."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, complex):
            return [value.real, value.imag]
        if hasattr(value, "item"):
            return value.item()
        return value

    def write(self, batch_idx: int, data: Signal | list[Signal]) -> None:  # noqa: ARG002
        """Write a batch of signals to disk.

        Args:
            batch_idx: Index of the batch (unused, for API compatibility)
            data: Single Signal or list of Signals to write
        """
        signals = [data] if isinstance(data, Signal) else data

        for signal in signals:
            # Prepare and validate IQ data before creating either output file.
            signal_data = np.asarray(signal.data)
            if not np.all(np.isfinite(signal_data)):
                raise ValueError("IQ data must contain only finite values")

            # For integer types, scale from float [-1, 1] to the symmetric
            # representable range [-INT_MAX, INT_MAX].
            if self.dtype in INT_SCALING_FACTOR:
                real_max = np.abs(np.real(signal_data)).max(initial=0.0)
                imag_max = np.abs(np.imag(signal_data)).max(initial=0.0)
                if real_max > 1.0 or imag_max > 1.0:
                    raise ValueError(
                        f"Cannot serialize to {self.dtype}: IQ data has values outside [-1, 1] (real max: {real_max:.4f}, imag max: {imag_max:.4f}). Normalize data before writing to integer formats."
                    )
                scale = INT_SCALING_FACTOR[self.dtype]
                real_scaled = np.real(signal_data) * scale
                imag_scaled = np.imag(signal_data) * scale
            else:
                real_scaled = np.real(signal_data)
                imag_scaled = np.imag(signal_data)

            interleaved = np.empty(signal_data.size * 2, dtype=self.dtype)
            interleaved[::2] = real_scaled.astype(self.dtype)
            interleaved[1::2] = imag_scaled.astype(self.dtype)

            # Extract metadata using get_full_metadata() which includes parent metadata
            metadata = {}
            full_metadata = signal.get_full_metadata()
            for key, value in full_metadata.items():
                if value is None:
                    continue
                try:
                    metadata[key] = self._normalize_value(value)
                except (TypeError, ValueError, AttributeError) as e:
                    warnings.warn(
                        f"Skipping metadata key '{key}' with value type {type(value)}: {e}",
                        stacklevel=3,
                    )
                    continue

            # Apply field mapping
            metadata = {self.field_mapping.get(k, k): v for k, v in metadata.items()}

            # Apply filtering
            if self.blocklist:
                metadata = {k: v for k, v in metadata.items() if k not in self.blocklist}
            if self.allowlist:
                metadata = {k: v for k, v in metadata.items() if k in self.allowlist}

            # Add data format metadata
            metadata.update(
                {
                    "data_type": self.data_type_str,
                    "complex_type": True,
                    "item_size": 2 * self.dtype.itemsize,
                    "swapped": False,
                }
            )

            # Determine filenames (use index only for filesystem safety)
            yaml_path = self.root / f"{self._counter}.yaml"
            dat_path = self.root / f"{self._counter}.dat"

            # Both outputs are created only after serialization has succeeded.
            with Path(dat_path).open("wb") as f:
                f.write(interleaved.tobytes())

            yaml.add_representer(list, custom_representer)
            with Path(yaml_path).open("w") as f:
                yaml.dump(
                    metadata,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    width=200,
                )

            self._counter += 1

    def __len__(self) -> int:
        """Return the number of samples written."""
        return self._counter


class GeoDatasetReader(FileReader):
    """Reads .yaml and .dat file pairs into Signal objects.

    The files are expected to be in the format written by GeoDatasetWriter:
    - {index}.yaml: YAML metadata file
    - {index}.dat: Binary IQ data file with interleaved I/Q samples

    For integer types (int16, int32), IQ samples are scaled back from the
    integer range to float [-1, 1].

    Args:
        root: Directory containing .yaml and .dat files
        field_mapping: Dict mapping internal metadata keys to file keys.
            Same direction as GeoDatasetWriter (internal -> file);
            the reader automatically reverses this mapping.
        **kwargs: Additional arguments passed to FileReader

    Example:
        >>> reader = GeoDatasetReader(root="./output")
        >>> signal = reader.read(0)
        >>> print(signal["lat"], signal["lon"])
    """

    def __init__(
        self,
        root: str,
        field_mapping: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(root=root, **kwargs)
        # Build reverse mapping: file key -> Signal key
        # Both default and user-provided mappings are internal -> file, so reverse them
        combined_mapping = {**DEFAULT_FIELD_MAPPING, **(field_mapping or {})}
        mapped_fields = list(combined_mapping.values())
        duplicate_fields = sorted(field for field in set(mapped_fields) if mapped_fields.count(field) > 1)
        if duplicate_fields:
            raise ValueError(f"Field mapping collision: multiple metadata fields map to {duplicate_fields}")
        self._reverse_map = {v: k for k, v in combined_mapping.items()}
        self._files: dict[int, tuple[Path, Path]] | None = None

    def _build_index_map(self) -> dict[int, tuple[Path, Path]]:
        """Build mapping from index to (yaml_path, dat_path)."""
        if self._files is not None:
            return self._files

        yaml_files = {int(path.stem): path for path in self.root.glob("*.yaml") if path.stem.isdigit()}
        dat_files = {int(path.stem): path for path in self.root.glob("*.dat") if path.stem.isdigit()}

        indexes = sorted(yaml_files.keys() & dat_files.keys())
        if indexes != list(range(len(indexes))):
            raise ValueError(f"GeoDataset file indexes must be contiguous and start at 0; found {indexes}")

        index_map = {idx: (yaml_files[idx], dat_files[idx]) for idx in indexes}

        self._files = index_map
        return self._files

    def read(self, idx: int) -> Signal:
        """Read a signal from disk.

        Args:
            idx: Index of the signal to read

        Returns:
            Signal object with data and metadata

        Raises:
            IndexError: If index is not found
        """
        index_map = self._build_index_map()

        if idx not in index_map:
            raise IndexError(f"Index {idx} not found in {self.root}")

        yaml_path, dat_path = index_map[idx]

        # Load YAML metadata
        with Path(yaml_path).open() as f:
            metadata = yaml.safe_load(f) or {}

        # Apply reverse field mapping
        metadata = {self._reverse_map.get(k, k): v for k, v in metadata.items()}

        # Load DAT file
        with Path(dat_path).open("rb") as f:
            raw_data = f.read()

        # Get dtype from metadata, reject unknown types
        file_data_type = metadata.get("data_type", "float32")
        dtype = DATA_TYPE_TO_DTYPE.get(file_data_type.lower())
        if dtype is None:
            raise ValueError(f"Unsupported data_type in file: {file_data_type}. Supported types: {list(DATA_TYPE_TO_DTYPE.keys())}")

        # Convert to numpy array and then to complex
        interleaved = np.frombuffer(raw_data, dtype=dtype)
        if interleaved.size % 2:
            raise ValueError(f"{dat_path.name} must contain an even number of values forming complete I/Q pairs")

        # Scale back from integer range if needed
        if dtype in INT_SCALING_FACTOR:
            scale = INT_SCALING_FACTOR[dtype]
            real = interleaved[::2].astype(np.float64) / scale
            imag = interleaved[1::2].astype(np.float64) / scale
            data = real + 1j * imag
        else:
            data = interleaved[::2] + 1j * interleaved[1::2]

        return Signal(data=data, **metadata)

    def __len__(self) -> int:
        """Return the number of signals in the dataset."""
        yaml_files = {int(path.stem): path for path in self.root.glob("*.yaml") if path.stem.isdigit()}
        dat_files = {int(path.stem): path for path in self.root.glob("*.dat") if path.stem.isdigit()}

        missing_dat = sorted(yaml_files.keys() - dat_files.keys())
        if missing_dat:
            idx = missing_dat[0]
            raise ValueError(f"{yaml_files[idx].name} has no matching {idx}.dat file")

        missing_yaml = sorted(dat_files.keys() - yaml_files.keys())
        if missing_yaml:
            idx = missing_yaml[0]
            raise ValueError(f"{dat_files[idx].name} has no matching {idx}.yaml file")

        return len(self._build_index_map())


class GeoDatasetFileHandler(BaseFileHandler):
    """Factory class for creating GeoDataset readers and writers.

    Example:
        >>> # Create a writer
        >>> handler = GeoDatasetFileHandler.create_handler("w", "./output", data_type="float32")
        >>>
        >>> # Create a reader
        >>> handler = GeoDatasetFileHandler.create_handler("r", "./output")
    """

    reader_class: type[FileReader] = GeoDatasetReader
    writer_class: type[FileWriter] = GeoDatasetWriter

    @staticmethod
    def create_handler(mode: str, root: str, **kwargs) -> GeoDatasetWriter | GeoDatasetReader:
        """Creates FileWriter or FileReader.

        Args:
            mode: read or write mode ("r" or "w")
            root: where file handler will be running

        Raises:
            ValueError: invalid mode

        Returns:
            FileWriter | FileReader: FileHandler's reader or writer.
        """
        if mode == "r":
            return GeoDatasetFileHandler.reader_class(root, **kwargs)
        if mode == "w":
            return GeoDatasetFileHandler.writer_class(root, **kwargs)
        raise ValueError(f"Invalid File Handler mode: {mode}")
