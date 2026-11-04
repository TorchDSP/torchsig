"""PyTorch integration for materialized structured HDF5 datasets."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchsig.utils.file_handlers import StructuredHDF5Reader, StructuredHDF5Writer, infer_structured_sample_schema

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["StructuredHDF5Dataset", "materialize_structured_dataset"]

ValidationMode = Literal["full", "sampled"]


def _identity_collate(samples: list[Any]) -> list[Any]:
    """Return samples unchanged so their container types remain intact."""
    return samples


def _leading_sizes(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        return [size for item in value.values() for size in _leading_sizes(item)]
    if isinstance(value, (tuple, list)):
        return [size for item in value for size in _leading_sizes(item)]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError("A collated tensor leaf must have a batch dimension")
        return [len(value)]
    if isinstance(value, (np.ndarray, np.generic, int, float, complex, bool)):
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError("A collated array leaf must have a batch dimension")
        return [len(array)]
    raise TypeError(f"Unsupported collated batch leaf: {type(value).__name__}")


def _sample_at(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _sample_at(item, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_sample_at(item, index) for item in value)
    if isinstance(value, list):
        return [_sample_at(item, index) for item in value]
    return value[index]


def _unbatch_collated(value: Any) -> list[Any]:
    """Split a nested collated batch using numeric leaves, not field count."""
    sizes = _leading_sizes(value)
    if not sizes:
        raise ValueError("A collated batch must contain at least one numeric leaf")
    if len(set(sizes)) != 1:
        raise ValueError(f"Collated batch leaves have inconsistent batch sizes: {sizes}")
    return [_sample_at(value, index) for index in range(sizes[0])]


def _sample_list(value: Any) -> list[Any] | None:
    """Recognize an identity-collated list of schema-compatible samples."""
    if not isinstance(value, list) or not value:
        return None
    try:
        schema = infer_structured_sample_schema(value[0])
        for sample in value[1:]:
            schema.validate(sample)
    except (TypeError, ValueError):
        return None
    return value


class StructuredHDF5Dataset(Dataset):
    """PyTorch map-style dataset backed by structured homogeneous HDF5."""

    def __init__(self, root: str | PathLike[str]) -> None:
        """Open the structured dataset stored beneath ``root``."""
        self.root = root
        self.reader = StructuredHDF5Reader(root)

    def __len__(self) -> int:
        """Return the number of materialized samples."""
        return len(self.reader)

    def __getitem__(self, index: int) -> Any:
        """Read one structured sample."""
        return self.reader.read(index)

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Read an index batch while preserving sampler order and duplicates."""
        return self.reader.read_indices(indices)

    def close(self) -> None:
        """Close the underlying HDF5 reader."""
        self.reader.close()


def _source_length(source: Dataset | DataLoader, expected_length: int | None) -> int:
    if expected_length is not None:
        if not isinstance(expected_length, int) or isinstance(expected_length, bool):
            raise TypeError("expected_length must be an integer")
        if expected_length < 0:
            raise ValueError("expected_length must be non-negative")
        return expected_length
    dataset = source.dataset if isinstance(source, DataLoader) else source
    try:
        return len(dataset)
    except (TypeError, NotImplementedError) as error:
        raise ValueError("expected_length is required when the source dataset has no length") from error


def _remove_path(path: Path) -> None:
    """Remove a file or directory used during atomic publication."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _destination_conflicts(path: Path) -> bool:
    """Return whether a path contains an existing destination to preserve."""
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def _publish_structured_dataset(staging_root: Path, root: Path, overwrite: bool) -> None:
    """Publish a validated staging directory while preserving prior output."""
    if _destination_conflicts(root) and not overwrite:
        raise FileExistsError(f"Structured dataset destination already exists: {root}")
    if not root.exists():
        staging_root.replace(root)
        return

    backup_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.", suffix=".backup", dir=root.parent))
    backup_root.rmdir()
    root.replace(backup_root)
    try:
        staging_root.replace(root)
    except Exception:
        backup_root.replace(root)
        raise
    _remove_path(backup_root)


def _validation_indices(length: int, mode: ValidationMode | None, sample_count: int, seed: int) -> range | frozenset[int]:
    """Return the source indices whose materialized values must be checked."""
    if mode is None:
        return range(0)
    if mode == "full":
        return range(length)
    count = min(sample_count, length)
    generator = np.random.default_rng(seed)
    return frozenset(int(index) for index in generator.choice(length, size=count, replace=False))


def _copy_validation_sample(value: Any) -> Any:
    """Copy one structured sample without retaining tensor or array storage."""
    if isinstance(value, Mapping):
        return {key: _copy_validation_sample(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_validation_sample(item) for item in value)
    if isinstance(value, list):
        return [_copy_validation_sample(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _format_validation_path(path: tuple[str | int, ...]) -> str:
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f"[{part!r}]"
    return result


def _validate_materialized_samples(
    reader: StructuredHDF5Reader,
    expected: Mapping[int, Any],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare captured source samples with their staged representations."""
    for index in sorted(expected):
        actual = reader.read(index)
        try:
            expected_leaves = reader.schema.validate(expected[index])
            actual_leaves = reader.schema.validate(actual)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Structured validation failed at sample {index}: {error}") from error
        for field, expected_value, actual_value in zip(reader.schema.fields, expected_leaves, actual_leaves, strict=True):
            equal = np.array_equal(expected_value, actual_value, equal_nan=True) if rtol == 0.0 and atol == 0.0 else np.allclose(expected_value, actual_value, rtol=rtol, atol=atol, equal_nan=True)
            if not equal:
                path = _format_validation_path(field.path)
                raise ValueError(f"Structured validation failed at sample {index}, field {path}: values differ")


def _record_validation_metadata(
    root: Path,
    mode: ValidationMode | None,
    count: int,
    seed: int,
    rtol: float,
    atol: float,
) -> None:
    """Record the completed validation configuration and result."""
    with h5py.File(root / "data.h5", "r+") as handle:
        handle.attrs["validation_mode"] = mode or "none"
        handle.attrs["validation_result"] = "passed" if mode is not None else "not_requested"
        handle.attrs["validation_sample_count"] = count
        handle.attrs["validation_seed"] = seed
        handle.attrs["validation_rtol"] = rtol
        handle.attrs["validation_atol"] = atol
        handle.flush()


def materialize_structured_dataset(
    source: Dataset | DataLoader,
    root: str | PathLike[str],
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    expected_length: int | None = None,
    progress: bool = True,
    overwrite: bool = False,
    validation: ValidationMode | None = None,
    validation_samples: int = 100,
    validation_seed: int = 0,
    validation_rtol: float = 0.0,
    validation_atol: float = 0.0,
    writer_kwargs: Mapping[str, Any] | None = None,
) -> StructuredHDF5Dataset:
    """Materialize a PyTorch Dataset or DataLoader into structured HDF5.

    Dataset sources are loaded with an identity collator by default, preserving
    each sample's tuple/list/mapping structure. If ``collate_fn`` is supplied,
    or if ``source`` is already a DataLoader, collated numeric leaves are split
    along their leading batch dimension before writing.

    Args:
        source: Map-style or iterable Dataset, or an existing DataLoader.
        root: Output directory for the structured HDF5 dataset.
        batch_size: Batch size used when constructing a DataLoader for a Dataset.
        num_workers: Worker count used when constructing that DataLoader.
        collate_fn: Optional collator used only for a Dataset source.
        expected_length: Required count, inferred from ``len(dataset)`` when possible.
        progress: Whether to display materialization progress.
        overwrite: Whether to atomically replace an existing destination after
            the new dataset has been completed and validated.
        validation: Optional ``"full"`` or deterministic ``"sampled"``
            source/output validation.
        validation_samples: Maximum number of samples checked in sampled mode.
        validation_seed: Seed controlling deterministic sampled indices.
        validation_rtol: Relative tolerance for value comparisons.
        validation_atol: Absolute tolerance for value comparisons.
        writer_kwargs: Optional keyword arguments for StructuredHDF5Writer.

    Returns:
        An open map-style dataset for the completed output.
    """
    if not isinstance(source, (Dataset, DataLoader)):
        raise TypeError("source must be a PyTorch Dataset or DataLoader")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if validation not in (None, "full", "sampled"):
        raise ValueError("validation must be None, 'full', or 'sampled'")
    if not isinstance(validation_samples, int) or isinstance(validation_samples, bool) or validation_samples < 1:
        raise ValueError("validation_samples must be a positive integer")
    if not isinstance(validation_seed, int) or isinstance(validation_seed, bool):
        raise TypeError("validation_seed must be an integer")
    for name, value in (("validation_rtol", validation_rtol), ("validation_atol", validation_atol)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    destination = Path(root).resolve()
    if _destination_conflicts(destination) and not overwrite:
        raise FileExistsError(f"Structured dataset destination already exists: {destination}")
    if isinstance(source, DataLoader):
        if collate_fn is not None:
            raise ValueError("collate_fn cannot replace the collator of an existing DataLoader")
        loader = source
        batches_are_sample_lists = False
    else:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        batches_are_sample_lists = collate_fn is None
        loader = DataLoader(
            source,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_identity_collate if collate_fn is None else collate_fn,
        )

    required_count = _source_length(source, expected_length)
    if required_count == 0:
        raise ValueError("Cannot infer a structured schema from an empty source")
    kwargs = dict(writer_kwargs or {})
    written = 0
    selected_indices = _validation_indices(required_count, validation, validation_samples, validation_seed)
    validation_expected: dict[int, Any] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    try:
        with (
            StructuredHDF5Writer(staging_root, **kwargs) as writer,
            tqdm(total=required_count, desc="Materializing structured dataset", disable=not progress, unit="sample") as progress_bar,
        ):
            for batch_index, batch in enumerate(loader):
                if getattr(loader, "batch_size", None) is None:
                    samples = [batch]
                elif batches_are_sample_lists:
                    samples = batch
                else:
                    samples = _sample_list(batch) or _unbatch_collated(batch)
                if not samples:
                    raise ValueError(f"Source produced an empty batch at index {batch_index}")
                if written + len(samples) > required_count:
                    raise RuntimeError(f"Source produced more than the expected {required_count} samples")
                for offset, sample in enumerate(samples):
                    sample_index = written + offset
                    if sample_index in selected_indices:
                        validation_expected[sample_index] = _copy_validation_sample(sample)
                writer.write(batch_index, samples)
                written += len(samples)
                progress_bar.update(len(samples))

            if written != required_count:
                raise RuntimeError(f"Source produced {written} samples; expected {required_count}")

        with StructuredHDF5Reader(staging_root) as staged_reader:
            len(staged_reader)
            if validation is not None:
                _validate_materialized_samples(
                    staged_reader,
                    validation_expected,
                    rtol=float(validation_rtol),
                    atol=float(validation_atol),
                )
        _record_validation_metadata(
            staging_root,
            validation,
            len(validation_expected),
            validation_seed,
            float(validation_rtol),
            float(validation_atol),
        )
        _publish_structured_dataset(staging_root, destination, overwrite)
    finally:
        if staging_root.exists():
            _remove_path(staging_root)

    return StructuredHDF5Dataset(destination)
