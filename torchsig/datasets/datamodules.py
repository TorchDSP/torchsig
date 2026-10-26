"""PyTorch Lightning DataModules
Learn More: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
If dataset does not exist at root, creates new dataset and writes to disk
If dataset does exist, simply loaded it back in
"""

from __future__ import annotations

import inspect
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, Subset, get_worker_info, random_split
from torch.utils.data._utils.collate import default_collate

from torchsig.datasets.datasets import (
    StaticTorchSigDataset,
    TorchSigDatasetConfig,
    TorchSigIterableDataset,
)
from torchsig.datasets.structured import StructuredHDF5Dataset, materialize_structured_dataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import GroupingLabel, YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.file_handlers.packed_hdf5 import (
    PackedHDF5Reader,
    PackedHDF5Writer,
)
from torchsig.utils.file_handlers.packed_npy import PackedNPYReader, PackedNPYWriter
from torchsig.utils.random import Seedable
from torchsig.utils.writer import DatasetCreator, identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml

if TYPE_CHECKING:
    from torchsig.utils.file_handlers import BaseFileHandler

__all__ = [
    "SplitTorchSigDataModule",
    "TorchSigDataModule",
    "set_global_seed",
]

# --------------------------------------------------------------
#  GLOBAL REPRODUCIBILITY HELPERS
# --------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set *all* relevant RNGs to the same seed."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seed the global RNG used by downstream code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms (fails loudly if an op is nondet.)
    torch.use_deterministic_algorithms(mode=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------
#  DATA MODULE
# --------------------------------------------------------------


def _resolve_file_reader(file_writer, file_reader):
    """Infer or validate known HDF5 writer/reader format pairs.

    Homogeneous HDF5 is opt-in and requires fixed-shape, fixed-dtype top-level
    samples. Component counts and component arrays may remain variable.
    """
    known_pairs = {
        HDF5Writer: HDF5Reader,
        PackedHDF5Writer: PackedHDF5Reader,
        HomogeneousHDF5Writer: HomogeneousHDF5Reader,
        PackedNPYWriter: PackedNPYReader,
    }
    expected_reader = known_pairs.get(file_writer)
    if file_reader is None:
        return expected_reader or HDF5Reader
    if expected_reader is not None and file_reader is not expected_reader:
        raise ValueError(f"Incompatible file handler pair: {file_writer.__name__} requires {expected_reader.__name__}, got {file_reader.__name__}")
    return file_reader


def _validate_file_writer_kwargs(file_writer, options):
    """Copy and validate keyword arguments accepted by a writer class."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError("file_writer_kwargs must be a dictionary")
    result = dict(options)
    try:
        inspect.signature(file_writer).bind_partial(root=Path(), **result)
    except ValueError:
        return result
    except TypeError as error:
        raise TypeError(f"Invalid options for {file_writer.__name__}: {error}") from error
    return result


def _load_dataset_config(
    cfg: TorchSigDatasetConfig | str | Path,
) -> TorchSigDatasetConfig:
    """Load a TorchSig dataset config when given a YAML path."""
    if isinstance(cfg, TorchSigDatasetConfig):
        return cfg

    return load_config_from_yaml(Path(cfg))


def _dataset_metadata(cfg: TorchSigDatasetConfig) -> dict[str, Any]:
    """Merge TorchSig default metadata with config-specific metadata."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(cfg.dataset_metadata)
    return metadata


def _config_transforms(cfg: TorchSigDatasetConfig) -> list[Any]:
    """Construct output transforms for a dataset config."""
    output_representation = cfg.output_representation.lower()

    if output_representation == "iq":
        return [ComplexTo2D()]

    if output_representation == "spectrogram":
        fft_size = getattr(cfg, "output_spectrogram_fft", None)

        if fft_size is None:
            fft_size = getattr(cfg, "fft_size", None)

        if fft_size is None:
            fft_size = cfg.dataset_metadata.get("fft_size", 256)

        return [
            Spectrogram(fft_size=int(fft_size)),
            YOLOLabel(),
        ]

    fft_size = getattr(cfg, "fft_size", 256)
    return [Spectrogram(fft_size=int(fft_size))]


def _metadata_debug_options(
    metadata_debug: bool | Mapping[str, Any],
) -> dict[str, Any] | None:
    """Normalize DataModule metadata-debug configuration.

    Args:
        metadata_debug: ``False`` to disable debugging, ``True`` to use the
            default metadata-debug configuration, or a mapping of keyword
            arguments accepted by ``enable_metadata_debug``.

    Returns:
        Debug keyword arguments, or ``None`` when debugging is disabled.

    Raises:
        TypeError: If ``metadata_debug`` is not a boolean or mapping.
    """
    if isinstance(metadata_debug, bool):
        return {} if metadata_debug else None
    if isinstance(metadata_debug, Mapping):
        return dict(metadata_debug)
    raise TypeError("metadata_debug must be a boolean or mapping")


def _enable_dataset_metadata_debug(
    dataset: TorchSigIterableDataset,
    options: dict[str, Any] | None,
) -> None:
    """Enable metadata debugging on an iterable dataset when requested."""
    if options is not None:
        dataset.enable_metadata_debug(**options)


def _qualified_name(value: type) -> str:
    """Return a stable import-style name for a configured class."""
    return f"{value.__module__}.{value.__qualname__}"


def _source_fingerprint(root: Path) -> dict[str, int]:
    """Fingerprint the source HDF5 file used for materialization reuse."""
    stat = (root / "data.h5").stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _manifest_path(root: Path) -> Path:
    """Keep compatibility metadata outside the atomically published dataset."""
    return root.with_name(f"{root.name}.manifest.json")


def _json_safe(value: Any) -> Any:
    """Normalize configuration values into deterministic JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically publish a structured-materialization manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    temporary.replace(path)


class _OnlineTransformDataset(Dataset, Seedable):
    """Apply requested epoch-varying transforms after materialized reads."""

    def __init__(self, dataset: Dataset, transforms: list[Callable], seed: int) -> None:
        Dataset.__init__(self)
        Seedable.__init__(self, seed=seed)
        self.dataset = dataset
        self.transforms = list(transforms)
        for transform in self.transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)

    def __len__(self) -> int:
        return len(self.dataset)

    def _transform(self, sample: Any) -> Any:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, index: int) -> Any:
        return self._transform(self.dataset[index])

    def __getitems__(self, indices: list[int]) -> list[Any]:
        getitems = getattr(self.dataset, "__getitems__", None)
        samples = getitems(indices) if getitems is not None else [self.dataset[index] for index in indices]
        return [self._transform(sample) for sample in samples]


class _StructuredMaterializationMixin:
    """Shared explicit structured-materialization configuration and lifecycle."""

    structured_materialization: bool
    structured_root: Path
    structured_overwrite: bool
    structured_batch_size: int
    structured_num_workers: int
    structured_validation: str | None
    structured_validation_samples: int
    structured_writer_kwargs: dict[str, Any]

    def _configure_structured_materialization(
        self,
        *,
        source_root: Path,
        enabled: bool,
        root: str | Path | None,
        overwrite: bool,
        batch_size: int | None,
        num_workers: int,
        validation: str | None,
        validation_samples: int,
        writer_kwargs: Mapping[str, Any] | None,
        train_transforms: list[Callable] | None,
        default_batch_size: int,
    ) -> None:
        if not isinstance(enabled, bool) or not isinstance(overwrite, bool):
            raise TypeError("structured_materialization and structured_overwrite must be booleans")
        if validation not in (None, "full", "sampled"):
            raise ValueError("structured_validation must be None, 'full', or 'sampled'")
        resolved_batch_size = default_batch_size if batch_size is None else batch_size
        if not isinstance(resolved_batch_size, int) or isinstance(resolved_batch_size, bool) or resolved_batch_size < 1:
            raise ValueError("structured_batch_size must be a positive integer")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ValueError("structured_num_workers must be a non-negative integer")
        if not isinstance(validation_samples, int) or isinstance(validation_samples, bool) or validation_samples < 1:
            raise ValueError("structured_validation_samples must be a positive integer")
        if writer_kwargs is not None and not isinstance(writer_kwargs, Mapping):
            raise TypeError("structured_writer_kwargs must be a mapping")
        self.structured_materialization = enabled
        self.structured_root = Path(root) if root is not None else source_root.with_name(f"{source_root.name}-structured")
        if self.structured_root.resolve() == source_root.resolve():
            raise ValueError("structured_root must be separate from the source dataset root")
        self.structured_overwrite = overwrite
        self.structured_batch_size = resolved_batch_size
        self.structured_num_workers = num_workers
        self.structured_validation = validation
        self.structured_validation_samples = validation_samples
        self.structured_writer_kwargs = dict(writer_kwargs or {})
        self.structured_train_transforms = list(train_transforms or [])

    def _structured_manifest(self, source_root: Path, output_root: Path, *, split: str, expected_length: int, target_labels: list[str] | None) -> dict[str, Any]:
        return {
            "version": 1,
            "layout": split,
            "source_root": str(source_root.resolve()),
            "output_root": str(output_root.resolve()),
            "source_fingerprint": _source_fingerprint(source_root),
            "source_reader": _qualified_name(self.file_reader),
            "expected_length": expected_length,
            "target_labels": list(target_labels) if target_labels is not None else None,
            "batch_size": self.structured_batch_size,
            "num_workers": self.structured_num_workers,
            "validation": self.structured_validation,
            "validation_samples": self.structured_validation_samples,
            "writer_kwargs": _json_safe(self.structured_writer_kwargs),
        }

    def _ensure_structured_dataset(self, source: Dataset, output_root: Path, manifest: dict[str, Any]) -> None:
        manifest_path = _manifest_path(output_root)
        compatible = False
        if output_root.exists() or manifest_path.exists():
            try:
                compatible = json.loads(manifest_path.read_text()) == _json_safe(manifest)
                if compatible:
                    stored = StructuredHDF5Dataset(output_root)
                    try:
                        compatible = len(stored) == manifest["expected_length"]
                    finally:
                        stored.close()
            except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
                compatible = False
            if compatible:
                return
            if not self.structured_overwrite:
                raise ValueError(f"Incompatible structured materialization at {output_root}; enable structured_overwrite to replace it")

        materialized = materialize_structured_dataset(
            source,
            output_root,
            batch_size=self.structured_batch_size,
            num_workers=self.structured_num_workers,
            overwrite=self.structured_overwrite or output_root.exists(),
            validation=self.structured_validation,
            validation_samples=self.structured_validation_samples,
            writer_kwargs=self.structured_writer_kwargs,
        )
        materialized.close()
        _write_manifest(manifest_path, manifest)

    def _open_structured_dataset(self, source_root: Path, output_root: Path, *, split: str, expected_length: int, target_labels: list[str] | None) -> StructuredHDF5Dataset:
        expected = self._structured_manifest(source_root, output_root, split=split, expected_length=expected_length, target_labels=target_labels)
        manifest_path = _manifest_path(output_root)
        try:
            actual = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Structured materialization manifest is missing or invalid: {manifest_path}") from error
        if actual != _json_safe(expected):
            raise ValueError(f"Incompatible structured materialization at {output_root}; rerun prepare_data with structured_overwrite enabled")
        dataset = StructuredHDF5Dataset(output_root)
        actual_length = len(dataset)
        if actual_length != expected_length:
            dataset.close()
            raise ValueError(f"Structured materialization at {output_root} has {actual_length} samples; expected {expected_length}")
        return dataset


def _seed_worker(worker_id: int) -> None:
    """Initialise deterministic NumPy / Python RNGs **inside a DataLoader worker**.

    * If the dataset that the worker sees is a ``torch.utils.data.Subset``,
      we unwrap it to reach the underlying concrete dataset (which inherits
      from ``Seedable`` and therefore owns ``random_generator``).
    * The ``worker_id`` is combined with the master seed to give each
      worker a *different* deterministic seed.
    """
    # -----------------------------------------------------------------
    #    Grab the dataset that lives inside the worker.
    # -----------------------------------------------------------------
    parent = get_worker_info().dataset

    # -----------------------------------------------------------------
    #    ``Subset`` is just a thin wrapper -- fetch the real dataset.
    # -----------------------------------------------------------------
    dataset = parent
    while isinstance(dataset, Subset):
        dataset = dataset.dataset

    # -----------------------------------------------------------------
    #    Pull a deterministic integer from the *parent* generator.
    # -----------------------------------------------------------------
    random_generator = getattr(dataset, "random_generator", None)
    master_seed = int(random_generator.integers(0, 2**31)) if random_generator is not None else torch.initial_seed() % (2**31)

    # -----------------------------------------------------------------
    #    Derive a unique seed for THIS worker.
    # -----------------------------------------------------------------
    # Multiplying by (worker_id + 1) guarantees distinct seeds across workers.
    worker_seed = master_seed * (worker_id + 1)

    # -----------------------------------------------------------------
    #    Give the worker its own NumPy Generator (so we never touch the
    #    global legacy RNG).
    # -----------------------------------------------------------------
    dataset.worker_rng = np.random.default_rng(worker_seed)

    # -----------------------------------------------------------------
    #    Seed the std-lib ``random`` module (still needed by some TorchSig code).
    # -----------------------------------------------------------------
    random.seed(worker_seed)


class TorchSigDataModule(_StructuredMaterializationMixin, pl.LightningDataModule):
    """PyTorch Lightning DataModule for creating and loading TorchSig datasets.

    This DataModule handles:
      - Dataset creation or loading from disk via a file handler.
      - Splitting into train/val/test subsets.
      - Batching, collation, and worker seeding for training.

    Attributes:
        root: Directory where datasets are stored or created.
        dataset_size: Total number of samples in the dataset.
        dataset_splits: Fractions or counts for train/val/test splits.
        dataset_metadata: Metadata describing the dataset.
        impairment_level: Optional interference level for synthetic impairments.
        transforms: Transforms applied to the input data.
        target_labels: Names of target metadata fields to include.
        batch_size: Batch size for the training/validation/testing DataLoaders.
        num_workers: Number of worker processes for data loading.
        collate_fn: Custom collate function for batching.
        shuffle: Whether to shuffle the data.
        create_batch_size: Batch size used during on-disk dataset creation.
        create_num_workers: Number of workers used during dataset creation.
        file_writer: FileHandler class for disk I/O.
        file_reader: FileReader class for disk I/O.
        overwrite: If True, existing on-disk data will be overwritten.
        seed: Optional random seed for reproducibility.
        metadata_debug_options: Normalized options used to enable metadata
            debugging during dataset creation, or ``None`` when disabled.
        train: Initialized training dataset (set in `setup()`).
        val: Initialized validation dataset (set in `setup()`).
        test: Initialized test dataset (set in `setup()`).
    """

    def __init__(
        self,
        root: str,
        metadata,
        dataset_size: int,
        *,
        dataset_splits: list[float] | list[int] = [0.70, 0.20, 0.10],
        batch_size: int = 1,
        num_workers: int | None = None,
        collate_fn: Callable | None = None,
        shuffle: bool = True,
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: BaseFileHandler = HDF5Writer,
        file_reader: BaseFileHandler | None = None,
        file_writer_kwargs: dict[str, Any] | None = None,
        overwrite: bool = False,
        impairment_level: int = 0,
        transforms: list | None = None,
        target_labels: list[str] | None = None,
        signal_generators: str | list = "all",
        sampling_grouping: GroupingLabel | str | Path | Mapping[str, Any] | None = None,
        seed: int | None = None,
        metadata_debug: bool | Mapping[str, Any] = False,
        experiment_config: Any = None,
        structured_materialization: bool = False,
        structured_root: str | Path | None = None,
        structured_overwrite: bool = False,
        structured_batch_size: int | None = None,
        structured_num_workers: int = 0,
        structured_validation: str | None = None,
        structured_validation_samples: int = 100,
        structured_writer_kwargs: Mapping[str, Any] | None = None,
        structured_train_transforms: list[Callable] | None = None,
    ):
        """Initialize the TorchSigDataModule.

        Args:
            root: Path to store or load the dataset.
            metadata: Metadata object, YAML file path, or dict describing classes and settings.
            dataset_size: Total number of samples to generate or load.
            dataset_splits: Fractions or counts for train/val/test splits. Defaults to [0.70, 0.20, 0.10].
            batch_size: Batch size for data loaders. Defaults to 1.
            num_workers: Number of worker processes for data loading. Defaults to 1.
            collate_fn: Custom function to collate batch samples. Defaults to None.
            create_batch_size: Batch size when writing data to disk. Defaults to 8.
            create_num_workers: Workers used when creating the on-disk dataset. Defaults to 4.
            file_writer: FileWriter class for disk I/O.
            file_reader: FileReader class for disk I/O.
            file_writer_kwargs: Options passed to the file writer constructor.
            overwrite: If True, existing data at `root` will be overwritten. Defaults to False.
            impairment_level: Level of synthetic impairment to apply. Defaults to 0 (no impairment).
            transforms: List of transforms applied to each sample's input. Defaults to [].
            target_labels: Names of metadata fields to include. Defaults to None.
            signal_generators: Signal generators available during dataset creation.
            sampling_grouping: Optional grouping used to balance generator sampling.
            seed: Seed for randomness and reproducibility. Defaults to None.
            metadata_debug: Enable metadata debugging with default settings,
                or provide keyword arguments for ``enable_metadata_debug``.
                Defaults to ``False``.
            experiment_config: Optional signal-generation configuration, YAML
                path, or mapping. Defaults to no overrides.
            structured_materialization: Opt in to model-ready structured HDF5.
            structured_root: Separate output root; defaults to ``<root>-structured``.
            structured_overwrite: Replace an incompatible structured output.
            structured_batch_size: Batch size for materialization; defaults to
                ``create_batch_size``.
            structured_num_workers: Workers used during materialization.
            structured_validation: Optional ``"full"`` or ``"sampled"`` validation.
            structured_validation_samples: Maximum sampled validation count.
            structured_writer_kwargs: Structured writer compression/chunk options.
            structured_train_transforms: Epoch-varying transforms applied only
                after reading materialized training samples.

        Raises:
            ValueError: If dataset_splits don't sum to 1.0 (when using fractions).
            FileNotFoundError: If metadata file path is invalid.
        """
        super().__init__()

        # ---- filesystem -------------------------------------------------
        self.root = Path(root)
        self.dataset_size = dataset_size
        self.dataset_splits = dataset_splits

        # ---- meta / transforms -------------------------------------------
        self.metadata = metadata
        self.impairment_level = impairment_level
        impairments = Impairments(level=impairment_level)
        self.burst_impairments = impairments.signal_transforms
        self.whole_signal_impairments = impairments.dataset_transforms
        self.transforms = [self.whole_signal_impairments, *(transforms or [])]

        self.target_labels = target_labels
        self.signal_generators = signal_generators
        self.sampling_grouping = sampling_grouping
        self.metadata_debug_options = _metadata_debug_options(metadata_debug)
        self.experiment_config = experiment_config

        # ---- dataloader configuration ------------------------------------
        self.batch_size = batch_size
        self.num_workers = 0 if num_workers is None else num_workers
        self.collate_fn = collate_fn or default_collate
        self.shuffle = shuffle

        # ---- dataset-creation configuration -------------------------------
        self.create_batch_size = create_batch_size
        self.create_num_workers = create_num_workers
        self.file_writer = file_writer
        self.file_reader = _resolve_file_reader(file_writer, file_reader)
        self.file_writer_kwargs = _validate_file_writer_kwargs(file_writer, file_writer_kwargs)
        self.overwrite = overwrite
        self._configure_structured_materialization(
            source_root=self.root,
            enabled=structured_materialization,
            root=structured_root,
            overwrite=structured_overwrite,
            batch_size=structured_batch_size,
            num_workers=structured_num_workers,
            validation=structured_validation,
            validation_samples=structured_validation_samples,
            writer_kwargs=structured_writer_kwargs,
            train_transforms=structured_train_transforms,
            default_batch_size=create_batch_size,
        )

        # ---- placeholders ------------------------------------------------
        self.train: Dataset | None = None
        self.val: Dataset | None = None
        self.test: Dataset | None = None

        # ---- reproducibility ---------------------------------------------
        self.seed = seed if seed is not None else 42

    @classmethod
    def from_config(
        cls,
        cfg: TorchSigDatasetConfig | str | Path,
        root: str | Path,
        *,
        dataset_size: int | None = None,
        dataset_splits: list[float] | list[int] = [0.70, 0.20, 0.10],
        batch_size: int = 1,
        num_workers: int | None = None,
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: type[BaseFileHandler] = HDF5Writer,
        file_reader: type[BaseFileHandler] | None = None,
        file_writer_kwargs: dict[str, Any] | None = None,
        overwrite: bool = False,
        shuffle: bool = True,
        collate_fn: Callable | None = None,
        target_labels: list[str] | None = None,
        signal_generators: str | list = "all",
        sampling_grouping: GroupingLabel | str | Path | Mapping[str, Any] | None = None,
        **kwargs,
    ) -> TorchSigDataModule:
        """Create a TorchSigDataModule from a TorchSigDatasetConfig or YAML path.

        Args:
            cfg: Either a TorchSigDatasetConfig instance or path (str/Path) to a YAML config file
            root: Directory where datasets are stored or created
            dataset_size: Optional override for the dataset size (default: None → uses cfg.dataset_length)
            dataset_splits: Fractions or counts for train/val/test splits (default: [0.70, 0.20, 0.10])
            batch_size: Batch size for data loaders (default: 1)
            num_workers: Number of worker processes for data loading (default: None)
            create_batch_size: Batch size when writing data to disk (default: 8)
            create_num_workers: Workers used when creating dataset (default: 4)
            file_writer: File writer class for disk I/O (default: HDF5Writer)
            file_reader: File reader class for disk I/O (default: HDF5Reader)
            file_writer_kwargs: Options passed to the file writer constructor
            overwrite: Whether to overwrite existing data (default: False)
            shuffle: Whether to shuffle training data (default: True)
            collate_fn: Custom collate function for batching (default: None → identity function)
            target_labels: List of target label names (default: None → auto-select based on output_representation)
            signal_generators: Signal generators available during dataset creation.
            sampling_grouping: Optional grouping used to balance generator sampling.
            **kwargs: Additional arguments passed to TorchSigDataModule constructor

        Returns:
            Configured TorchSigDataModule instance ready for training

        Raises:
            ValueError: If required parameters for spectrogram output are missing
        """
        # Convert path to config if needed
        if isinstance(cfg, (str, Path)):
            cfg = load_config_from_yaml(Path(cfg))

        # Use provided dataset_size if given, otherwise fall back to config
        final_dataset_size = dataset_size if dataset_size is not None else cfg.dataset_length

        # Merge default metadata with custom metadata from config
        base_metadata = TorchSigDefaults().default_dataset_metadata
        dataset_metadata = {**base_metadata, **cfg.dataset_metadata}

        use_default_target_labels = False

        # Configure output-specific transforms
        additional_transforms: list = []
        if target_labels is None:
            use_default_target_labels = True
            target_labels = []

        if cfg.output_representation == "spectrogram":
            fft_size = cfg.output_spectrogram_fft or dataset_metadata.get("fft_size")
            if fft_size is None:
                raise ValueError("For spectrogram output, either `output_spectrogram_fft` must be set in the config or `fft_size` must be present in dataset_metadata")
            additional_transforms.append(Spectrogram(fft_size=int(fft_size)))
            # Only add YOLOLabel if explicitly requested via target_labels
            if "yolo_label" in target_labels or use_default_target_labels:
                additional_transforms.append(YOLOLabel())
                target_labels = list({*target_labels, "yolo_label"})  # Avoid duplicates
        elif cfg.output_representation == "iq":
            additional_transforms.append(ComplexTo2D())

        return cls(
            root=root,
            metadata=dataset_metadata,
            dataset_size=final_dataset_size,
            dataset_splits=dataset_splits,
            batch_size=batch_size,
            num_workers=num_workers,
            create_batch_size=create_batch_size,
            create_num_workers=create_num_workers,
            file_writer=file_writer,
            file_reader=file_reader,
            file_writer_kwargs=file_writer_kwargs,
            overwrite=overwrite,
            shuffle=shuffle,
            collate_fn=collate_fn,
            impairment_level=cfg.impairment_level,
            transforms=additional_transforms,
            target_labels=target_labels or None,
            signal_generators=signal_generators,
            sampling_grouping=sampling_grouping,
            seed=cfg.seed,
            experiment_config=cfg.experiment_config,
            **kwargs,
        )

    def prepare_data(self) -> None:
        """Prepares the dataset by creating new datasets if they do not exist on disk.

        The datasets are created using the `DatasetCreator` class.
        If the dataset already exists on disk, it is loaded back into memory.

        Raises:
            FileNotFoundError: If the root directory cannot be created.
            RuntimeError: If dataset creation fails.
        """
        dataset = TorchSigIterableDataset(
            metadata=self.metadata,
            signal_generators=self.signal_generators,
            sampling_grouping=self.sampling_grouping,
            transforms=self.transforms,
            component_transforms=[self.burst_impairments],
            target_labels=self.target_labels,
            seed=self.seed,
            experiment_config=self.experiment_config,
        )
        _enable_dataset_metadata_debug(dataset, self.metadata_debug_options)
        loader = WorkerSeedingDataLoader(
            dataset=dataset,
            batch_size=self.create_batch_size,
            collate_fn=self.collate_fn,
            seed=self.seed,
        )
        creator = DatasetCreator(
            dataloader=loader,
            dataset_length=self.dataset_size,
            root=self.root,
            overwrite=self.overwrite,
            file_handler=self.file_writer,
            file_reader=self.file_reader,
            **self.file_writer_kwargs,
        )
        print(f"Full Dataset: Impairment Level {self.impairment_level}, {self.dataset_size} samples")
        creator.create()
        if self.structured_materialization:
            source = StaticTorchSigDataset(root=self.root, file_handler_class=self.file_reader, target_labels=self.target_labels)
            try:
                manifest = self._structured_manifest(
                    self.root,
                    self.structured_root,
                    split="full",
                    expected_length=self.dataset_size,
                    target_labels=self.target_labels,
                )
                self._ensure_structured_dataset(source, self.structured_root, manifest)
            finally:
                source.reader.teardown()

    def setup(self, stage: str = "fit") -> None:
        """Sets up the train and validation datasets for the given stage.

        Args:
            stage: The stage of the DataModule, typically 'train' or 'test'. Defaults to 'train'.

        Raises:
            FileNotFoundError: If the dataset files are not found at the specified root.
            ValueError: If dataset splits are invalid.
        """
        _ = stage  # Required by the LightningDataModule hook signature.
        if self.structured_materialization:
            full_dataset: Dataset = self._open_structured_dataset(
                self.root,
                self.structured_root,
                split="full",
                expected_length=self.dataset_size,
                target_labels=self.target_labels,
            )
        else:
            full_dataset = StaticTorchSigDataset(
                root=self.root,
                file_handler_class=self.file_reader,
                target_labels=self.target_labels,
            )
        self.train, self.val, self.test = random_split(
            full_dataset,
            self.dataset_splits,
            generator=Generator().manual_seed(self.seed),
        )
        if self.structured_materialization and self.structured_train_transforms:
            self.train = _OnlineTransformDataset(self.train, self.structured_train_transforms, self.seed)

    # -----------------------------------------------------------------
    # Helper that builds a *deterministic* DataLoader
    # -----------------------------------------------------------------
    def _build_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(self.seed)  # same seed for every epoch

        # ``persistent_workers`` must be a bool; we evaluate it safely.
        persistent = bool(self.num_workers) and (self.num_workers > 0)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=gen,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent,
        )

    # -----------------------------------------------------------------
    #  Lightning-specific hooks
    # -----------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset.

        Returns:
            A PyTorch DataLoader for the training dataset.

        Raises:
            RuntimeError: If the training dataset is not initialized.
        """
        return self._build_dataloader(self.train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset.

        Returns:
            A PyTorch DataLoader for the validation dataset.

        Raises:
            RuntimeError: If the validation dataset is not initialized.
        """
        return self._build_dataloader(self.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset.

        Returns:
            A PyTorch DataLoader for the test dataset.

        Raises:
            RuntimeError: If the test dataset is not initialized.
        """
        return self._build_dataloader(self.test, shuffle=False)


class SplitTorchSigDataModule(_StructuredMaterializationMixin, pl.LightningDataModule):
    """Lightning DataModule for independently generated TorchSig data splits.

    This DataModule creates separate static training, validation, and test
    datasets from three ``TorchSigDatasetConfig`` objects. Unlike
    ``TorchSigDataModule``, it does not create one dataset and partition it
    with ``random_split``.

    The generated directory layout is::

        root/
            dataset_id/
                train/
                val/
                test/

    Args:
        train_cfg: Training dataset config or path to a YAML config.
        val_cfg: Validation dataset config or path to a YAML config.
        test_cfg: Test dataset config or path to a YAML config.
        root: Parent directory in which datasets are stored.
        signal_generators: Signal generators available during dataset creation.
        batch_size: Batch size used by model-facing DataLoaders.
        num_workers: Number of workers used by model-facing DataLoaders.
        create_batch_size: Batch size used while creating static datasets.
        create_num_workers: Number of workers used while creating datasets.
        file_writer: File handler used to write static datasets.
        file_reader: File handler used to read static datasets.
        overwrite: Whether existing datasets should be overwritten.
        shuffle: Whether the training DataLoader should shuffle samples.
        collate_fn: Function used to collate model-facing batches.
        target_labels: Metadata fields returned as targets.
        metadata_debug: Enable metadata debugging with default settings, or
            provide keyword arguments for ``enable_metadata_debug``.
        structured_materialization: Opt in to model-ready structured HDF5.
        structured_root: Separate output root; defaults to a ``-structured``
            sibling of the dataset root.
        structured_overwrite: Replace incompatible structured split outputs.
        structured_batch_size: Batch size used during materialization.
        structured_num_workers: Worker count used during materialization.
        structured_validation: Optional ``"full"`` or ``"sampled"`` validation.
        structured_validation_samples: Maximum sampled validation count.
        structured_writer_kwargs: Structured writer compression/chunk options.
        structured_train_transforms: Epoch-varying transforms applied only to
            materialized training samples.
    """

    def __init__(
        self,
        train_cfg: TorchSigDatasetConfig | str | Path,
        val_cfg: TorchSigDatasetConfig | str | Path,
        test_cfg: TorchSigDatasetConfig | str | Path,
        root: str | Path,
        *,
        signal_generators: str | list[str] = "all",
        batch_size: int = 1,
        num_workers: int | None = None,
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: type[BaseFileHandler] = HDF5Writer,
        file_reader: type[BaseFileHandler] | None = None,
        file_writer_kwargs: dict[str, Any] | None = None,
        overwrite: bool = False,
        shuffle: bool = True,
        collate_fn: Callable | None = None,
        target_labels: list[str] | None = None,
        metadata_debug: bool | Mapping[str, Any] = False,
        structured_materialization: bool = False,
        structured_root: str | Path | None = None,
        structured_overwrite: bool = False,
        structured_batch_size: int | None = None,
        structured_num_workers: int = 0,
        structured_validation: str | None = None,
        structured_validation_samples: int = 100,
        structured_writer_kwargs: Mapping[str, Any] | None = None,
        structured_train_transforms: list[Callable] | None = None,
    ) -> None:
        """Initialize the split-based TorchSig DataModule."""
        super().__init__()

        self.train_cfg = _load_dataset_config(train_cfg)
        self.val_cfg = _load_dataset_config(val_cfg)
        self.test_cfg = _load_dataset_config(test_cfg)

        self.root = Path(root) / self.train_cfg.dataset_id
        self.signal_generators = signal_generators

        self.batch_size = batch_size
        self.num_workers = 0 if num_workers is None else num_workers
        self.create_batch_size = create_batch_size
        self.create_num_workers = create_num_workers

        self.file_writer = file_writer
        self.file_reader = _resolve_file_reader(file_writer, file_reader)
        self.file_writer_kwargs = _validate_file_writer_kwargs(file_writer, file_writer_kwargs)
        self.overwrite = overwrite

        self.shuffle = shuffle
        self.collate_fn = collate_fn or default_collate
        self.target_labels = target_labels or ["class_index"]
        self.metadata_debug_options = _metadata_debug_options(metadata_debug)
        self._configure_structured_materialization(
            source_root=self.root,
            enabled=structured_materialization,
            root=structured_root,
            overwrite=structured_overwrite,
            batch_size=structured_batch_size,
            num_workers=structured_num_workers,
            validation=structured_validation,
            validation_samples=structured_validation_samples,
            writer_kwargs=structured_writer_kwargs,
            train_transforms=structured_train_transforms,
            default_batch_size=create_batch_size,
        )

        self.train: Dataset | None = None
        self.val: Dataset | None = None
        self.test: Dataset | None = None

        self._validate_configs()

    def _validate_configs(self) -> None:
        """Validate assumptions shared by the split configs."""
        train_representation = self.train_cfg.output_representation.lower()

        for split_name, cfg in (
            ("validation", self.val_cfg),
            ("test", self.test_cfg),
        ):
            representation = cfg.output_representation.lower()

            if representation != train_representation:
                raise ValueError(f"All split configs must use the same output representation. Training uses {train_representation!r}, but {split_name} uses {representation!r}.")

    def _create_split(
        self,
        cfg: TorchSigDatasetConfig,
        split: str,
    ) -> None:
        """Create one static dataset split."""
        split_root = self.root / split

        dataset = TorchSigIterableDataset(
            metadata=_dataset_metadata(cfg),
            transforms=_config_transforms(cfg),
            signal_generators=self.signal_generators,
            seed=cfg.seed,
            experiment_config=cfg.experiment_config,
        )
        _enable_dataset_metadata_debug(dataset, self.metadata_debug_options)

        loader = WorkerSeedingDataLoader(
            dataset=dataset,
            batch_size=self.create_batch_size,
            num_workers=self.create_num_workers,
            collate_fn=identity_collate_fn,
            seed=cfg.seed,
        )

        creator = DatasetCreator(
            dataloader=loader,
            dataset_length=int(cfg.dataset_length),
            root=split_root,
            overwrite=self.overwrite,
            file_handler=self.file_writer,
            file_reader=self.file_reader,
            **self.file_writer_kwargs,
        )
        creator.create()
        if self.structured_materialization:
            source = StaticTorchSigDataset(root=split_root, file_handler_class=self.file_reader, target_labels=self.target_labels)
            output_root = self.structured_root / split
            try:
                manifest = self._structured_manifest(
                    split_root,
                    output_root,
                    split=split,
                    expected_length=int(cfg.dataset_length),
                    target_labels=self.target_labels,
                )
                self._ensure_structured_dataset(source, output_root, manifest)
            finally:
                source.reader.teardown()

    def prepare_data(self) -> None:
        """Create independent training, validation, and test datasets."""
        self.root.mkdir(parents=True, exist_ok=True)

        self._create_split(self.train_cfg, "train")
        self._create_split(self.val_cfg, "val")
        self._create_split(self.test_cfg, "test")

    def _load_split(self, split: str) -> Dataset:
        """Load one static dataset split."""
        if self.structured_materialization:
            config = {"train": self.train_cfg, "val": self.val_cfg, "test": self.test_cfg}[split]
            return self._open_structured_dataset(
                self.root / split,
                self.structured_root / split,
                split=split,
                expected_length=int(config.dataset_length),
                target_labels=self.target_labels,
            )
        return StaticTorchSigDataset(
            root=self.root / split,
            file_handler_class=self.file_reader,
            target_labels=self.target_labels,
        )

    def setup(self, stage: str | None = None) -> None:
        """Load the datasets required for a Lightning stage."""
        if stage in (None, "fit"):
            if self.train is None:
                self.train = self._load_split("train")
                if self.structured_materialization and self.structured_train_transforms:
                    self.train = _OnlineTransformDataset(self.train, self.structured_train_transforms, self.train_cfg.seed)

            if self.val is None:
                self.val = self._load_split("val")

        if stage == "validate" and self.val is None:
            self.val = self._load_split("val")

        if stage in (None, "test", "predict") and self.test is None:
            self.test = self._load_split("test")

    def _build_dataloader(
        self,
        dataset: Dataset | None,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        """Build a deterministic model-facing DataLoader."""
        if dataset is None:
            raise RuntimeError("Dataset has not been initialized. Call setup() before requesting its DataLoader.")

        generator = Generator().manual_seed(seed)
        persistent_workers = self.num_workers > 0

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_dataloader(
            self.train,
            shuffle=self.shuffle,
            seed=self.train_cfg.seed,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_dataloader(
            self.val,
            shuffle=False,
            seed=self.val_cfg.seed,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self._build_dataloader(
            self.test,
            shuffle=False,
            seed=self.test_cfg.seed,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the test split for prediction."""
        return self._build_dataloader(
            self.test,
            shuffle=False,
            seed=self.test_cfg.seed,
        )
