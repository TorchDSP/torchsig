"""Dataset Writer Utils"""

from __future__ import annotations

import concurrent.futures
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import yaml
from torch.utils.data._utils.collate import default_collate as torch_default_collate
from tqdm.auto import tqdm

from torchsig.utils.file_handlers.hdf5 import HDF5Writer

# TorchSig
from torchsig.utils.yaml import write_dict_to_yaml

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from torchsig.utils.file_handlers.base_handler import FileWriter


def default_collate_fn(batch):
    """Collates a batch by zipping its elements together. Note: not pickle-safe for complex
    nested structures, but works for typical (data, label) batches.

    Args:
        batch (tuple): A batch from the dataloader.

    Returns:
        tuple: A tuple of zipped elements, where each element corresponds to a single batch item.
    """
    return tuple(zip(*batch))


def identity_collate_fn(batch):
    """Pickle-safe identity collate for Signal objects (returns list unchanged)."""
    return batch


@dataclass(frozen=True)
class _DatasetExistenceProbe:
    """Configurable notion of 'dataset exists' without entering FileWriter.__enter__()."""
    root: Path
    maybe_data_file: Path | None

    def exists(self) -> bool:
        if not self.root.exists() or not self.root.is_dir():
            return False
        if self.maybe_data_file is not None:
            return self.maybe_data_file.exists()
        # fallback: any content
        return any(self.root.iterdir())

def _deep_equal(a: Any, b: Any, *, float_rtol: float = 1e-9, float_atol: float = 0.0) -> bool:
    """Recursive equality for YAML-loaded structures (dict/list/scalars)."""
    if a is b:
        return True
    if a is None or b is None:
        return a is b

    # Floats: tolerate tiny rounding changes from serialization/IO
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(float(a), float(b), rel_tol=float_rtol, abs_tol=float_atol)
        return int(a) == int(b)

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k], float_rtol=float_rtol, float_atol=float_atol) for k in a)

    # Sequences
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y, float_rtol=float_rtol, float_atol=float_atol) for x, y in zip(a, b, strict=True))

    return a == b


class DatasetCreator:
    """Class for creating a dataset and saving it to disk in batches.

    This class generates a dataset if it does not already exist on disk.
    It processes the data in batches and saves it using a specified file handler.
    The class allows setting options like whether to overwrite existing datasets,
    batch size, and number of worker threads.

    Attributes:
        dataloader (DataLoader): The DataLoader used to load data in batches.
        root (Path): The root directory where the dataset will be saved.
        overwrite (bool): Flag indicating whether to overwrite an existing dataset.
        tqdm_desc (str): A description for the progress bar.
        file_handler (FileWriter): The file handler used for saving the dataset.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        dataset_length: int | None = None,
        root: str = ".",
        overwrite: bool = True,
        tqdm_desc: str | None = None,
        file_handler: FileWriter = HDF5Writer,
        multithreading: bool = True,
        max_inflight_futures: int = 32,
        **kwargs,
    ):
        """Initializes the DatasetCreator.

        Args:
            dataloader (DataLoader): DataLoader used to load data in batches. Required.
            dataset_length (int): Number of dataset items to be created. Length inferrence attempted if not provided.
            root (Path): Root directory where the dataset files will be saved. Defaults to current directory.
            overwrite (bool): Flag indicating whether to overwrite an existing dataset. Defaults to True.
            tqdm_desc (str): Description for the progress bar.
            file_handler (FileWriter): File handler used to write dataset. Defaults to HDF5Writer.
            multithreading (bool): Whether to use multithreading for writing batches. Defaults to True.
            max_inflight_futures (int): Maximum number of concurrent futures when using multithreading. Defaults to 32.
            **kwargs: Additional arguments for the file handler.
        """
        # File attributes
        self.root = Path(root)
        self.dataset_info_filepath = self.root.joinpath("dataset_info.yaml")
        self.writer_info_filepath = self.root.joinpath("writer_info.yaml")
        self.file_handler = file_handler
        self.kwargs = dict(kwargs)

        self.overwrite = bool(overwrite)
        self.multithreading = bool(multithreading)
        self.max_inflight_futures = int(max_inflight_futures)

        self.dataloader = dataloader
        self.dataset_length_requested = self._infer_dataset_length(dataset_length)

        # optional
        self.batch_size = getattr(dataloader, "batch_size", None)
        self.num_workers = getattr(dataloader, "num_workers", None)

        self.tqdm_desc = "Generating Dataset:" if tqdm_desc is None else tqdm_desc

        # counters
        self.items_written = 0
        self._counter_lock = threading.Lock()
        self._msg_timer = None

    def _infer_dataset_length(self, dataset_length: int | None) -> int:
        """Infer dataset length or require it explicitly for iterable datasets.
        For map-style datasets, the length can be inferred from the dataset object.
        For iterable datasets, the length cannot be inferred and must be provided.

        Args:
            dataset_length (int | None): The length of the dataset to be created. If None
                the method will attempt to infer the length from the dataloader's dataset.

        Returns:
            int: dataset length.
        """
        if dataset_length is not None:
            return int(dataset_length)

        # map-style datasets: try len(dataloader.dataset) to infer dataset length
        try:
            inferred_length = len(self.dataloader.dataset)
        except Exception as e:  # pylint: disable=broad-except
            raise ValueError(
                "dataset_length must be provided when writing from an IterableDataset "
                "(e.g., TorchSigIterableDataset), because length cannot be inferred."
            ) from e
        return int(inferred_length)

    def _existence_probe(self) -> _DatasetExistenceProbe:
        """Instantiate writer without entering context to not enter setup, resetting folder."""
        maybe_data_file = None
        try:
            writer = self.file_handler(root=self.root)
            maybe_data_file = getattr(writer, "datapath", None)
            if isinstance(maybe_data_file, (str, Path)):
                maybe_data_file = Path(maybe_data_file)
        except Exception:
            # best-effort only
            maybe_data_file = None
        return _DatasetExistenceProbe(root=self.root, maybe_data_file=maybe_data_file)

    def _get_dataset_metadata_dict(self) -> dict[str, Any]:
        """Best-effort extraction of dataset metadata for YAML.

        Returns:
            dict: dictionary containing dataset metadata information.
        """
        ds = self.dataloader.dataset
        if hasattr(ds, "get_full_metadata"):
            return ds.get_full_metadata()
        return {}

    def get_dataset_info_dict(self, *, dataset_length: int, original_target_labels: Any) -> dict[str, Any]:
        """Get metadata content for the dataset_info.yaml file.

        Returns:
            Dict[str, Any]: Dictionary containing the dataset metadata information.
        """
        ds = self.dataloader.dataset
        seed = getattr(ds, "rng_seed", None)

        return {
            "dataset_length": int(dataset_length),
            "seed": None if seed is None else int(seed),
            "target_labels": original_target_labels,
            "dataset_metadata": self._get_dataset_metadata_dict(),
        }

    def get_writer_info_dict(self, *, complete: bool) -> dict[str, Any]:
        """Returns a dictionary with information about the dataset writing configuration.
        Used primarily for creating content for the writer_info.yaml summary file.

        Returns:
            Dict[str, Any]: Dictionary containing the dataset writing configuration.
        """
        return {
            "root": str(self.root),
            "overwrite": bool(self.overwrite),
            "batch_size": None if self.batch_size is None else int(self.batch_size),
            "num_workers": None if self.num_workers is None else int(self.num_workers),
            "file_handler": getattr(self.file_handler, "__name__", str(self.file_handler)),
            "multithreading": bool(self.multithreading),
            "dataset_length_requested": int(self.dataset_length_requested),
            "items_written": int(self.items_written),
            "complete": bool(complete),
            "timestamp_unix": int(time()),
        }

    def check_yamls(self, *, expected_dataset_info: dict[str, Any]) -> tuple[bool, list[tuple[str, Any, Any]]]:
        """Returns (complete, differences) without mutating dataset or entering writer context."""
        differences: list[tuple[str, Any, Any]] = []

        if not self.writer_info_filepath.exists():
            return False, [("writer_info.yaml", "missing", "expected present")]

        with open(self.writer_info_filepath) as f:
            writer_disk = yaml.safe_load(f) or {}
        complete = bool(writer_disk.get("complete", False))

        if not self.dataset_info_filepath.exists():
            differences.append(("dataset_info.yaml", "missing", "expected present"))
            return complete, differences

        with open(self.dataset_info_filepath) as f:
            dataset_disk = yaml.safe_load(f) or {}

        stable_keys = ["seed", "target_labels", "dataset_metadata"]
        for k in stable_keys:
            if k not in dataset_disk:
                differences.append((k, "missing_on_disk", expected_dataset_info.get(k)))
                continue
            if not _deep_equal(dataset_disk.get(k), expected_dataset_info.get(k)):
                differences.append((k, dataset_disk.get(k), expected_dataset_info.get(k)))

        # Length must match requested for "no regeneration needed"
        disk_len = dataset_disk.get("dataset_length", None)
        if disk_len is None or int(disk_len) != int(self.dataset_length_requested):
            differences.append(("dataset_length", disk_len, int(self.dataset_length_requested)))

        return complete, differences

    def _ensure_signal_batch_mode(self) -> tuple[Any, Any]:
        """Mutate dataset/dataloader so DataLoader yields Signal objects; return (orig_target_labels, orig_collate_fn)."""
        ds = self.dataloader.dataset
        orig_target_labels = getattr(ds, "target_labels", None)
        orig_collate_fn = getattr(self.dataloader, "collate_fn", None)

        # Force dataset to return Signal objects (TorchSigIterableDataset behavior)
        if hasattr(ds, "target_labels"):
            ds.target_labels = None

        # Ensure DataLoader does not try to default-collate Signal objects into tensors
        if getattr(ds, "target_labels", None) is None and self.dataloader.collate_fn in (torch_default_collate, default_collate_fn):
            self.dataloader.collate_fn = identity_collate_fn

        return orig_target_labels, orig_collate_fn

    def create(self) -> None:
        """Creates the dataset on disk by writing batches to the file handler.

        This method generates the dataset in batches and saves it to disk. If the
        dataset already exists and `overwrite` is set to False, it will skip regeneration.

        The method also writes the dataset metadata and writing information to YAML files.

        Raises:
            ValueError: If the dataset is already generated and `overwrite` is set to False.
        """
        ds = self.dataloader.dataset
        orig_target_labels = getattr(ds, "target_labels", None)
        orig_collate_fn = getattr(self.dataloader, "collate_fn", None)

        expected_dataset_info = self.get_dataset_info_dict(
            dataset_length=self.dataset_length_requested,
            original_target_labels=orig_target_labels,
        )

        # Existence/overwrite decision before entering writer context.
        probe = self._existence_probe()
        if probe.exists() and not self.overwrite:
            complete, diffs = self.check_yamls(expected_dataset_info=expected_dataset_info)
            if complete and len(diffs) == 0:
                print(f"Dataset already exists in {self.root}. Not regenerating.")
                return
            if not complete:
                raise RuntimeError(
                    f"Dataset only partially exists in {self.root}. "
                    "Regenerate by setting overwrite=True."
                )
            print(f"Dataset exists at {self.root} but differs from current dataset config. Using dataset on disk.")
            for k, disk_v, cur_v in diffs:
                print(f"\t{k}: disk={disk_v} current={cur_v}")
            return

        # create dataset
        try:
            orig_target_labels, orig_collate_fn = self._ensure_signal_batch_mode()
            self.items_written = 0
            self._msg_timer = time()

            with self.file_handler(root=self.root) as writer:
                # Write initial YAMLs
                write_dict_to_yaml(self.dataset_info_filepath, self.get_dataset_info_dict(
                    dataset_length=0, original_target_labels=orig_target_labels,
                ))
                write_dict_to_yaml(self.writer_info_filepath, self.get_writer_info_dict(complete=False))

                remaining = self.dataset_length_requested

                # Best-effort progress bar total
                total_batches = None
                if isinstance(self.batch_size, int) and self.batch_size > 0:
                    total_batches = math.ceil(self.dataset_length_requested / self.batch_size)

                pbar = tqdm(desc=self.tqdm_desc, total=total_batches)

                if self.multithreading:
                    writer_lock = threading.Lock()
                    futures: list[concurrent.futures.Future[int]] = []

                    def submit_write(batch_idx: int, batch: Any) -> int:
                        # Lock writer.write for safety with HDF5Writer buffering
                        with writer_lock:
                            writer.write(batch_idx, batch)
                        return len(batch) if hasattr(batch, "__len__") else 1

                    batch_idx = 0
                    # Single executor; max_workers=1 is enough since writer calls are serialized
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        for batch in self.dataloader:
                            if remaining <= 0:
                                break
                            if hasattr(batch, "__len__") and len(batch) > remaining:
                                batch = batch[:remaining]
                            batch_len = len(batch) if hasattr(batch, "__len__") else 1
                            futures.append(executor.submit(submit_write, batch_idx, batch))
                            batch_idx += 1
                            remaining -= batch_len

                            if len(futures) >= self.max_inflight_futures:
                                for fut in futures:
                                    self.items_written += fut.result()
                                    pbar.update(1)
                                futures.clear()

                        for fut in futures:
                            self.items_written += fut.result()
                            pbar.update(1)

                else: # single-threaded writing
                    batch_idx = 0
                    for batch in self.dataloader:
                        if remaining <= 0:
                            break

                        if hasattr(batch, "__len__") and len(batch) > remaining:
                            batch = batch[:remaining]

                        batch_len = len(batch) if hasattr(batch, "__len__") else 1
                        writer.write(batch_idx, batch)
                        batch_idx += 1
                        remaining -= batch_len

                        self.items_written += batch_len
                        pbar.update(1)

                pbar.close()

            # Validate after successful context close
            if self.items_written != self.dataset_length_requested:
                raise RuntimeError(
                    f"DatasetCreator wrote {self.items_written} samples, "
                    f"expected {self.dataset_length_requested}."
                )

            # Final YAML update
            write_dict_to_yaml(self.dataset_info_filepath, self.get_dataset_info_dict(
                dataset_length=self.items_written,
                original_target_labels=orig_target_labels,
            ))
            write_dict_to_yaml(self.writer_info_filepath, self.get_writer_info_dict(complete=True))

        finally:
            # Always restore caller-visible state
            if hasattr(ds, "target_labels"):
                ds.target_labels = orig_target_labels
            if orig_collate_fn is not None:
                self.dataloader.collate_fn = orig_collate_fn
