from torchsig.utils.dataset import SignalDataset
from torch.utils.data import DataLoader
from typing import Callable, Optional
from functools import partial
from io import BytesIO
import numpy as np
import pickle
import random
import lmdb
import torch
import tqdm
import os


class DatasetLoader:
    """Dataset Loader takes on the responsibility of defining how a SignalDataset
    is loaded into memory (usually in parallel)

    Args:
        dataset (SignalDataset): Dataset from which to pull data
        seed (int): seed for the underlying dataset
        num_workers (int, optional): _description_. Defaults to os.cpu_count().
        batch_size (int, optional): _description_. Defaults to os.cpu_count().
    """

    @staticmethod
    def worker_init_fn(worker_id: int, seed: int):
        seed = seed + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __init__(
        self,
        dataset: SignalDataset,
        seed: int,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        num_workers = num_workers if num_workers else os.cpu_count()
        batch_size = batch_size if batch_size else os.cpu_count()
        assert num_workers is not None
        assert batch_size is not None
        self.loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            worker_init_fn=partial(DatasetLoader.worker_init_fn, seed=seed),
            multiprocessing_context=torch.multiprocessing.get_context("fork"),
            collate_fn=collate_fn,
        )

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        data, label = next(self.loader)
        return data, label

    def __iter__(self):
        return iter(self.loader)


class DatasetWriter:
    """The Interface for DatasetWriter classes to override"""

    def exists(self):
        raise NotImplementedError

    def write(self, batch):
        raise NotImplementedError


class LMDBDatasetWriter(DatasetWriter):
    """A DatasetWriter for lmdb databases

    Args:
        path (str): directory in which to keep the database files
    """

    def __init__(self, path: str, compress: bool = False, *args, **kwargs):
        super(LMDBDatasetWriter, self).__init__(*args, **kwargs)
        self.path = path
        self.compress = compress
        self.env = lmdb.Environment(path, subdir=True, map_size=int(4e12), max_dbs=2)
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")

    def exists(self):
        with lmdb.Transaction(self.env, self.data_db) as txn:
            if txn.stat()["entries"] > 0:
                return True
        return False

    @staticmethod
    def _compress(
        data: np.ndarray, storage_type: np.dtype = np.dtype(np.int16)
    ) -> np.ndarray:
        if storage_type == np.float64:
            return data

        floats = data.view(np.float64)
        max_amp = np.max(np.abs(floats))
        normalized = (np.iinfo(storage_type).max - 1) * floats / max_amp
        digitized: np.ndarray = (
            np.digitize(
                normalized,
                np.arange(np.iinfo(storage_type).min, np.iinfo(storage_type).max),
                right=True,
            )
            - np.iinfo(storage_type).max
        )
        return digitized.astype(storage_type)

    def write(self, batch):
        data, labels = batch
        with self.env.begin() as txn:
            last_idx = txn.stat(db=self.data_db)["entries"]

        with self.env.begin(write=True) as txn:
            if isinstance(labels, tuple):
                for label_idx, label in enumerate(labels):
                    txn.put(
                        pickle.dumps(last_idx + label_idx),
                        pickle.dumps(tuple(label)),
                        db=self.label_db,
                    )
            if isinstance(labels, list):
                for label_idx, label in enumerate(zip(*labels)):
                    txn.put(
                        pickle.dumps(last_idx + label_idx),
                        pickle.dumps(label),
                        db=self.label_db,
                    )
            for element_idx in range(len(data)):
                if not isinstance(data[element_idx], np.ndarray):
                    compressed = self._compress(
                        data[element_idx].numpy(),
                        np.int16 if self.compress else np.float64,
                    )
                    txn.put(
                        pickle.dumps(last_idx + element_idx),
                        pickle.dumps(compressed),
                        db=self.data_db,
                    )
                    continue
                compressed = self._compress(
                    data[element_idx],
                    np.int16 if self.compress else np.float64,
                )
                txn.put(
                    pickle.dumps(last_idx + element_idx),
                    pickle.dumps(compressed),
                    db=self.data_db,
                )


class DatasetCreator:
    """Class is whose sole responsibility is to interface a dataset (a generator)
    with a DatasetLoader and a DatasetWriter to produce a static dataset with a
    parallelized generation scheme and some specified storage format.

    Args:
        dataset (SignalDataset): dataset class
        seed (int): seed for the dataset
        path (str): path to store the static dataset
        writer (DatasetWriter, optional): DatasetWriter. Defaults to LMDBDatasetWriter.
        loader (DatasetLoader, optional): DatasetLoader. Defaults to DatasetLoader.
    """

    def __init__(
        self,
        dataset: SignalDataset,
        seed: int,
        path: str,
        writer: Optional[DatasetWriter] = None,
        loader: Optional[DatasetLoader] = None,
    ) -> None:
        self.loader = DatasetLoader(dataset=dataset, seed=seed)
        self.loader = self.loader if not loader else loader
        self.writer = LMDBDatasetWriter(path=path)
        self.writer = self.writer if not writer else writer  # type: ignore
        self.path = path

    def create(self):
        if self.writer.exists():
            print("Dataset already exists in {}. Not regenerating".format(self.path))
            return

        for batch in tqdm.tqdm(self.loader, total=len(self.loader)):
            self.writer.write(batch)
