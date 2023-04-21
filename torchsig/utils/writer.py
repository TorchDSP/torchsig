from torchsig.utils.dataset import SignalDataset
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import pickle
import random
import torch
import tqdm
import lmdb
import os


class DatasetLoader:
    """Dataset Loader takes on the responsibility of defining how a SignalDataset
    is loaded into memory (usually in parallel)

    Args:
        dataset (SignalDataset): _description_
        seed (int): _description_
        num_workers (int, optional): _description_. Defaults to os.cpu_count().
        batch_size (int, optional): _description_. Defaults to os.cpu_count().
    """

    def __init__(
        self,
        dataset: SignalDataset,
        seed: int,
        num_workers: int = os.cpu_count(),
        batch_size: int = os.cpu_count(),
    ) -> None:

        self.loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            worker_init_fn=partial(DatasetLoader.worker_init_fn, seed=seed),
        )
        self.seed = seed
        self.length = int(len(dataset) / batch_size)

    def __len__(self):
        return self.length

    @staticmethod
    def worker_init_fn(worker_id: int, seed: int):
        seed = seed + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __next__(self):
        data, label = next(self.loader)
        return data, label

    def __iter__(self):
        return iter(self.loader)


class DatasetWriter:
    def write(self, batch):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError


class LMDBDatasetWriter(DatasetWriter):
    def __init__(self, path: str, *args, **kwargs):
        super(LMDBDatasetWriter, self).__init__(*args, **kwargs)
        self.path = path
        self.env = lmdb.Environment(path, subdir=True, map_size=int(1e12), max_dbs=2)
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")

    def write(self, batch):
        data, labels = batch
        with self.env.begin(write=True) as txn:
            last_idx = txn.stat(db=self.data_db)["entries"]
            for element_idx in range(len(data)):
                txn.put(
                    pickle.dumps(last_idx + element_idx),
                    pickle.dumps(data[element_idx]),
                    db=self.data_db,
                )
                txn.put(
                    pickle.dumps(last_idx + element_idx),
                    pickle.dumps((labels[0][element_idx], labels[1][element_idx])),
                    db=self.label_db,
                )

    def finalize(self):
        pass


class DatasetCreator:
    def __init__(self, loader: DatasetLoader, writer: DatasetWriter) -> None:
        self.loader = loader
        self.writer = writer

    def create(self):
        for batch in tqdm.tqdm(self.loader, total=len(self.loader)):
            self.writer.write(batch)
        self.writer.finalize()
