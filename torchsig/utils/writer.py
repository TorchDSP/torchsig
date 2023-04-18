from torchsig.utils.dataset import SignalDataset
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import tempfile
import pickle
import random
import torch
import tqdm
import lmdb
import os


class DatasetLoader:
    def __init__(
        self,
        dataset: SignalDataset,
        seed: int,
        num_workers: int = 16,
        batch_size: int = 128,
    ) -> None:
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
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

        # This is where we have to store information about
        # reading and writing the data.
        self.metadata_db = self.env.open_db(b"metadata")
        self.itemsize = 0
        self.pickle_sizes = []

    def write(self, batch):
        with open(os.path.join(self.path, "raw.bin"), "ab+") as binary_file:
            raw_binary = pickle.dumps(batch)
            binary_file.write(raw_binary)
            self.pickle_sizes.append(len(raw_binary))

    def finalize(self):
        with self.env.begin(db=self.data_db, write=True) as data_txn:
            with open(os.path.join(self.path, "raw.bin"), "rb") as binary_file:
                for size in tqdm.tqdm(self.pickle_sizes, total=len(self.pickle_sizes)):
                    item = pickle.loads(binary_file.read(size))
                    data, labels = item
                    for item_idx in range(len(data)):
                        item = [data[item_idx]]
                        for label_idx in range(len(labels)):
                            item.append(labels[label_idx][item_idx])

                        last_idx = data_txn.stat()["entries"]
                        data_txn.put(pickle.dumps(last_idx), pickle.dumps(tuple(item)))


class DatasetCreator:
    def __init__(self, loader: DatasetLoader, writer: DatasetWriter) -> None:
        self.loader = loader
        self.writer = writer

    def create(self):
        for batch in tqdm.tqdm(self.loader, total=len(self.loader)):
            self.writer.write(batch)
        self.writer.finalize()
