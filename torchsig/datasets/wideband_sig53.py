import os
import lmdb
import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm.autonotebook import tqdm
from typing import Callable, Optional, Tuple
from multiprocessing import Pool
import torch

from torchsig.datasets import conf
from torchsig.datasets.wideband import WidebandModulationsDataset
from torchsig.transforms.target_transforms import (
    DescToListTuple,
    ListTupleToDesc,
)
from torchsig.utils.types import SignalData


def _identity(x):
    return x


# Helper function for multiprocessing
def _get_data(idx, cfg):
    np.random.seed(cfg.seed + idx * 53)
    wb_mds = WidebandModulationsDataset(
        level=cfg.level,
        num_iq_samples=cfg.num_iq_samples,
        num_samples=1,  # Dataset is randomly generated when indexed, so length here does not matter
        target_transform=DescToListTuple(),
        seed=cfg.seed + idx * 53,
        use_gpu=cfg.use_gpu,
    )
    return wb_mds[0]


class WidebandSig53:
    """The Official WidebandSig53 dataset

    Args:
        root (string): Root directory of dataset. A folder will be created for the requested version
            of the dataset, an mdb file inside contains the data and labels.
        train (bool, optional): If True, constructs the corresponding training set,
            otherwise constructs the corresponding val set
        impaired (bool, optional): If True, will construct the impaired version of the dataset,
            with data passed through a seeded channel model
        transform (callable, optional): A function/transform that takes in a complex64 ndarray
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target class (int) and returns a transformed version
        regenerate (bool, optional): If True, data will be generated from scratch, otherwise the version
            on disk will be used if it exists.
        use_signal_data (bool, optional): If True, data and annotations will be setup as SignalData objects
        gen_batch_size (int, optional): Batch size for parallelized data generation

    """

    modulation_list = [
        "ook",
        "bpsk",
        "4pam",
        "4ask",
        "qpsk",
        "8pam",
        "8ask",
        "8psk",
        "16qam",
        "16pam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32pam",
        "32ask",
        "32psk",
        "64qam",
        "64pam",
        "64ask",
        "64psk",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        impaired: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_signal_data: bool = True,
    ):
        self.root = Path(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.train = train
        self.impaired = impaired

        self.T = transform if transform else _identity
        self.TT = target_transform if target_transform else _identity

        cfg = (
            "WidebandSig53"
            + ("Impaired" if impaired else "Clean")
            + ("Train" if train else "Val")
            + "Config"
        )
        cfg = getattr(conf, cfg)()

        self.use_signal_data = use_signal_data
        self.signal_desc_transform = ListTupleToDesc(
            num_iq_samples=cfg.num_iq_samples,
            class_list=self.modulation_list,
        )

        self.path = self.root / cfg.name
        self.env = lmdb.Environment(
            str(self.path).encode(), map_size=int(1e12), max_dbs=2, lock=False
        )
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")
        with self.env.begin(db=self.data_db) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx)).numpy()

        with self.env.begin(db=self.label_db) as label_txn:
            label = pickle.loads(label_txn.get(encoded_idx))

        if self.use_signal_data:
            data = SignalData(
                data=deepcopy(iq_data.tobytes()),
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=self.signal_desc_transform(label),
            )
            data = self.T(data)
            target = self.TT(data.signal_description)
            data = data.iq_data
        else:
            data = self.T(iq_data)
            target = self.TT(label)
        return data, target
