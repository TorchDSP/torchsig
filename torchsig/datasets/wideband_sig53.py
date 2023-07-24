from torchsig.transforms.target_transforms import ListTupleToDesc
from torchsig.transforms.transforms import Identity
from torchsig.utils.types import Signal, create_signal_data
from torchsig.datasets import conf
from typing import Callable, List, Optional
from pathlib import Path
import numpy as np
import pickle
import lmdb
import os


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

    modulation_list: List[str] = [
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
    ):
        self.root = Path(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.train = train
        self.impaired = impaired

        self.T = transform if transform else Identity()
        self.TT = target_transform if target_transform else Identity()

        cfg = (
            "WidebandSig53"
            + ("Impaired" if impaired else "Clean")
            + ("Train" if train else "Val")
            + "Config"
        )
        cfg = getattr(conf, cfg)()

        self.signal_desc_transform = ListTupleToDesc(
            num_iq_samples=cfg.num_iq_samples,  # type: ignore
            class_list=self.modulation_list,
        )

        self.path = self.root / cfg.name  # type: ignore
        self.env = lmdb.Environment(
            str(self.path).encode(), map_size=int(1e12), max_dbs=2, lock=False
        )
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")
        with self.env.begin(db=self.data_db) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple:
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db) as data_txn:
            iq_data: np.ndarray = pickle.loads(data_txn.get(encoded_idx))

        with self.env.begin(db=self.label_db) as label_txn:
            label = pickle.loads(label_txn.get(encoded_idx))

        signal = Signal(
            data=create_signal_data(samples=iq_data),
            metadata=self.signal_desc_transform(label),
        )
        signal = self.T(signal)  # type: ignore
        target = self.TT(signal["metadata"])  # type: ignore

        return signal["data"]["samples"], target
