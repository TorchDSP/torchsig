from torchsig.transforms.target_transforms import ListTupleToDesc
from torchsig.transforms.transforms import Identity
from torchsig.utils.types import Signal, create_signal_data
from torchsig.datasets import conf
from torchsig.datasets.signal_classes import sig53
from typing import Callable, List, Optional
from pathlib import Path
import numpy as np
import pickle
import lmdb
import os


class WidebandSig53:
    """The Official WidebandSig53 dataset with optimized loading."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        impaired: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        class_list: Optional[List] = None
    ):
        self.root = Path(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.train = train
        self.impaired = impaired
        self.class_list = sig53.class_list if class_list is None else class_list

        self.T = transform if transform else Identity()
        self.TT = target_transform if target_transform else Identity()

        cfg = ("WidebandSig53" + ("Impaired" if impaired else "Clean") + ("Train" if train else "Val") + "Config")
        cfg = getattr(conf, cfg)()

        self.path = self.root / cfg.name  # type: ignore
        self.env = lmdb.open(str(self.path), map_size=int(1e12), max_dbs=2, readonly=True, lock=False, readahead=False)
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")
        
        with self.env.begin(db=self.data_db, write=False) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        return self.length

    def _get_data_label(self, idx: int):
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db, write=False) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx))
        
        with self.env.begin(db=self.label_db, write=False) as label_txn:
            label = pickle.loads(label_txn.get(encoded_idx))

        return iq_data, label

    def __getitem__(self, idx: int) -> tuple:
        iq_data, label = self._get_data_label(idx)
        
        signal = Signal(data=create_signal_data(samples=iq_data), metadata=(label))
        signal = self.T(signal)  # type: ignore
        target = self.TT(signal["metadata"])  # type: ignore
        
        return signal["data"]["samples"], target