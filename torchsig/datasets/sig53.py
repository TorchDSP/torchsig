import os
import lmdb
import shutil
import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm.autonotebook import tqdm
from typing import Callable, Optional, Tuple

from torchsig.utils.types import SignalData, SignalDescription
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf


def _identity(x):
    return x


class Sig53:
    """The Official Sig53 dataset

    Args:
        root (string):
            Root directory of dataset. A folder will be created for the
            requested version of the dataset, an mdb file inside contains the
            data and labels.

        train (bool, optional):
            If True, constructs the corresponding training set, otherwise
            constructs the corresponding val set

        impaired (bool, optional):
            If True, will construct the impaired version of the dataset, with
            data passed through a seeded channel model

        eb_no (bool, optional):
            If True, will define SNR as Eb/No; If False, will define SNR as Es/No

        transform (callable, optional):
            A function/transform that takes in a complex64 ndarray and returns
            a transformed version

        target_transform (callable, optional):
            A function/transform that takes in the target class (int) and
            returns a transformed version

        regenerate (bool, optional):
            If True, data will be generated from scratch, otherwise the version
            on disk will be used if it exists.

        use_signal_data (bool, optional):
            If True, data will be converted to SignalData objects as read in.
            Default: False.

    """

    @staticmethod
    def convert_idx_to_name(idx: int) -> str:
        return Sig53._idx_to_name_dict.get(idx, "unknown")

    @staticmethod
    def convert_name_to_idx(name: str) -> int:
        return Sig53._name_to_idx_dict.get(name, -1)

    def __init__(
        self,
        root: str,
        train: bool = True,
        impaired: bool = True,
        eb_no: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        regenerate: bool = False,
        use_signal_data: bool = False,
        generation_test: bool = False,
    ):
        self.root = Path(root)
        self.train = train
        self.impaired = impaired
        self.eb_no = eb_no
        self.use_signal_data = use_signal_data

        self.T = transform if transform else _identity
        self.TT = target_transform if target_transform else _identity

        cfg = (
            "Sig53"
            + ("Impaired" if impaired else "Clean")
            + ("EbNo" if (impaired and eb_no) else "")
            + ("Train" if train else "Val")
            + ("QA" if generation_test else "")
            + "Config"
        )

        cfg = getattr(conf, cfg)()

        self.path = self.root / cfg.name
        self.length = cfg.num_samples
        regenerate = regenerate or not os.path.isdir(self.path)

        if regenerate and os.path.isdir(self.path):
            shutil.rmtree(self.path)

        self._env = lmdb.open(
            str(self.path).encode(),
            max_dbs=3,
            map_size=int(1e12),
            max_readers=512,
            readahead=False,
        )

        self._sample_db = self._env.open_db(b"iq_samples")
        self._modulation_db = self._env.open_db(b"modulation")
        self._snr_db = self._env.open_db(b"snr")

        if regenerate:
            print("Generating dataset...")
            self._generate_data(cfg)
        else:
            print("Existing data found, skipping data generation")

        self._sample_txn = self._env.begin(db=self._sample_db)
        self._modulation_txn = self._env.begin(db=self._modulation_db)
        self._snr_txn = self._env.begin(db=self._snr_db)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        idx = str(idx).encode()
        x = pickle.loads(self._sample_txn.get(idx))
        y = int(self._modulation_txn.get(idx))
        snr = float(self._snr_txn.get(idx))
        if self.use_signal_data:
            signal_desc = SignalDescription(
                class_name=self._idx_to_name_dict[y],
                class_index=y,
                snr=snr,
            )
            data = SignalData(
                data=deepcopy(x.tobytes()),
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex64),
                signal_description=[signal_desc],
            )
            data = self.T(data)
            target = self.TT(data.signal_description)
            data = data.iq_data
        else:
            data = self.T(x)
            target = (self.TT(y), snr)
        return data, target

    def _generate_data(self, cfg: conf.Sig53Config) -> None:
        state = np.random.get_state()
        np.random.seed(cfg.seed)
        mds = ModulationsDataset(
            level=cfg.level,
            num_samples=cfg.num_samples,
            num_iq_samples=cfg.num_iq_samples,
            use_class_idx=cfg.use_class_idx,
            include_snr=cfg.include_snr,
            eb_no=cfg.eb_no,
        )

        metadata = {
            "impaired": self.impaired,
            "train": self.train,
            "eb_no": self.eb_no,
            "num_samples": cfg.num_samples,
            "num_iq_samples": cfg.num_iq_samples,
            "idx_to_name": Sig53._idx_to_name_dict,
            "num_classes": 53,
        }

        with self._env.begin(write=True) as txn:
            txn.put(b"metadata", pickle.dumps(metadata))

        for i in tqdm(range(len(mds))):
            data, (mod, snr) = mds[i]

            data_c64 = data.astype(np.complex64)

            with self._env.begin(write=True) as txn:
                txn.put(str(i).encode(), pickle.dumps(data_c64), db=self._sample_db)
                txn.put(str(i).encode(), str(mod).encode(), db=self._modulation_db)
                txn.put(str(i).encode(), str(snr).encode(), db=self._snr_db)

        np.random.set_state(state)

    _idx_to_name_dict = dict(zip(range(53), ModulationsDataset.default_classes))
    _name_to_idx_dict = dict(zip(ModulationsDataset.default_classes, range(53)))
