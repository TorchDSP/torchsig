"""TorchSig Narrowband Dataset
"""

from torchsig.utils.types import SignalData, ModulatedRFMetadata, Signal
from torchsig.datasets.signal_classes import torchsig_signals
from typing import Any, Callable, Optional, Tuple
from torchsig.transforms import Identity
from torchsig.datasets import conf
from pathlib import Path
import numpy as np
import pickle
import lmdb


class TorchSigNarrowband:
    """The Official TorchSigNarrowband dataset

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

        use_signal_data (bool, optional):
            If True, data will be converted to SignalData objects as read in.
            Default: False.

    """

    _idx_to_name_dict = dict(zip(range(len(torchsig_signals.class_list)), torchsig_signals.class_list))
    _name_to_idx_dict = dict(zip(torchsig_signals.class_list, range(len(torchsig_signals.class_list))))

    @staticmethod
    def convert_idx_to_name(idx: int) -> str:
        return TorchSigNarrowband._idx_to_name_dict.get(idx, "unknown")

    @staticmethod
    def convert_name_to_idx(name: str) -> int:
        return TorchSigNarrowband._name_to_idx_dict.get(name, -1)

    def __init__(
        self,
        root: str,
        train: bool = True,
        impaired: bool = True,
        eb_no: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_signal_data: bool = False,
    ):
        self.root = Path(root)
        self.train = train
        self.impaired = impaired
        self.eb_no = eb_no
        self.use_signal_data = use_signal_data

        self.T = transform if transform else Identity()
        self.TT = target_transform if target_transform else Identity()

        cfg: conf.NarrowbandConfig = (
            "Narrowband"  # type: ignore
            + ("Impaired" if impaired else "Clean")
            + ("EbNo" if (impaired and eb_no) else "")
            + ("Train" if train else "Val")
            + "Config"
        )

        cfg = getattr(conf, cfg)()  # type: ignore

        self.path = self.root / cfg.name
        self.env = lmdb.Environment(str(self.path).encode(), map_size=int(1e12), max_dbs=2, lock=False)
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")
        with self.env.begin(db=self.data_db) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx))

        with self.env.begin(db=self.label_db) as label_txn:
            mod, snr = pickle.loads(label_txn.get(encoded_idx))

        mod = int(mod)
        signal_meta = ModulatedRFMetadata(
            sample_rate=0.0,
            num_samples=iq_data.shape[0],
            complex=True,
            lower_freq=-0.25,
            upper_freq=0.25,
            center_freq=0.0,
            bandwidth=0.5,
            start=0.0,
            stop=1.0,
            duration=1.0,
            bits_per_symbol=0.0,
            samples_per_symbol=0.0,
            excess_bandwidth=0.0,
            class_name=self._idx_to_name_dict[mod],
            class_index=mod,
            snr=snr,
        )
        signal_data: SignalData = SignalData(samples=iq_data)
        signal = Signal(data=signal_data, metadata=[signal_meta])
        if self.use_signal_data:
            signal = self.T(signal)  # type: ignore
            target = self.TT(signal["metadata"])  # type: ignore
            return signal["data"]["samples"], target

        signal = self.T(signal)  # type: ignore
        target = (self.TT(mod), snr)  # type: ignore

        return signal["data"]["samples"], target
