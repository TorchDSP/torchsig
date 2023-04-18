from torchsig.utils.types import SignalData, SignalDescription
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.transforms.transforms import NoTransform
from torchsig.datasets import conf
from torch.utils.data import Subset
from copy import deepcopy
from pathlib import Path
import numpy as np
import pickle
import lmdb


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

        use_signal_data (bool, optional):
            If True, data will be converted to SignalData objects as read in.
            Default: False.

    """

    _idx_to_name_dict = dict(zip(range(53), ModulationsDataset.default_classes))
    _name_to_idx_dict = dict(zip(ModulationsDataset.default_classes, range(53)))

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
        transform: callable = None,
        target_transform: callable = None,
        use_signal_data: bool = False,
    ):
        self.root = Path(root)
        self.train = train
        self.impaired = impaired
        self.eb_no = eb_no
        self.use_signal_data = use_signal_data

        self.T = transform if transform else NoTransform()
        self.TT = target_transform if target_transform else NoTransform()

        cfg: conf.Sig53Config = (
            "Sig53"
            + ("Impaired" if impaired else "Clean")
            + ("EbNo" if (impaired and eb_no) else "")
            + ("Train" if train else "Val")
            + "Config"
        )

        cfg = getattr(conf, cfg)()

        self.path = self.root / cfg.name
        self.env = lmdb.Environment(
            str(self.path).encode(), map_size=int(1e12), max_dbs=2
        )
        self.data_db = self.env.open_db(b"data")
        with self.env.begin(db=self.data_db, write=True) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple:
        with self.env.begin(db=self.data_db) as data_txn:
            idx = pickle.dumps(idx)
            item = pickle.loads(data_txn.get(idx))

        iq_data, mod, snr = item
        if self.use_signal_data:
            signal_desc = SignalDescription(
                class_name=self._idx_to_name_dict[mod],
                class_index=mod,
                snr=snr,
            )
            data = SignalData(
                data=deepcopy(iq_data.tobytes()),
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex64),
                signal_description=[signal_desc],
            )
            data = self.T(data)
            target = self.TT(data.signal_description)
            data = data.iq_data
        else:
            data = self.T(iq_data)
            target = (self.TT(mod), snr)

        return data, target


if __name__ == "__main__":
    from torchsig.utils.writer import DatasetLoader, LMDBDatasetWriter, DatasetCreator

    cfg = conf.Sig53CleanTrainConfig

    dataset = ModulationsDataset(
        level=cfg.level,
        num_samples=cfg.num_samples,
        num_iq_samples=cfg.num_iq_samples,
        use_class_idx=cfg.use_class_idx,
        include_snr=cfg.include_snr,
        eb_no=cfg.eb_no,
    )

    # metadata = {
    #     "impaired": True,
    #     "train": True,
    #     "eb_no": False,
    #     "num_samples": cfg.num_samples,
    #     "num_iq_samples": cfg.num_iq_samples,
    #     "idx_to_name": Sig53._idx_to_name_dict,
    #     "num_classes": 53,
    # }

    loader = DatasetLoader(
        Subset(dataset, np.arange(1000).tolist()),
        seed=12345678,
        num_workers=16,
        batch_size=16,
    )
    writer = LMDBDatasetWriter(path="torchsig/datasets/test1")
    creator = DatasetCreator(loader, writer)
    creator.create()

    loader2 = DatasetLoader(
        Subset(dataset, np.arange(1000).tolist()),
        seed=12345678,
        num_workers=16,
        batch_size=16,
    )
    writer = LMDBDatasetWriter(path="torchsig/datasets/test2")
    creator = DatasetCreator(loader2, writer)
    creator.create()
