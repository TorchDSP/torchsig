import os
import lmdb
import pickle
import shutil
import numpy as np
from pathlib import Path
from copy import deepcopy
from ast import literal_eval
from tqdm.autonotebook import tqdm
from typing import Callable, Optional, Tuple
from multiprocessing import Pool

from torchsig.datasets import conf
from torchsig.datasets.wideband import WidebandModulationsDataset
import torchsig.transforms as ST
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
        target_transform=ST.DescToListTuple(),
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
        regenerate: bool = False,
        use_signal_data: bool = True,
        gen_batch_size: int = 1,
        use_gpu: Optional[bool] = None,
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
        cfg.use_gpu = use_gpu if use_gpu is not None else cfg.use_gpu

        self.use_signal_data = use_signal_data
        self.signal_desc_transform = ST.ListTupleToDesc(
            num_iq_samples=cfg.num_iq_samples,
            class_list=self.modulation_list,
        )

        self.path = self.root / cfg.name
        self.length = cfg.num_samples
        regenerate = regenerate or not os.path.isdir(self.path)

        if regenerate and os.path.isdir(self.path):
            shutil.rmtree(self.path)

        self._env = lmdb.open(
            str(self.path).encode(),
            max_dbs=2,
            map_size=int(1e12),
            max_readers=512,
            readahead=False,
        )

        self._sample_db = self._env.open_db(b"iq_samples")
        self._annotation_db = self._env.open_db(b"annotation")

        if regenerate:
            if self.length % gen_batch_size != 0:
                while self.length % gen_batch_size != 0:
                    gen_batch_size -= 1
                print("Rounding batch size down to {}".format(gen_batch_size))
            self.gen_batch_size = gen_batch_size

            print("Generating dataset...")
            self._generate_data(cfg)
        else:
            print("Existing data found, skipping data generation")

        self._sample_txn = self._env.begin(db=self._sample_db)
        self._annotation_txn = self._env.begin(db=self._annotation_db)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        idx = str(idx).encode()
        x = pickle.loads(self._sample_txn.get(idx))
        y = literal_eval(self._annotation_txn.get(idx).decode("utf8"))
        if self.use_signal_data:
            data = SignalData(
                data=deepcopy(x.tobytes()),
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex64),
                signal_description=self.signal_desc_transform(y),
            )
            data = self.T(data)
            target = self.TT(data.signal_description)
            data = data.iq_data
        else:
            data = self.T(x)
            target = self.TT(y)
        return data, target

    def _generate_data(self, cfg: conf.WidebandSig53Config) -> None:
        state = np.random.get_state()
        np.random.seed(cfg.seed)

        # Data retrieval batching for speed
        batch_size = self.gen_batch_size
        num_batches = int(self.length / batch_size)

        if batch_size == 1:
            # Splitting case for single batch for tqdm progress bar over samples instead of batches
            # Sequentially write retrieved data, annotations to LMDB
            for i in tqdm(range(self.length)):
                np.random.seed(cfg.seed + i * 53)
                wb_mds = WidebandModulationsDataset(
                    level=cfg.level,
                    num_iq_samples=cfg.num_iq_samples,
                    num_samples=1,  # Dataset is randomly generated when indexed, so length here does not matter
                    target_transform=ST.DescToListTuple(),
                    seed=cfg.seed + i * 53,
                    use_gpu=cfg.use_gpu,
                )
                data, annotation = wb_mds[0]
                data_c64 = data.astype(np.complex64)
                with self._env.begin(write=True) as txn:
                    txn.put(str(i).encode(), pickle.dumps(data_c64), db=self._sample_db)
                    txn.put(
                        str(i).encode(),
                        str(annotation).encode(),
                        db=self._annotation_db,
                    )

        else:
            # Batched multiprocessing data, annotation retrieval
            lmdb_idx = 0
            for batch_idx in tqdm(range(num_batches)):
                process_index = []
                for batch_sample_idx in range(batch_size):
                    process_index.append(
                        (int(batch_idx * batch_size + batch_sample_idx), cfg)
                    )
                pool = Pool(batch_size)
                result = pool.starmap(_get_data, process_index)

                # Sequentially write retrieved data, annotations to LMDB
                for data, annotation in result:
                    data_c64 = data.astype(np.complex64)
                    with self._env.begin(write=True) as txn:
                        txn.put(
                            str(lmdb_idx).encode(),
                            pickle.dumps(data_c64),
                            db=self._sample_db,
                        )
                        txn.put(
                            str(lmdb_idx).encode(),
                            str(annotation).encode(),
                            db=self._annotation_db,
                        )
                    lmdb_idx += 1

        np.random.set_state(state)
