"""File handlers for NumPy standard binary *.npy files."""

# TorchSig
import bisect
import csv
import itertools
import json
from pathlib import Path

# Built-In
# Third Party
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers import FileReader


class NPYReader(FileReader):
    """ "Handles reading externally stored data into TorchSig as datasets, with data
    formatted in standard NumPy binary .npy files and metadata in a JSON format.
    """

    def __init__(self, root: str):
        super().__init__(root=root)
        self.root_dir = root
        self.npy_files = sorted(Path(root).glob("*.npy"))
        if not self.npy_files:
            raise FileNotFoundError("No .npy files found in directory.")

        # Determine cumulative sample counts for each file
        self.file_start_indices = (
            []
        )  # start index of each file in the global index space
        total = 0
        for file_path in self.npy_files:
            # Load the file header (memory-mapped) to get number of samples in this file
            arr = np.load(file_path, mmap_mode="r")  # memmap, does not load entire file
            length = arr.shape[0]  # number of samples in this file (note: arr not kept)
            self.file_start_indices.append(total)
            total += length
        self.total_elements = total

        self.class_list: list[str] = ["BPSK", "QPSK", "Noise"]
        self.dataset_size: int = None
        self.dataset_metadata: dict = self._load_json_metadata()

        self.dataset_size = 0
        try:
            with open(f"{self.root}/info.json") as f:
                dataset_info = json.load(f)

            self.dataset_size = dataset_info["size"]
        except:
            raise ValueError(f"Error loading {self.root}/info.json")

    def _load_json_metadata(self) -> dict:
        try:
            with open(f"{self.root}/info.json") as f:
                dataset_info = json.load(f)
                return dataset_info
        except:
            raise ValueError(f"Error loading {self.root}/info.json")

    def read(self, idx: int) -> tuple[np.ndarray, list[dict]]:
        """Read and return the sample at global index `idx`."""
        if idx < 0 or idx >= self.total_elements:
            raise IndexError(
                f"Index {idx} out of range (0 <= idx < {self.total_elements})."
            )

        # Data
        # determine which file contains this index using binary search on file start indices
        file_idx = bisect.bisect_right(self.file_start_indices, idx) - 1

        # compute index within selected file
        in_file_idx = idx - self.file_start_indices[file_idx]

        # load only needed file chunk (memory-mapped)
        file_path = self.npy_files[file_idx]
        arr = np.load(file_path, mmap_mode="r")  # memmap file
        data = arr[in_file_idx]  # retrieve specific sample

        # Metadata
        with open(f"{self.root}/metadata.csv") as f:
            reader = csv.DictReader(
                f, fieldnames=["index", "label", "modcod", "sample_rate"]
            )
            # get to idx row
            row = next(itertools.islice(reader, idx, idx + 1), None)
            if row is None:
                raise IndexError(f"Metadata idx {idx} is out of bounds")

            row["index"] = int(row["index"])
            row["sample_rate"] = float(row["sample_rate"])
            # add class_name
            row["class_name"] = row["label"].lower()
            # add class index
            row["class_index"] = self.class_list.index(row["label"])

            row["num_signals_max"] = 1

            metadata = row

        return Signal(data=data, component_signals=[], metadata=metadata)

    def __len__(self) -> int:
        return self.dataset_size
