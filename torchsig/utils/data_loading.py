"""Collate function and DataLoader with worker seeding for TorchSig.
Provides:
    - metadata_padding_collate_fn: pads variable-length metadata in each batch.
    - WorkerSeedingDataLoader: seeds each worker process differently for reproducibility.
"""

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchsig.utils.random import Seedable


def metadata_padding_collate_fn(batch):
    """Collate a batch of (data, metadata_list) pairs, padding metadata to equal lengths.

    Metadata for each sample is a list of dicts. This function:
        1. Finds the maximum metadata-list length in the batch.
        2. Pads shorter metadata lists with default values.
        3. Stacks data tensors and metadata fields into batched tensors.

    Args:
        batch: A list where each element is a tuple of:
            - x: any object convertible to a NumPy array (e.g., tensor, array).
            - y: a list of metadata dicts, where each dict shares the same set of keys.

    Returns:
        A tuple containing:
            - data_tensor: stacked torch.Tensor of all x values, shape (batch_size, ...).
            - metadata_tensors: dict mapping each metadata key to a Tensor of shape
              (batch_size, max_sequence_length).

    Raises:
        ValueError: if any element in `batch` is not a tuple of length 2.
    """
    default_y_value = 0

    batch_max_len = 0
    iqs = []
    y_tensor_obj = {}

    for data_pair in batch:
        if not isinstance(data_pair, tuple) or len(data_pair) != 2:
            raise ValueError(
                f"{data_pair} is not a valid (x, y) pair; this collate function "
                "expects datasets to return tuples of (x, y)"
            )

        _, metadata_list = data_pair
        batch_max_len = max(batch_max_len, len(metadata_list))

        for metadata_obj in metadata_list:
            for key in metadata_obj:
                if key not in y_tensor_obj:
                    y_tensor_obj[key] = []

        iqs.append(data_pair[0])

    if batch_max_len < 1:
        # No metadata to pad, return raw list for metadata
        return torch.Tensor(np.array(iqs)), y_tensor_obj

    # Initialize per-key lists for each time step
    for key in y_tensor_obj:
        y_tensor_obj[key] = [[] for _ in range(batch_max_len)]

    # Fill in metadata values or default where missing
    for _, metadata_list in batch:
        for i in range(batch_max_len):
            if i < len(metadata_list):
                metadata_obj = metadata_list[i]
                # Use .items() here to iterate key-value pairs in y_tensor_obj
                for key, value_lists in y_tensor_obj.items():
                    # Use .get() with default_y_value
                    value_lists[i].append(metadata_obj.get(key, default_y_value))
            else:
                for value_lists in y_tensor_obj.values():
                    value_lists[i].append(default_y_value)

    # Convert lists to tensors, dropping invalid keys
    final_tensor_obj = {}
    for key, sequences in y_tensor_obj.items():
        try:
            final_tensor_obj[key] = torch.Tensor(np.array(sequences))
        except (ValueError, TypeError, MemoryError) as e:
            warnings.warn(
                f"Dropping key value: '{key}' because it contained invalid tensor values: {type(e).__name__}",
                stacklevel=2
            )
    return torch.Tensor(np.array(iqs)), final_tensor_obj


class WorkerSeedingDataLoader(DataLoader, Seedable):
    """DataLoader that seeds each worker process differently using a shared seed.

    This loader prohibits external `worker_init_fn` definitions and sets its own
    init function to ensure reproducible randomness in multi-worker pipelines.
    """

    def __init__(self, dataset, seed=None, **kwargs):
        """Initialize DataLoader and Seedable, then assign custom worker init.

        Args:
            dataset: The dataset to load.
            seed: Optional seed value. If None, a random seed is generated.
            **kwargs: Passed to both `DataLoader` and `Seedable` initializers.

        Raises:
            ValueError: if `worker_init_fn` is provided in kwargs.
        """
        if seed is None:
            seed = np.random.randint(
                1000
            )  # just pick a seed if none is given; should still seed with somehting
        DataLoader.__init__(self, dataset, **kwargs)
        Seedable.__init__(self, seed=seed)
        dataset.seed(seed)
        if self.worker_init_fn:
            raise ValueError(
                "No worker_init_fn should be given to WorkerSeedingDataLoader; "
                "it will set its own worker_init_fn."
            )

        self.worker_init_fn = self.init_worker_seed

    def seed(self, seed_val):
        """Set the seed value for both the loader and its dataset.

        Args:
            seed_val: The seed value to set.
        """
        Seedable.seed(self, seed_val)
        self.dataset.seed(seed_val)

    def init_worker_seed(self, worker_id):
        """Set a unique random seed for each worker process.

        Uses the shared `random_generator` from the `Seedable` mixin to derive
        a new seed per `worker_id`.

        Args:
            worker_id: The integer ID of the worker process.
        """
        from torch.utils.data import get_worker_info

        seed = int(self.random_generator.random() * 100 + 1) * (worker_id + 1)
        print(worker_id, "seed: ", seed)
        get_worker_info().dataset.seed(seed)
