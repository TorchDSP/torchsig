import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from torchsig.utils.data_loading import (
    WorkerSeedingDataLoader,
    metadata_padding_collate_fn,
)


class SeedableToyDataset(Dataset):
    def __init__(self, length=8):
        self.length = length
        self.seed_history = []
        self.rng = np.random.default_rng(0)

    def seed(self, seed):
        self.seed_history.append(seed)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32), [{"value": float(self.rng.integers(0, 1000))}]


def test_metadata_padding_collate_fn_pads_variable_length_metadata():
    batch = [
        (np.array([1.0, 2.0]), [{"snr": 10.0, "class": 1.0}]),
        (
            np.array([3.0, 4.0]),
            [
                {"snr": 20.0, "class": 2.0},
                {"snr": 30.0, "class": 3.0},
            ],
        ),
    ]

    data, metadata = metadata_padding_collate_fn(batch)

    assert torch.equal(data, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert set(metadata) == {"snr", "class"}

    # Current implementation stores metadata as (max_metadata_len, batch_size).
    assert torch.equal(metadata["snr"], torch.tensor([[10.0, 20.0], [0.0, 30.0]]))
    assert torch.equal(metadata["class"], torch.tensor([[1.0, 2.0], [0.0, 3.0]]))


def test_metadata_padding_collate_fn_uses_default_for_missing_keys():
    batch = [
        (np.array([1.0]), [{"snr": 10.0}]),
        (np.array([2.0]), [{"class": 2.0}]),
    ]

    _, metadata = metadata_padding_collate_fn(batch)

    assert torch.equal(metadata["snr"], torch.tensor([[10.0, 0.0]]))
    assert torch.equal(metadata["class"], torch.tensor([[0.0, 2.0]]))


def test_metadata_padding_collate_fn_returns_empty_metadata_dict_when_no_metadata():
    batch = [
        (np.array([1.0, 2.0]), []),
        (np.array([3.0, 4.0]), []),
    ]

    data, metadata = metadata_padding_collate_fn(batch)

    assert torch.equal(data, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert metadata == {}


@pytest.mark.parametrize(
    "bad_batch_item",
    [
        [np.array([1.0]), []],
        (np.array([1.0]), [], "extra"),
        np.array([1.0]),
        {"x": np.array([1.0]), "y": []},
    ],
)
def test_metadata_padding_collate_fn_rejects_invalid_batch_items(bad_batch_item):
    with pytest.raises(ValueError, match="expects datasets to return tuples"):
        metadata_padding_collate_fn([bad_batch_item])


def test_metadata_padding_collate_fn_drops_invalid_tensor_keys_with_warning():
    batch = [
        (np.array([1.0]), [{"valid": 1.0, "invalid": object()}]),
        (np.array([2.0]), [{"valid": 2.0, "invalid": object()}]),
    ]

    with pytest.warns(UserWarning, match=re.escape("Dropping key value: 'invalid'")):
        _, metadata = metadata_padding_collate_fn(batch)

    assert set(metadata) == {"valid"}
    assert torch.equal(metadata["valid"], torch.tensor([[1.0, 2.0]]))


def test_worker_seeding_dataloader_seeds_dataset_on_init():
    dataset = SeedableToyDataset()

    loader = WorkerSeedingDataLoader(dataset, seed=123, batch_size=2)

    assert loader.seed == 123 or getattr(loader, "seed_value", 123) == 123
    assert dataset.seed_history == [123]
    assert loader.worker_init_fn == loader.init_worker_seed


def test_worker_seeding_dataloader_seed_updates_dataset_seed():
    dataset = SeedableToyDataset()
    loader = WorkerSeedingDataLoader(dataset, seed=123, batch_size=2)

    loader.seed(456)

    assert dataset.seed_history == [123, 456]


def test_worker_seeding_dataloader_rejects_external_worker_init_fn():
    dataset = SeedableToyDataset()

    with pytest.raises(ValueError, match="No worker_init_fn should be given"):
        WorkerSeedingDataLoader(
            dataset,
            seed=123,
            batch_size=2,
            worker_init_fn=lambda _: None,
        )


def test_worker_seeding_dataloader_iterates_with_zero_workers():
    dataset = SeedableToyDataset(length=4)
    loader = WorkerSeedingDataLoader(
        dataset,
        seed=123,
        batch_size=2,
        num_workers=0,
        collate_fn=metadata_padding_collate_fn,
    )

    data, metadata = next(iter(loader))

    assert data.shape == (2, 1)
    assert set(metadata) == {"value"}
    assert metadata["value"].shape == (1, 2)


def test_worker_seeding_dataloader_is_reproducible_with_same_seed_zero_workers():
    dataset_a = SeedableToyDataset(length=4)
    dataset_b = SeedableToyDataset(length=4)

    loader_a = WorkerSeedingDataLoader(
        dataset_a,
        seed=123,
        batch_size=4,
        num_workers=0,
        collate_fn=metadata_padding_collate_fn,
    )
    loader_b = WorkerSeedingDataLoader(
        dataset_b,
        seed=123,
        batch_size=4,
        num_workers=0,
        collate_fn=metadata_padding_collate_fn,
    )

    _, metadata_a = next(iter(loader_a))
    _, metadata_b = next(iter(loader_b))

    assert torch.equal(metadata_a["value"], metadata_b["value"])


class FakeRandomGenerator:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_init_worker_seed_seeds_worker_dataset_with_worker_specific_seed():
    dataset = SeedableToyDataset()
    worker_dataset = Mock()

    loader = WorkerSeedingDataLoader(dataset, seed=123, batch_size=2)
    loader.random_generator = FakeRandomGenerator(0.42)

    with patch(
        "torchsig.utils.data_loading.get_worker_info",
        return_value=SimpleNamespace(dataset=worker_dataset),
    ):
        loader.init_worker_seed(worker_id=2)

    worker_dataset.seed.assert_called_once_with(129)


def test_init_worker_seed_uses_worker_id_to_change_seed():
    dataset = SeedableToyDataset()
    worker_dataset = Mock()

    loader = WorkerSeedingDataLoader(dataset, seed=123, batch_size=2)
    loader.random_generator = FakeRandomGenerator(0.42)

    with patch(
        "torchsig.utils.data_loading.get_worker_info",
        return_value=SimpleNamespace(dataset=worker_dataset),
    ):
        loader.init_worker_seed(worker_id=0)
        loader.init_worker_seed(worker_id=1)

    worker_dataset.seed.assert_any_call(43)
    worker_dataset.seed.assert_any_call(86)
