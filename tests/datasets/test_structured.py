"""Tests for structured HDF5 PyTorch materialization."""

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import torchsig.datasets.structured as structured_module
from torchsig.datasets import StructuredHDF5Dataset, materialize_structured_dataset


class _TupleDataset(Dataset):
    def __init__(self, length: int = 5) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return (
            np.full((2, 3), index, dtype=np.float32),
            {"class": np.int64(index), "valid": np.bool_(index % 2)},
        )


class _Iterable(IterableDataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __iter__(self):
        for index in range(self.length):
            yield {"x": np.array([index], dtype=np.float32), "y": np.int16(index)}


def _stack_tuple_batch(samples):
    return (
        torch.from_numpy(np.stack([sample[0] for sample in samples])),
        {
            "class": torch.tensor([sample[1]["class"] for sample in samples], dtype=torch.int64),
            "valid": torch.tensor([sample[1]["valid"] for sample in samples], dtype=torch.bool),
        },
    )


def _assert_tuple_sample(actual, index: int) -> None:
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], np.full((2, 3), index, dtype=np.float32))
    assert actual[0].dtype == np.dtype(np.float32)
    assert actual[1]["class"] == index
    assert actual[1]["class"].dtype == np.dtype(np.int64)
    assert actual[1]["valid"] == bool(index % 2)


def test_materializes_dataset_and_preserves_sample_structure(tmp_path) -> None:
    result = materialize_structured_dataset(
        _TupleDataset(),
        tmp_path,
        batch_size=2,
        progress=False,
        writer_kwargs={"compression": None},
    )
    try:
        assert isinstance(result, StructuredHDF5Dataset)
        assert len(result) == 5
        for index in range(5):
            _assert_tuple_sample(result[index], index)
    finally:
        result.close()


def test_consumes_already_collated_dataloader_without_using_field_count(tmp_path) -> None:
    loader = DataLoader(_TupleDataset(), batch_size=3, collate_fn=_stack_tuple_batch)

    result = materialize_structured_dataset(loader, tmp_path, progress=False)
    try:
        assert len(result) == 5
        assert isinstance(result[0], tuple)
        _assert_tuple_sample(result[4], 4)
    finally:
        result.close()


def test_consumes_identity_collated_dataloader(tmp_path) -> None:
    loader = DataLoader(_TupleDataset(), batch_size=2, collate_fn=lambda samples: samples)

    result = materialize_structured_dataset(loader, tmp_path, progress=False)
    try:
        assert len(result) == 5
        _assert_tuple_sample(result[3], 3)
    finally:
        result.close()


def test_dataset_source_supports_custom_collate_and_final_partial_batch(tmp_path) -> None:
    result = materialize_structured_dataset(
        _TupleDataset(7),
        tmp_path,
        batch_size=3,
        collate_fn=_stack_tuple_batch,
        progress=False,
    )
    try:
        assert len(result) == 7
        _assert_tuple_sample(result[6], 6)
    finally:
        result.close()


def test_iterable_dataset_requires_and_accepts_expected_length(tmp_path) -> None:
    source = _Iterable(4)
    with pytest.raises(ValueError, match="expected_length is required"):
        materialize_structured_dataset(source, tmp_path / "missing", progress=False)

    result = materialize_structured_dataset(
        source,
        tmp_path / "complete",
        batch_size=3,
        expected_length=4,
        progress=False,
    )
    try:
        assert len(result) == 4
        np.testing.assert_array_equal(result[3]["x"], [3])
    finally:
        result.close()


@pytest.mark.parametrize(
    ("source", "expected_length", "message"),
    [
        (_Iterable(2), 3, "produced 2 samples; expected 3"),
        (_Iterable(4), 3, "more than the expected 3 samples"),
    ],
)
def test_rejects_too_few_or_too_many_samples(tmp_path, source, expected_length, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        materialize_structured_dataset(
            source,
            tmp_path,
            batch_size=2,
            expected_length=expected_length,
            progress=False,
        )
    assert list(tmp_path.iterdir()) == []


def test_progress_can_be_enabled_or_disabled(tmp_path, capsys) -> None:
    disabled = materialize_structured_dataset(_TupleDataset(2), tmp_path / "disabled", progress=False)
    disabled.close()
    assert "Materializing structured dataset" not in capsys.readouterr().err

    enabled = materialize_structured_dataset(_TupleDataset(2), tmp_path / "enabled", progress=True)
    enabled.close()
    assert "Materializing structured dataset" in capsys.readouterr().err


def test_map_dataset_uses_index_batch_reader(tmp_path, monkeypatch) -> None:
    result = materialize_structured_dataset(_TupleDataset(), tmp_path, progress=False)
    calls = []
    original = result.reader.read_indices

    def record(indices):
        calls.append(indices)
        return original(indices)

    monkeypatch.setattr(result.reader, "read_indices", record)
    try:
        samples = result.__getitems__([1, 2, 3])
        assert calls == [[1, 2, 3]]
        assert len(samples) == 3
        shuffled = result.__getitems__([3, 0, 3])
        assert calls[-1] == [3, 0, 3]
        for sample, index in zip(shuffled, [3, 0, 3], strict=True):
            _assert_tuple_sample(sample, index)
    finally:
        result.close()


def test_existing_destination_requires_explicit_overwrite(tmp_path) -> None:
    root = tmp_path / "dataset"
    original = materialize_structured_dataset(_TupleDataset(2), root, progress=False)
    original.close()

    with pytest.raises(FileExistsError, match="destination already exists"):
        materialize_structured_dataset(_TupleDataset(3), root, progress=False)

    unchanged = StructuredHDF5Dataset(root)
    try:
        assert len(unchanged) == 2
        _assert_tuple_sample(unchanged[1], 1)
    finally:
        unchanged.close()


def test_overwrite_replaces_complete_or_incomplete_destination(tmp_path) -> None:
    for name, complete in (("complete", True), ("incomplete", False)):
        root = tmp_path / name
        root.mkdir()
        with h5py.File(root / "data.h5", "w") as handle:
            handle.attrs["complete"] = complete

        result = materialize_structured_dataset(_TupleDataset(3), root, progress=False, overwrite=True)
        try:
            assert len(result) == 3
            _assert_tuple_sample(result[2], 2)
        finally:
            result.close()


def test_schema_failure_leaves_no_destination_or_temporary_artifacts(tmp_path) -> None:
    class _InvalidDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"unsupported": object()}

    root = tmp_path / "dataset"
    with pytest.raises(TypeError, match="Unsupported value"):
        materialize_structured_dataset(_InvalidDataset(), root, progress=False)

    assert not root.exists()
    assert list(tmp_path.iterdir()) == []


def test_mid_write_failure_preserves_existing_dataset(tmp_path) -> None:
    root = tmp_path / "dataset"
    original = materialize_structured_dataset(_TupleDataset(2), root, progress=False)
    original.close()

    class _InvalidSecondSample(_TupleDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            if index == 1:
                return (np.zeros((4,), dtype=np.float32), sample[1])
            return sample

    with pytest.raises(ValueError, match="Shape"):
        materialize_structured_dataset(
            _InvalidSecondSample(3),
            root,
            batch_size=1,
            progress=False,
            overwrite=True,
        )

    unchanged = StructuredHDF5Dataset(root)
    try:
        assert len(unchanged) == 2
        _assert_tuple_sample(unchanged[1], 1)
    finally:
        unchanged.close()
    assert [path.name for path in tmp_path.iterdir()] == ["dataset"]


def test_publication_failure_restores_existing_dataset(tmp_path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    original = materialize_structured_dataset(_TupleDataset(2), root, progress=False)
    original.close()
    real_replace = structured_module.Path.replace

    def fail_staging_publication(source, destination):
        if str(source).endswith(".tmp") and destination == root:
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(structured_module.Path, "replace", fail_staging_publication)
    with pytest.raises(OSError, match="simulated publication failure"):
        materialize_structured_dataset(_TupleDataset(3), root, progress=False, overwrite=True)

    unchanged = StructuredHDF5Dataset(root)
    try:
        assert len(unchanged) == 2
        _assert_tuple_sample(unchanged[1], 1)
    finally:
        unchanged.close()
    assert [path.name for path in tmp_path.iterdir()] == ["dataset"]


def test_full_validation_records_success_metadata(tmp_path) -> None:
    result = materialize_structured_dataset(_TupleDataset(5), tmp_path, progress=False, validation="full")
    result.close()

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["validation_mode"] == "full"
        assert handle.attrs["validation_result"] == "passed"
        assert handle.attrs["validation_sample_count"] == 5
        assert handle.attrs["validation_rtol"] == 0.0
        assert handle.attrs["validation_atol"] == 0.0


def test_sampled_validation_uses_deterministic_indices(tmp_path, monkeypatch) -> None:
    seed = 13
    checked = []
    original_read = structured_module.StructuredHDF5Reader.read

    def record_read(reader, index):
        checked.append(index)
        return original_read(reader, index)

    monkeypatch.setattr(structured_module.StructuredHDF5Reader, "read", record_read)
    result = materialize_structured_dataset(
        _TupleDataset(12),
        tmp_path,
        progress=False,
        validation="sampled",
        validation_samples=4,
        validation_seed=seed,
    )
    result.close()

    expected = sorted(int(index) for index in np.random.default_rng(seed).choice(12, size=4, replace=False))
    assert checked == expected
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["validation_mode"] == "sampled"
        assert handle.attrs["validation_sample_count"] == 4
        assert handle.attrs["validation_seed"] == seed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda sample: (sample[0] + 1, sample[1]), r"sample 0, field \$\[0\]: values differ"),
        (lambda sample: (sample[0].astype(np.float64), sample[1]), r"sample 0: Dtype mismatch at \$\[0\]"),
        (lambda sample: (sample[0][:1], sample[1]), r"sample 0: Shape mismatch at \$\[0\]"),
        (lambda sample: [sample[0], sample[1]], r"sample 0: Container mismatch at \$: expected tuple"),
        (lambda sample: (sample[0], {"other": sample[1]["class"]}), r"sample 0: Mapping keys mismatch at \$\[1\]"),
    ],
)
def test_validation_reports_sample_and_field_failures(tmp_path, monkeypatch, mutation, message) -> None:
    original_read = structured_module.StructuredHDF5Reader.read

    def mutate_read(reader, index):
        sample = original_read(reader, index)
        return mutation(sample) if index == 0 else sample

    monkeypatch.setattr(structured_module.StructuredHDF5Reader, "read", mutate_read)
    with pytest.raises(ValueError, match=message):
        materialize_structured_dataset(_TupleDataset(2), tmp_path, progress=False, validation="full")

    assert list(tmp_path.iterdir()) == []


def test_validation_supports_nan_complex_values_and_configured_tolerance(tmp_path, monkeypatch) -> None:
    class _SpecialDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return np.array([np.nan + 1j * index, index + 2j], dtype=np.complex64)

    exact = materialize_structured_dataset(_SpecialDataset(), tmp_path / "exact", progress=False, validation="full")
    exact.close()

    original_read = structured_module.StructuredHDF5Reader.read

    def perturb_read(reader, index):
        sample = original_read(reader, index).copy()
        sample[1] += np.complex64(1e-4)
        return sample

    monkeypatch.setattr(structured_module.StructuredHDF5Reader, "read", perturb_read)
    tolerant = materialize_structured_dataset(
        _SpecialDataset(),
        tmp_path / "tolerant",
        progress=False,
        validation="full",
        validation_atol=1e-3,
    )
    tolerant.close()


def test_validation_failure_preserves_existing_output(tmp_path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    original = materialize_structured_dataset(_TupleDataset(2), root, progress=False)
    original.close()
    original_read = structured_module.StructuredHDF5Reader.read

    def corrupt_read(reader, index):
        sample = original_read(reader, index)
        return (sample[0] + 1, sample[1])

    monkeypatch.setattr(structured_module.StructuredHDF5Reader, "read", corrupt_read)
    with pytest.raises(ValueError, match="values differ"):
        materialize_structured_dataset(
            _TupleDataset(3),
            root,
            progress=False,
            overwrite=True,
            validation="full",
        )

    monkeypatch.setattr(structured_module.StructuredHDF5Reader, "read", original_read)
    unchanged = StructuredHDF5Dataset(root)
    try:
        assert len(unchanged) == 2
        _assert_tuple_sample(unchanged[1], 1)
    finally:
        unchanged.close()
    assert [path.name for path in tmp_path.iterdir()] == ["dataset"]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"validation": "invalid"}, ValueError, "validation must be"),
        ({"validation_samples": 0}, ValueError, "validation_samples"),
        ({"validation_seed": True}, TypeError, "validation_seed"),
        ({"validation_rtol": -1.0}, ValueError, "validation_rtol"),
        ({"validation_atol": np.nan}, ValueError, "validation_atol"),
    ],
)
def test_rejects_invalid_validation_configuration(tmp_path, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        materialize_structured_dataset(_TupleDataset(1), tmp_path, progress=False, **kwargs)
