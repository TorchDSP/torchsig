"""Tests for the central file-handler backend registry."""

import pytest

from torchsig.utils.file_handlers import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.registry import BackendSpec, FileHandlerRegistry, file_handler_registry


def test_builtin_registry_resolves_alias_and_writer_pair() -> None:
    spec = file_handler_registry.resolve("legacy_hdf5")

    assert spec.name == "hdf5"
    assert spec.reader is HDF5Reader
    assert spec.writer is HDF5Writer
    assert file_handler_registry.reader_for_writer(HDF5Writer) is HDF5Reader
    assert "max_batches_in_memory" in spec.accepted_writer_options


def test_registry_capabilities_are_queryable_and_enforced() -> None:
    assert file_handler_registry.require("homogeneous_hdf5", "batch_read").supports("batch_read")
    with pytest.raises(ValueError, match="does not support"):
        file_handler_registry.require("npy", "compression")


def test_registry_rejects_unknown_and_duplicate_names() -> None:
    registry = FileHandlerRegistry([BackendSpec("custom", HDF5Reader, HDF5Writer, aliases=("alias",))])

    with pytest.raises(ValueError, match="unknown"):
        registry.resolve("missing")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(BackendSpec("alias", HDF5Reader))


def test_third_party_backend_can_be_registered() -> None:
    registry = FileHandlerRegistry()
    spec = BackendSpec("third_party", HDF5Reader, capabilities=frozenset({"compression"}))

    registry.register(spec)

    assert registry.resolve("third_party") is spec
