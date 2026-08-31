"""Tests for standard HDF5 writer options."""

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5 import HDF5Writer


def test_standard_hdf5_writer_supports_no_compression(tmp_path) -> None:
    """Record disabled compression without passing None to HDF5 attributes."""
    signals = [Signal(data=np.ones(4, dtype=np.complex64))]

    with HDF5Writer(
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
    ) as writer:
        writer.write(0, signals)

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data"]["0"].compression is None
