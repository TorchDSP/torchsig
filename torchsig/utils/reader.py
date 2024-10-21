"""SigMF reader
"""

from torchsig.utils.types import *
import numpy as np


def reader_from_sigmf(signal_file: SignalCapture) -> Signal:
    """
    Args:
        signal_file:

    Returns:
        signal_data: SignalData object with meta-data parsed from sigMF file

    """
    with open(signal_file.absolute_path, "rb") as file_object:
        file_object.seek(signal_file.byte_offset)
        item_type = signal_file.item_type
        data_type = (
            np.dtype(np.complex128) if signal_file.is_complex else np.dtype(np.float64)
        )
        return create_signal(
            data=create_signal_data(
                samples=np.frombuffer(file_object.read(signal_file.num_bytes))
                .astype(item_type)
                .view(data_type)
            ),
            metadata=[signal_file.meta],
        )
