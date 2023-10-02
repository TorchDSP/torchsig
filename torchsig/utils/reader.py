import pickle
import numpy as np

from torchsig.utils.types import SignalCapture, SignalData


def pickle_loads(instance):
    return pickle.loads(instance)


def reader_from_sigmf(signal_file: SignalCapture) -> SignalData:
    """
    Args:
        signal_file:

    Returns:
        signal_data: SignalData object with meta-data parsed from sigMF file

    """
    with open(signal_file.absolute_path, "rb") as file_object:
        file_object.seek(signal_file.byte_offset)
        return SignalData(
            data=file_object.read(signal_file.num_bytes),
            item_type=signal_file.item_type,
            data_type=np.dtype(np.complex128) if signal_file.is_complex else np.dtype(np.float64),
            signal_description=signal_file.signal_description,
        )
