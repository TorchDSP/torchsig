"""Utility functions for transforms testing."""

from torchsig.utils.defaults import default_dataset



def generate_test_signal(num_iq_samples: int = 10, scale: float = 1.0):
    """Generate a scaled, high SNR baseband QPSK Signal.

    Args:
    num_iq_samples (int, optional): Length of sample. Defaults to 10.
    scale (float, optional): scale normalized signal data. Defaults to 1.0.

    Returns:
        signal: generated Signal.

    """
    dataset = default_dataset(
        signal_generators=["qpsk"], num_signals_min=1, num_signals_max=1, signal_duration_in_samples_min=num_iq_samples, signal_duration_in_samples_max=num_iq_samples, start_in_samples=0, seed=67
    )
    signal = dataset.signal_generators[0]()
    signal.data = signal.data * scale
    return signal


def generate_tone_signal(num_iq_samples: int = 10, scale: float = 1.0):
    """Generate a scaled, high SNR baseband tone Signal.

    Args:
    num_iq_samples (int, optional): Length of sample. Defaults to 10.
    scale (int, optional): scale normalized signal data. Defaults to 1.0.

    Returns:
        signal: generated Signal.

    """
    dataset = default_dataset(
        signal_generators=["tone"], num_signals_min=1, num_signals_max=1, signal_duration_in_samples_min=num_iq_samples, signal_duration_in_samples_max=num_iq_samples, start_in_samples=0, seed=67
    )
    signal = dataset.signal_generators[0]()
    signal.data = signal.data * scale
    return signal
