"""Utility functions for transforms testing."""
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.defaults import default_dataset
from torchsig.utils.dsp import TorchSigComplexDataType


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


def generate_composite_signal(num_iq_samples: int = 128, num_components: int = 2) -> Signal:
    """Build a specifically parameterized container Signal wrapping N component signals over a shared timeline."""
    components = []
    for n in range(num_components):
        comp = generate_test_signal(num_iq_samples=num_iq_samples, scale=1.0)
        # give each component a distinct, non-default bbox so updates are observable
        comp["num_iq_samples_dataset"] = num_iq_samples
        comp["start_in_samples"] = int(num_iq_samples * (0.1 + 0.3 * n))
        comp["duration_in_samples"] = int(num_iq_samples * 0.2)
        comp["center_freq"] = 0.05 + 0.1 * n
        comp["bandwidth"] = 0.04
        components.append(comp)

    container = Signal(
        data=np.zeros(num_iq_samples, dtype=TorchSigComplexDataType),
        component_signals=components,
        num_iq_samples_dataset=num_iq_samples,
        start_in_samples=0,
        duration_in_samples=num_iq_samples,
        center_freq=0.0,
        bandwidth=1.0,
    )
    # Put nonzero IQ on the container so data-level transforms are observable.
    container.data = (
        np.arange(num_iq_samples, dtype=np.float32)
        + 1j * np.arange(num_iq_samples, dtype=np.float32)[::-1]
    ).astype(TorchSigComplexDataType)
    return container


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
