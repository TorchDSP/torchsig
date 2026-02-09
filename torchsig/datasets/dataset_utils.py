"""Dataset Utilities"""

import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    frequency_shift,
    upconversion_anti_aliasing_filter,
)

# name of yaml file where dataset information will be written
dataset_yaml_name = "create_dataset_info.yaml"
# name of yaml file where dataset writing information will be written
writer_yaml_name = "writer_info.yaml"


def frequency_shift_signal(
    signal: Signal,
    center_freq_min: float,
    center_freq_max: float,
    sample_rate: float,
    frequency_max: float,
    frequency_min: float,
    random_generator: np.random.Generator | None = None,
) -> Signal:
    """Randomly shifts the frequency of a signal to a new center frequency and applies aliasing filters if necessary.

    Args:
        signal (Signal): The signal object to be frequency shifted.
        center_freq_min (float): Minimum center frequency for the random shift.
        center_freq_max (float): Maximum center frequency for the random shift.
        sample_rate (float): The sample rate of the signal.
        frequency_max (float): Maximum frequency limit for aliasing.
        frequency_min (float): Minimum frequency limit for aliasing.
        random_generator (np.random.Generator, optional): Random number generator for generating the random shift. Defaults to `np.random.default_rng()`.

    Returns:
        Signal: The frequency-shifted signal with updated metadata.
    """
    random_generator = np.random.default_rng(seed=None) if random_generator is None else random_generator

    # randomize the center frequency
    center_freq = random_generator.uniform(low=center_freq_min, high=center_freq_max)

    # frequency shift to center_freq
    signal.data = frequency_shift(signal.data, center_freq, sample_rate)

    # update center_freq field in metadata
    signal["center_freq"] = center_freq

    # calculate upper and lower frequency edges of signal
    upper_freq = signal.upper_freq
    lower_freq = signal.lower_freq

    # has aliasing occured due to the upconversion to the signal?
    if upper_freq > frequency_max or lower_freq < frequency_min:
        # apply an anti-aliasing filter to the signal to attenuate energy that
        # wrapped around -fs/2 or fs/2. additionally, due to the filtering the
        # bandwidth changed bandwidth, and therefore changed the center frequency,
        # so update the two metadata fields accordingly
        signal.data, signal["center_freq"], signal["bandwidth"] = (
            upconversion_anti_aliasing_filter(
                signal.data,
                signal["center_freq"],
                signal["bandwidth"],
                sample_rate,
                frequency_max,
                frequency_min,
            )
        )
    # do nothing

    # center frequency is now set, and therefore can be verified
    signal["center_freq_set"] = True

    return signal


def save_type(transforms: list, target_transforms: list):
    """Determines if the dataset will generate 'raw' IQ data, which means no transform and target transforms have been applied.

    Args:
        transforms (list): A list of transformations to be applied to the data.
        target_transforms (list): A list of target transformations.

    Returns:
        bool: `True` if no transformations are applied, indicating raw IQ data; otherwise `False`.
    """
    return len(transforms) == 0 and len(target_transforms) == 0
