"""
Utility functions for dealing with the Signal type
"""

from torchsig.utils.dsp import low_pass_iterative_design


from typing import List
import numpy as np

def check_signal_class(name: str, possible_names: List[str]) -> bool:
    """Check if the provided signal name matches any of the possible signal names.

    Args:
        name (str): The signal name to check.
        possible_names (List[str]): A list of possible signal names to compare against.

    Returns:
        bool: True if the signal name matches any of the possible names, otherwise False.
    
    """
    is_type_signal = [n in name for n in possible_names]
    return any(is_type_signal)

def random_limiting_filter_design(
        bandwidth: float, 
        sample_rate: float, 
        rng: np.random.Generator = np.random.default_rng(seed=None)
):
    """Design a coarse bandwidth limiting filter with randomized parameters
    using TorchSig's iterative designer function.
    
    Args:
        bandwidth (float): Occupied bandwidth for signal (Hz).
        sample_rate (float): Signal sampling rate (Hz).
        rng (np.random.Generator, optional): Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        np.array: filter taps of designed low pass filter.
    """

    # randomize the cutoff
    cutoff = rng.uniform(0.8*bandwidth/2,0.95*sample_rate/2)
    
    # calculate maximum transition bandwidth
    max_transition_bandwidth = sample_rate/2 - cutoff
    
    # transition bandwidth is randomized value less than max transition bandwidth
    transition_bandwidth = rng.uniform(0.5,1.5)*max_transition_bandwidth
    
    # design bandwidth-limiting filter
    lpf = low_pass_iterative_design(cutoff=cutoff,transition_bandwidth=transition_bandwidth,sample_rate=sample_rate)
    
    return lpf