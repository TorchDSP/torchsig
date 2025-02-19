"""Chirp Signal
"""
import numpy as np

def chirp(f0:float, f1:float, samples_per_symbol:int, phi:float=0) -> np.ndarray:
    """Creates chirp symbol.

    Args:
        f0 (float): Starting frequency for the chirp.
        f1 (float): Ending frequency for the chirp.
        samples_per_symbol (int): Number of samples for the chirp symbol
        phi (float, optional): Starting phase in radians. Defaults to 0.

    Returns:
        np.ndarray: Chirp waveform.
    """

    t = np.arange(0, samples_per_symbol)
    b = (f1 - f0) / (t[-1] - t[0])
    phase = 2 * np.pi * (f0 * t + 0.5 * b * t * t) # Linear FM
    phi *= np.pi / 180
    return np.exp(1j*(phase+phi))
