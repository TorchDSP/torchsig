import numpy as np


def estimate_filter_length(
    transition_bandwidth: float, attenuation_db: int = 72, sample_rate: float = 1.0
) -> int:
    # estimate the length of an FIR filter using harris' approximaion,
    # N ~= (sampling rate/transition bandwidth)*(sidelobe attenuation in dB / 22)
    # fred harris, Multirate Signal Processing for Communication Systems,
    # Second Edition, p.59
    filter_length = int(
        np.round((sample_rate / transition_bandwidth) * (attenuation_db / 22))
    )

    # odd-length filters are desirable because they do not introduce a half-sample delay
    if np.mod(filter_length, 2) == 0:
        filter_length += 1

    return filter_length
