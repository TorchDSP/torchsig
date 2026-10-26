"""Frequency-response contract for the sampling-clock prototype filter."""

import numpy as np
from scipy import signal as sp

from torchsig.utils.dsp import (
    SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH,
    TorchSigRealDataType,
    prototype_polyphase_filter,
)


def test_sampling_clock_prototype_meets_documented_bandwidth_contract() -> None:
    uprate = 5000
    # Match the coefficient conversion performed by the public transforms.
    taps = prototype_polyphase_filter(uprate).astype(TorchSigRealDataType)
    passband_edge = SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH / 2
    frequencies = np.array([0.0, passband_edge, 0.7])

    _, response = sp.freqz(taps, worN=2 * np.pi * frequencies / uprate)
    response_db = 20 * np.log10(np.abs(response) / np.abs(response[0]))

    assert abs(response_db[1]) < 0.05
    assert response_db[2] < -80.0


def test_sampling_clock_float32_conversion_preserves_contract_response() -> None:
    uprate = 5000
    float64_taps = prototype_polyphase_filter(uprate).astype(np.float64)
    float32_taps = float64_taps.astype(TorchSigRealDataType)
    frequencies = np.array([0.0, SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH / 2, 0.7])

    responses = []
    for taps in (float64_taps, float32_taps):
        _, response = sp.freqz(taps, worN=2 * np.pi * frequencies / uprate)
        responses.append(20 * np.log10(np.abs(response) / np.abs(response[0])))

    np.testing.assert_allclose(responses[1], responses[0], rtol=0.0, atol=1e-3)
    assert float32_taps.nbytes * 2 == float64_taps.nbytes


def test_sampling_clock_bandwidth_contract_is_full_two_sided_bandwidth() -> None:
    assert SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH == 2 * 0.35
