"""Numba-accelerated kernels for performance-critical functional transforms.

Currently provides:
  * ``sampling_clock_impairments_numba_wrapper`` - drop-in replacement for
    ``torchsig.utils.dsp.sampling_clock_impairments`` (clock_drift / clock_jitter).
    Aims for bit-identical results and fixes the RNG ordering by generating
    random numbers in the same order as NumPy.
  * ``digital_agc_numba`` - the sequential AGC sample loop used by ``digital_agc``.

Importing this module requires numba; callers should fall back to the pure-NumPy
implementations if the import fails.
"""

import numpy as np
from numba import jit
from numba.types import complex64, float32


@jit(nopython=True, cache=True)
def partition_polyphase_numba(h, up_rate, taps_per_phase):
    """Numba version of partition_polyphase."""
    h_pfb = np.zeros((up_rate, taps_per_phase), dtype=np.float32)
    for phase in range(up_rate):
        tap_idx = phase
        for idx in range(taps_per_phase):
            if tap_idx < len(h):
                h_pfb[phase, idx] = h[tap_idx] * up_rate
            else:
                h_pfb[phase, idx] = 0.0
            tap_idx += up_rate
    return h_pfb


@jit(nopython=True, cache=True)
def sampling_clock_impairments_numba(
    h,
    x_real,
    x_imag,
    uprate,
    drate,
    jitter_ppm,
    drift_ppm,
    jitter_drift_pool,
    h_pfb_reversed,
    taps_per_phase,
    padded_len,
    max_start,
    num_output_samples,
):
    """Numba-optimized sampling clock impairments.
    Uses pre-reversed filter bank and pre-generated interleaved random number pools.
    """
    input_padded_real = np.zeros(padded_len, dtype=np.float32)
    input_padded_imag = np.zeros(padded_len, dtype=np.float32)

    start = taps_per_phase - 1
    end = start + len(x_real)

    input_padded_real[start:end] = x_real
    input_padded_imag[start:end] = x_imag

    q_step = uprate / drate

    output_real = np.zeros(num_output_samples, dtype=np.float32)
    output_imag = np.zeros(num_output_samples, dtype=np.float32)

    output_idx = 0
    input_idx = 0
    clock_drift = 0.0

    while input_idx <= max_start:
        while q_step >= uprate:
            q_step -= uprate
            input_idx += 1

        if input_idx > max_start:
            break

        phase = int(q_step)

        acc_re = 0.0
        acc_im = 0.0

        for i in range(taps_per_phase):
            acc_re += h_pfb_reversed[phase, i] * input_padded_real[input_idx + i]
            acc_im += h_pfb_reversed[phase, i] * input_padded_imag[input_idx + i]

        output_real[output_idx] = acc_re
        output_imag[output_idx] = acc_im
        output_idx += 1

        if jitter_ppm != 0.0 or drift_ppm != 0.0:
            pool_index = (output_idx - 1) * 2
            clock_jitter = jitter_drift_pool[pool_index]
            clock_drift += jitter_drift_pool[pool_index + 1]
            q_step += drate + clock_jitter + clock_drift
        else:
            q_step += drate

    if output_idx > 0:
        result = np.zeros(output_idx, dtype=np.complex64)
        for i in range(output_idx):
            result[i] = output_real[i] + 1j * output_imag[i]
        return result

    return np.zeros(0, dtype=np.complex64)


def sampling_clock_impairments_numba_wrapper(h, x, uprate, drate, jitter_ppm, drift_ppm, rng):
    """Wrapper for the numba-optimized sampling clock impairments function.

    Matches the signature of the original function and aims for bit-identical results.
    Generates interleaved jitter/drift pairs to match NumPy RNG consumption order.
    """
    taps_per_phase = int(np.ceil(len(h) / uprate))

    h_pfb = partition_polyphase_numba(h, uprate, taps_per_phase)
    h_pfb_reversed = np.ascontiguousarray(np.flip(h_pfb, axis=1))

    padded_len = len(x) + 2 * taps_per_phase - 1
    max_start = padded_len - taps_per_phase

    num_output_samples = int(np.ceil(padded_len * uprate / drate)) + 1

    if jitter_ppm != 0.0 or drift_ppm != 0.0:
        jitter_std = jitter_ppm * 1e-6
        drift_std = drift_ppm * 1e-6

        pairs = rng.normal(0.0, 1.0, (num_output_samples, 2)).astype(np.float32)

        jitter_drift_pool = np.empty(num_output_samples * 2, dtype=np.float32)
        jitter_drift_pool[0::2] = pairs[:, 0] * jitter_std
        jitter_drift_pool[1::2] = pairs[:, 1] * drift_std
    else:
        jitter_drift_pool = np.zeros(num_output_samples * 2, dtype=np.float32)

    x_real = x.real.astype(np.float32)
    x_imag = x.imag.astype(np.float32)

    return sampling_clock_impairments_numba(
        h,
        x_real,
        x_imag,
        uprate,
        drate,
        jitter_ppm,
        drift_ppm,
        jitter_drift_pool,
        h_pfb_reversed,
        taps_per_phase,
        padded_len,
        max_start,
        num_output_samples,
    )


@jit(nopython=True, cache=True)
def digital_agc_numba(
    data: complex64[:],  # 1D complex64 array (Numba type)
    initial_gain_db: float32,  # All scalars must be Numba types
    alpha_smooth: float32,
    alpha_track: float32,
    alpha_overflow: float32,
    alpha_acquire: float32,
    ref_level_db: float32,
    track_range_db: float32,
    low_level_db: float32,
    high_level_db: float32,
):
    """Numba version of the digital AGC sample-by-sample loop."""
    n = len(data)
    output = np.empty(n, dtype=np.complex64)  # Pre-allocate (faster than zeros_like)
    gain_db = initial_gain_db
    level_db = 0.0

    for sample_idx in range(n):
        sample = data[sample_idx]
        mag = np.abs(sample)  # MUST use np.abs (not built-in abs)
        if mag == 0.0:
            level_db = -200.0
        elif sample_idx == 0:
            level_db = np.log(mag)
        else:
            level_db = level_db * alpha_smooth + np.log(mag) * (1 - alpha_smooth)

        output_db = level_db + gain_db
        diff_db = ref_level_db - output_db

        if level_db <= low_level_db:
            alpha_adjust = 0.0
        elif output_db >= high_level_db:
            alpha_adjust = alpha_overflow
        elif np.abs(diff_db) > track_range_db:  # MUST use np.abs
            alpha_adjust = alpha_acquire
        else:
            alpha_adjust = alpha_track

        gain_db += diff_db * alpha_adjust
        output[sample_idx] = sample * np.exp(gain_db)

    return output
