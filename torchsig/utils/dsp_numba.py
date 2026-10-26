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

from torchsig.utils.dsp import generate_sampling_positions

_OUTPUT_CAPACITY_ERROR = "sampling clock output capacity exhausted"
_MAX_OUTPUT_CAPACITY_MULTIPLIER = 16
_MAX_OUTPUT_SAMPLES = 10_000_000


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
def polyphase_resample_at_positions_numba(x_real, x_imag, uprate, positions, h_pfb_reversed, taps_per_phase):
    """Numba interpolation kernel for precomputed sampling positions."""
    padded_len = len(x_real) + 2 * taps_per_phase - 1
    input_padded_real = np.zeros(padded_len, dtype=np.float32)
    input_padded_imag = np.zeros(padded_len, dtype=np.float32)
    input_start = taps_per_phase - 1
    input_padded_real[input_start : input_start + len(x_real)] = x_real
    input_padded_imag[input_start : input_start + len(x_imag)] = x_imag
    result = np.empty(len(positions), dtype=np.complex64)
    for output_idx, sample_position in enumerate(positions):
        input_idx = int(sample_position // uprate)
        phase_position = sample_position - input_idx * uprate
        phase = int(phase_position)
        fraction = phase_position - phase
        acc_re = 0.0
        acc_im = 0.0
        for tap_idx in range(taps_per_phase):
            coefficient = h_pfb_reversed[phase, tap_idx]
            input_position = input_idx + tap_idx
            acc_re += coefficient * input_padded_real[input_position]
            acc_im += coefficient * input_padded_imag[input_position]
        if fraction != 0.0:
            next_phase = phase + 1
            next_input_idx = input_idx
            if next_phase == uprate:
                next_phase = 0
                next_input_idx += 1
            next_re = 0.0
            next_im = 0.0
            for tap_idx in range(taps_per_phase):
                coefficient = h_pfb_reversed[next_phase, tap_idx]
                input_position = next_input_idx + tap_idx
                next_re += coefficient * input_padded_real[input_position]
                next_im += coefficient * input_padded_imag[input_position]
            acc_re += fraction * (next_re - acc_re)
            acc_im += fraction * (next_im - acc_im)
        result[output_idx] = acc_re + 1j * acc_im
    return result


@jit(nopython=True, cache=True)
def sampling_clock_impairments_numba(  # noqa: PLR0917
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
    max_input_idx,
    nominal_position_increment,
    num_output_samples,
    initial_phase,
):
    """Apply sampling-clock impairments with precomputed filter and RNG data.

    The wrapper prepares contiguous real and imaginary inputs, a reversed
    polyphase filter bank, and interleaved jitter/drift samples for this kernel.
    """
    input_padded_real = np.zeros(padded_len, dtype=np.float32)
    input_padded_imag = np.zeros(padded_len, dtype=np.float32)

    input_start = taps_per_phase - 1
    input_end = input_start + len(x_real)

    input_padded_real[input_start:input_end] = x_real
    input_padded_imag[input_start:input_end] = x_imag

    # Track the ideal sampling path in absolute polyphase units. Timing
    # impairments are accumulated separately and applied only to each read.
    nominal_position = uprate / drate + initial_phase * uprate
    position_offset = 0.0
    max_sample_position = max_input_idx * uprate + (uprate - 1)

    output_real = np.zeros(num_output_samples, dtype=np.float32)
    output_imag = np.zeros(num_output_samples, dtype=np.float32)

    output_idx = 0
    iteration_idx = 0
    clock_drift = 0.0

    while nominal_position <= max_sample_position:
        if jitter_ppm != 0.0 or drift_ppm != 0.0:
            pool_index = iteration_idx * 2
            if (pool_index + 1) >= len(jitter_drift_pool):
                raise RuntimeError(_OUTPUT_CAPACITY_ERROR)

        sample_position = nominal_position + position_offset
        if 0.0 <= sample_position <= max_sample_position:
            if output_idx >= num_output_samples:
                raise RuntimeError(_OUTPUT_CAPACITY_ERROR)

            input_idx = int(sample_position // uprate)
            phase_position = sample_position - input_idx * uprate
            phase = int(phase_position)

            acc_re = 0.0
            acc_im = 0.0

            for tap_idx in range(taps_per_phase):
                coefficient = h_pfb_reversed[phase, tap_idx]
                input_position = input_idx + tap_idx
                acc_re += coefficient * input_padded_real[input_position]
                acc_im += coefficient * input_padded_imag[input_position]

            output_real[output_idx] = acc_re
            output_imag[output_idx] = acc_im
            output_idx += 1

        if jitter_ppm != 0.0 or drift_ppm != 0.0:
            clock_jitter = jitter_drift_pool[pool_index]
            clock_drift += jitter_drift_pool[pool_index + 1]
            position_offset += clock_jitter + clock_drift

        iteration_idx += 1
        nominal_position += nominal_position_increment

    result = np.zeros(output_idx, dtype=np.complex64)
    for idx in range(output_idx):
        result[idx] = output_real[idx] + 1j * output_imag[idx]

    return result


def sampling_clock_impairments_numba_wrapper(  # noqa: PLR0917
    h,
    x,
    uprate,
    drate,
    jitter_ppm,
    drift_ppm,
    rng,
    initial_phase=0.0,
    jitter_model="independent",
    jitter_alpha=0.9,
    drift_model="constant",
    drift_rate_ppm_per_sec=0.0,
    drift_noise_ppm_per_sqrt_sec=0.0,
    sample_rate_hz=1.0,
):
    """Wrapper for the numba-optimized sampling clock impairments function.

    Matches the signature of the original function and aims for bit-identical results.
    Generates positions with the shared clock model before Numba interpolation.
    """
    # Construct the polyphase filter bank.
    if not isinstance(uprate, (int, np.integer)) or uprate <= 0:
        raise ValueError("uprate must be a positive integer")
    if not np.isfinite(initial_phase) or not 0.0 <= initial_phase < 1.0:
        raise ValueError("initial_phase must be finite and in the interval [0, 1)")
    taps_per_phase = int(np.ceil(len(h) / uprate))

    nominal_increment = drate
    if np.isfinite(nominal_increment) and nominal_increment > 0.0:
        padded_len = len(x) + 2 * taps_per_phase - 1
        max_position = (padded_len - taps_per_phase) * uprate + (uprate - 1)
        first_position = uprate / drate + initial_phase * uprate
        nominal_count = 0 if first_position > max_position else int(np.floor((max_position - first_position) / nominal_increment)) + 1
        if nominal_count > _MAX_OUTPUT_SAMPLES:
            raise RuntimeError(_OUTPUT_CAPACITY_ERROR)

    h_pfb = partition_polyphase_numba(h, uprate, taps_per_phase)
    h_pfb_reversed = np.ascontiguousarray(
        np.flip(h_pfb, axis=1),
        dtype=np.float32,
    )

    positions = generate_sampling_positions(
        num_input_samples=len(x),
        taps_per_phase=taps_per_phase,
        uprate=uprate,
        drate=drate,
        jitter_ppm=jitter_ppm,
        drift_ppm=drift_ppm,
        rng=rng,
        initial_phase=initial_phase,
        jitter_model=jitter_model,
        jitter_alpha=jitter_alpha,
        drift_model=drift_model,
        drift_rate_ppm_per_sec=drift_rate_ppm_per_sec,
        drift_noise_ppm_per_sqrt_sec=drift_noise_ppm_per_sqrt_sec,
        sample_rate_hz=sample_rate_hz,
    )

    x_real = np.ascontiguousarray(x.real, dtype=np.float32)
    x_imag = np.ascontiguousarray(x.imag, dtype=np.float32)

    return polyphase_resample_at_positions_numba(x_real, x_imag, uprate, positions.actual, h_pfb_reversed, taps_per_phase)


@jit(nopython=True, cache=True)
def digital_agc_numba(  # noqa: PLR0917
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
