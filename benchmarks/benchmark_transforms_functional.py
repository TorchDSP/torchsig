import numpy as np
import pytest
from scipy import signal as sp

# Import all functions to benchmark
from torchsig.transforms.functional import (
    add_slope,
    additive_noise,
    adjacent_channel_interference,
    awgn,
    carrier_frequency_drift,
    carrier_phase_noise,
    channel_swap,
    clock_drift,
    clock_jitter,
    coarse_gain_change,
    cochannel_interference,
    complex_to_2d,
    cut_out,
    digital_agc,
    doppler,
    drop_samples,
    fading,
    interleave_complex,
    intermodulation_products,
    iq_imbalance,
    nonlinear_amplifier,
    nonlinear_amplifier_table,
    normalize,
    passband_ripple,
    patch_shuffle,
    phase_offset,
    quantize,
    shadowing,
    spectral_inversion,
    spectrogram,
    spectrogram_drop_samples,
    spectrogram_image,
    spurs,
    time_reversal,
    time_varying_noise,
)

SAMPLE_SIZE = 1024
SAMPLE_SIZE_LONG = 10240
SEED = 42


# Fixtures for test data
@pytest.fixture
def sample_data():
    """Generate sample complex IQ data."""
    np.random.seed(SEED)
    return (np.random.randn(SAMPLE_SIZE) + 1j * np.random.randn(SAMPLE_SIZE)).astype(np.complex64)


@pytest.fixture
def real_sample_data():
    """Generate sample real data for benchmarking."""
    np.random.seed(SEED)
    return np.random.randn(SAMPLE_SIZE).astype(np.float32)


@pytest.fixture
def long_sample_data():
    """Generate longer sample complex IQ data."""
    np.random.seed(SEED)
    return (np.random.randn(SAMPLE_SIZE_LONG) + 1j * np.random.randn(SAMPLE_SIZE_LONG)).astype(np.complex64)


@pytest.fixture
def spectrogram_data():
    """Generate sample data for spectrogram drop functions.
    The function spectrogram_drop_samples expects the 1st dimension to be 2
    (Real and Imaginary channels).
    """
    np.random.seed(SEED)
    return (np.random.randn(2, 128, 128) + 1j * np.random.randn(2, 128, 128)).astype(np.complex64)


@pytest.fixture
def rng():
    """Create a random number generator for benchmarking."""
    return np.random.default_rng(SEED)


class TestTransformBenchmarks:
    """Benchmark tests for all transform functions."""

    def test_benchmark_add_slope(self, benchmark, sample_data):
        result = benchmark(add_slope, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_additive_noise(self, benchmark, sample_data, rng):
        result = benchmark(additive_noise, sample_data, 1.0, "white", True, rng)
        assert result.dtype == np.complex64

    def test_benchmark_adjacent_channel_interference(self, benchmark, sample_data, rng):
        filter_weights = sp.firwin(101, 0.25).astype(np.float32)
        result = benchmark(adjacent_channel_interference, sample_data, 4.0, 1.0, 0.2, filter_weights, 1.0, 0.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_awgn(self, benchmark, sample_data, rng):
        result = benchmark(awgn, sample_data, -10.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_carrier_frequency_drift(self, benchmark, sample_data, rng):
        result = benchmark(carrier_frequency_drift, sample_data, 1.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_carrier_phase_noise(self, benchmark, sample_data, rng):
        result = benchmark(carrier_phase_noise, sample_data, 1.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_channel_swap(self, benchmark, sample_data):
        result = benchmark(channel_swap, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_clock_drift(self, benchmark, sample_data, rng):
        result = benchmark(clock_drift, sample_data, 10.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_clock_jitter(self, benchmark, sample_data, rng):
        result = benchmark(clock_jitter, sample_data, 10.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_coarse_gain_change(self, benchmark, sample_data):
        result = benchmark(coarse_gain_change, sample_data, 3.0, 512)
        assert result.dtype == np.complex64

    def test_benchmark_cochannel_interference(self, benchmark, sample_data, rng):
        filter_weights = sp.firwin(101, 0.25).astype(np.float32)
        result = benchmark(cochannel_interference, sample_data, 1.0, filter_weights, "white", True, rng)
        assert result.dtype == np.complex64

    def test_benchmark_complex_to_2d(self, benchmark, sample_data):
        result = benchmark(complex_to_2d, sample_data)
        assert result.shape[0] == 2

    def test_benchmark_cut_out(self, benchmark, sample_data, rng):
        result = benchmark(cut_out, sample_data, 0.2, 0.1, "zeros", rng)
        assert result.dtype == np.complex64

    def test_benchmark_digital_agc(self, benchmark, sample_data):
        result = benchmark(digital_agc, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_doppler(self, benchmark, sample_data):
        result = benchmark(doppler, sample_data, 1e1)
        assert result.dtype == np.complex64

    def test_benchmark_drop_samples(self, benchmark, sample_data):
        drop_starts = np.array([100, 500])
        drop_sizes = np.array([50, 30])
        result = benchmark(drop_samples, sample_data, drop_starts, drop_sizes, "ffill")
        assert result.dtype == np.complex64

    def test_benchmark_fading(self, benchmark, long_sample_data, rng):
        power_delay_profile = np.ones(32).astype(np.float32) / 32
        result = benchmark(fading, long_sample_data, 0.1, power_delay_profile, rng)
        assert result.dtype == np.complex64

    def test_benchmark_interleave_complex(self, benchmark, sample_data):
        result = benchmark(interleave_complex, sample_data)
        assert result.dtype == np.float32

    def test_benchmark_intermodulation_products(self, benchmark, sample_data):
        result = benchmark(intermodulation_products, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_iq_imbalance(self, benchmark, sample_data):
        result = benchmark(iq_imbalance, sample_data, 1.0, 0.1, -40, 0.0, -50)
        assert result.dtype == np.complex64

    def test_benchmark_nonlinear_amplifier(self, benchmark, sample_data):
        result = benchmark(nonlinear_amplifier, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_nonlinear_amplifier_table(self, benchmark, sample_data):
        result = benchmark(nonlinear_amplifier_table, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_normalize(self, benchmark, sample_data):
        result = benchmark(normalize, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_passband_ripple(self, benchmark, sample_data, rng):
        result = benchmark(passband_ripple, sample_data, num_taps=5, max_ripple_db=1.0, rng=rng)
        assert result.dtype == np.complex64

    def test_benchmark_patch_shuffle(self, benchmark, sample_data, rng):
        patch_size = 64
        patches_to_shuffle = np.array([0, 1, 2])
        result = benchmark(patch_shuffle, sample_data, patch_size, patches_to_shuffle, rng)
        assert result.dtype == np.complex64

    def test_benchmark_phase_offset(self, benchmark, sample_data):
        result = benchmark(phase_offset, sample_data, np.pi / 4)
        assert result.dtype == np.complex64

    def test_benchmark_quantize(self, benchmark, sample_data):
        result = benchmark(quantize, sample_data, 8)
        assert result.dtype == np.complex64

    def test_benchmark_shadowing(self, benchmark, sample_data, rng):
        result = benchmark(shadowing, sample_data, 4.0, 2.0, rng)
        assert result.dtype == np.complex64

    def test_benchmark_spectral_inversion(self, benchmark, sample_data):
        result = benchmark(spectral_inversion, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_spectrogram(self, benchmark, long_sample_data):
        result = benchmark(spectrogram, long_sample_data, 256, 128)
        assert np.issubdtype(result.dtype, np.floating)

    def test_benchmark_spectrogram_drop_samples(self, benchmark, spectrogram_data):
        drop_starts = np.array([10, 50])
        drop_sizes = np.array([5, 3])
        result = benchmark(spectrogram_drop_samples, spectrogram_data, drop_starts, drop_sizes, "ffill")
        assert result.dtype == np.complex64

    def test_benchmark_spectrogram_image(self, benchmark, long_sample_data):
        result = benchmark(spectrogram_image, long_sample_data, 256, 128)
        assert result.dtype == np.uint8

    def test_benchmark_spurs(self, benchmark, long_sample_data):
        center_freqs = [0.1, 0.2]
        relative_power_db = [10, 5]
        result = benchmark(spurs, long_sample_data, 1.0, center_freqs, relative_power_db)
        assert result.dtype == np.complex64

    def test_benchmark_time_reversal(self, benchmark, sample_data):
        result = benchmark(time_reversal, sample_data)
        assert result.dtype == np.complex64

    def test_benchmark_time_varying_noise(self, benchmark, long_sample_data, rng):
        result = benchmark(time_varying_noise, long_sample_data, -20.0, -10.0, 5, True, rng)
        assert result.dtype == np.complex64


class TestTransformBenchmarksWithSizes:
    """Benchmark transforms with different input sizes."""

    @pytest.mark.parametrize("size", [256, 1024, 4096, 16384])
    def test_benchmark_add_slope_various_sizes(self, benchmark, size):
        data = (np.random.randn(size) + 1j * np.random.randn(size)).astype(np.complex64)
        result = benchmark(add_slope, data)
        assert result.dtype == np.complex64

    @pytest.mark.parametrize("size", [256, 1024, 4096])
    def test_benchmark_fading_various_sizes(self, benchmark, size, rng):
        data = (np.random.randn(size) + 1j * np.random.randn(size)).astype(np.complex64)
        power_delay_profile = np.ones(16).astype(np.float32) / 16
        result = benchmark(fading, data, 0.1, power_delay_profile, rng)
        assert result.dtype == np.complex64

    @pytest.mark.parametrize("fft_size,fft_stride", [(64, 32), (256, 128), (512, 256)])
    def test_benchmark_spectrogram_various_sizes(self, benchmark, fft_size, fft_stride):
        data = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        result = benchmark(spectrogram, data, fft_size, fft_stride)
        assert np.issubdtype(result.dtype, np.floating)
