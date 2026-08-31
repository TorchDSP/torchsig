"""Unit tests for the Chirp Spread Spectrum signal generator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.chirpss import (
    ChirpSSSignalGenerator,
    chirpss_modulator,
    chirpss_modulator_baseband,
    get_symbol_map,
)
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.chirpss"


def test_get_symbol_map_returns_expected_values():
    """The symbol map should contain integers zero through 127."""
    result = get_symbol_map()

    expected = np.arange(128, dtype=np.float64)

    np.testing.assert_array_equal(result, expected)


def test_get_symbol_map_returns_128_symbols():
    """The ChirpSS constellation should contain 128 symbols."""
    result = get_symbol_map()

    assert result.shape == (128,)
    assert result.size == 128


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_chirpss_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        chirpss_modulator_baseband(
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -10])
def test_chirpss_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_chirpss_modulator_baseband_creates_default_rng():
    """A default NumPy generator should be created when rng is omitted."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.ones(128, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.chirp",
            return_value=upchirp,
        ),
    ):
        result = chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (128,)


def test_chirpss_modulator_baseband_generates_expected_symbol():
    """A symbol index of zero should use the beginning of the chirp."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(128, dtype=np.float32).astype(TorchSigComplexDataType)

    with patch(
        f"{MODULE_PATH}.chirp",
        return_value=upchirp,
    ) as chirp_mock:
        result = chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    chirp_mock.assert_called_once_with(
        -0.25,
        0.25,
        128,
    )

    np.testing.assert_array_equal(result, upchirp)


def test_chirpss_modulator_baseband_offsets_chirp_by_symbol_value():
    """The selected symbol should cyclically offset the upchirp."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([64]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(128, dtype=np.float32).astype(TorchSigComplexDataType)

    with patch(
        f"{MODULE_PATH}.chirp",
        return_value=upchirp,
    ):
        result = chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    double_upchirp = np.concatenate((upchirp, upchirp))

    # Symbol 64 of 128 starts halfway through the chirp.
    expected = double_upchirp[64:192]

    np.testing.assert_array_equal(result, expected)


def test_chirpss_modulator_baseband_truncates_final_symbol():
    """A final symbol should be truncated to the requested output length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0, 0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(128, dtype=np.float32).astype(TorchSigComplexDataType)

    with patch(
        f"{MODULE_PATH}.chirp",
        return_value=upchirp,
    ):
        result = chirpss_modulator_baseband(
            max_num_samples=150,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    expected = np.concatenate(
        (
            upchirp,
            upchirp[:22],
        )
    )

    assert result.shape == (150,)
    np.testing.assert_array_equal(result, expected)


def test_chirpss_modulator_baseband_requests_correct_symbol_count():
    """Enough symbols should be generated to fill the output buffer."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0, 1, 2]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.ones(128, dtype=np.complex64)

    with patch(
        f"{MODULE_PATH}.chirp",
        return_value=upchirp,
    ):
        chirpss_modulator_baseband(
            max_num_samples=300,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert rng.integers.call_args_list == [
        call(low=128, high=4096),
        call(0, 128, 3),
    ]


def test_chirpss_modulator_baseband_applies_filter_when_selected():
    """The bandwidth-limiting filter should be applied below probability 0.5."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0]),
    ]
    rng.uniform.return_value = 0.25

    upchirp = np.ones(128, dtype=np.complex64)
    limiting_filter = np.array([0.25, 0.5, 0.25])
    filtered = np.full(128, 2 + 1j, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirp",
            return_value=upchirp,
        ),
        patch(
            f"{MODULE_PATH}.random_limiting_filter_design",
            return_value=limiting_filter,
        ) as filter_design,
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=filtered,
        ) as convolve_mock,
    ):
        result = chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    filter_design.assert_called_once_with(
        0.25,
        1.0,
        rng,
    )
    convolve_mock.assert_called_once()

    np.testing.assert_array_equal(result, filtered)


def test_chirpss_modulator_baseband_skips_filter_when_not_selected():
    """No limiting filter should be designed when probability is at least 0.5."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0]),
    ]
    rng.uniform.return_value = 0.75

    upchirp = np.ones(128, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirp",
            return_value=upchirp,
        ),
        patch(
            f"{MODULE_PATH}.random_limiting_filter_design",
        ) as filter_design,
        patch(
            f"{MODULE_PATH}.convolve",
        ) as convolve_mock,
    ):
        result = chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    filter_design.assert_not_called()
    convolve_mock.assert_not_called()

    np.testing.assert_array_equal(result, upchirp)


def test_chirpss_modulator_baseband_returns_complex_dtype():
    """The unfiltered baseband result should use the TorchSig complex dtype."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0]),
    ]
    rng.uniform.return_value = 1.0

    with patch(
        f"{MODULE_PATH}.chirp",
        return_value=np.ones(128, dtype=np.complex64),
    ):
        result = chirpss_modulator_baseband(
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "num_samples", "expected_message"),
    [
        (0, 10_000, 128, "bandwidth must be positive"),
        (-1, 10_000, 128, "bandwidth must be positive"),
        (1_000, 0, 128, "sample_rate must be positive"),
        (1_000, -1, 128, "sample_rate must be positive"),
        (
            5_001,
            10_000,
            128,
            "bandwidth must be less than sample_rate/2",
        ),
        (1_000, 10_000, 0, "num_samples must be positive"),
        (1_000, 10_000, -1, "num_samples must be positive"),
    ],
)
def test_chirpss_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid modulator parameters should raise ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        chirpss_modulator(
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_chirpss_modulator_creates_default_rng():
    """A default random generator should be created when rng is omitted."""
    rng = MagicMock(spec=np.random.Generator)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.chirpss_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        chirpss_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_chirpss_modulator_calculates_baseband_parameters():
    """The wrapper should derive the expected resampling parameters."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirpss_modulator_baseband",
            return_value=baseband,
        ) as baseband_modulator,
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ) as resampler,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        chirpss_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    # oversampling_rate = 10
    # resample_rate_ideal = 10 / 4 = 2.5
    # num_samples_baseband = floor(100 / 2.5) = 40
    baseband_modulator.assert_called_once_with(
        40,
        4,
        rng,
    )
    resampler.assert_called_once_with(
        baseband,
        2.5,
    )


def test_chirpss_modulator_slices_resampled_signal_when_too_long():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.arange(110, dtype=np.float32).astype(np.complex64)
    sliced = np.arange(100, dtype=np.float32).astype(np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirpss_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
            return_value=sliced,
        ) as slice_mock,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad_mock,
    ):
        result = chirpss_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    slice_mock.assert_called_once_with(
        resampled,
        100,
    )
    pad_mock.assert_not_called()

    np.testing.assert_array_equal(
        result,
        sliced.astype(TorchSigComplexDataType),
    )


@pytest.mark.parametrize("resampled_length", [99, 100])
def test_chirpss_modulator_pads_when_not_longer_than_target(
    resampled_length,
):
    """Signals no longer than the target should use the padding helper."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirpss_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad_mock,
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
        ) as slice_mock,
    ):
        result = chirpss_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    pad_mock.assert_called_once_with(
        resampled,
        100,
    )
    slice_mock.assert_not_called()

    assert result.shape == (100,)
    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_chirpss_signal_generator_initialization():
    """The generator should configure required metadata and class name."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            ChirpSSSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = ChirpSSSignalGenerator(**metadata)

    base_init.assert_called_once_with(generator, **metadata)
    set_class_name.assert_called_once_with("chirpss")

    assert generator.required_metadata_fields == [
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_chirpss_signal_generator_generate():
    """The generator should sample metadata and return a configured Signal."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        150,
        800,
    ]

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(150, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.chirpss_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = ChirpSSSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
    ]

    modulator.assert_called_once_with(
        800,
        10_000,
        150,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        class_name="chirpss",
        center_freq=0,
        bandwidth=800,
    )

    assert result is expected_signal
