"""Unit tests for the LFM signal builder and modulator."""

from collections import OrderedDict
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.lfm import (
    LFMSignalGenerator,
    get_symbol_map,
    lfm_modulator,
    lfm_modulator_baseband,
)
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.lfm"


def test_get_symbol_map_returns_expected_ordered_dict():
    """The symbol map should expose data and radar modes in order."""
    result = get_symbol_map()

    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == ["data", "radar"]

    np.testing.assert_array_equal(
        result["data"],
        np.array([-1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        result["radar"],
        np.array([1.0]),
    )


def test_get_symbol_map_returns_fresh_arrays():
    """Each call should return independent symbol-map arrays."""
    first = get_symbol_map()
    second = get_symbol_map()

    first["data"][0] = 999

    assert second["data"][0] == -1.0


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_lfm_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive requested output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -10])
def test_lfm_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive nominal oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_lfm_modulator_baseband_rejects_unknown_lfm_type():
    """Unsupported LFM types should raise a descriptive ValueError."""
    with pytest.raises(
        ValueError,
        match=(
            r"Unsupported LFM type: invalid\. "
            r"Must be one of: \['data', 'radar'\]"
        ),
    ):
        lfm_modulator_baseband(
            lfm_type="invalid",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


def test_lfm_modulator_baseband_creates_default_rng():
    """A default NumPy random generator should be created when omitted."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.ones(128, dtype=np.complex64)
    downchirp = -np.ones(128, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.chirp",
            side_effect=[upchirp, downchirp],
        ),
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=128,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (128,)


@pytest.mark.parametrize(
    ("oversampling_rate", "expected_f0", "expected_f1"),
    [
        (1, -0.5, 0.5),
        (2, -0.25, 0.25),
        (4, -0.125, 0.125),
        (8, -0.0625, 0.0625),
    ],
)
def test_lfm_modulator_baseband_uses_expected_chirp_bounds(
    oversampling_rate,
    expected_f0,
    expected_f1,
):
    """Chirp bounds should span half the implied baseband bandwidth."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([1]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.ones(128, dtype=np.complex64)
    downchirp = -np.ones(128, dtype=np.complex64)

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[upchirp, downchirp],
    ) as chirp_mock:
        lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=rng,
        )

    assert chirp_mock.call_args_list == [
        call(expected_f0, expected_f1, 128),
        call(expected_f1, expected_f0, 128),
    ]


def test_lfm_modulator_baseband_randomizes_samples_per_symbol():
    """The chirp symbol length should be drawn from the configured range."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        256,
        np.array([1]),
    ]
    rng.uniform.return_value = 1.0

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[
            np.ones(256, dtype=np.complex64),
            -np.ones(256, dtype=np.complex64),
        ],
    ):
        lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=256,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert rng.integers.call_args_list[0] == call(
        low=128,
        high=4096,
    )


def test_lfm_modulator_baseband_requests_enough_symbols():
    """The symbol count should use the ceiling of output length per symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        128,
        np.array([1, 0, 1]),
    ]
    rng.uniform.return_value = 1.0

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[
            np.ones(128, dtype=np.complex64),
            -np.ones(128, dtype=np.complex64),
        ],
    ):
        lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=300,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert rng.integers.call_args_list == [
        call(low=128, high=4096),
        call(0, 2, 3),
    ]


def test_lfm_modulator_baseband_data_maps_positive_symbol_to_upchirp():
    """A positive data symbol should select the upchirp template."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([1]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(8, dtype=np.float32).astype(np.complex64)
    downchirp = -upchirp

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[upchirp, downchirp],
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    np.testing.assert_array_equal(result, upchirp)


def test_lfm_modulator_baseband_data_maps_negative_symbol_to_downchirp():
    """A negative data symbol should select the downchirp template."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(8, dtype=np.float32).astype(np.complex64)
    downchirp = -upchirp

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[upchirp, downchirp],
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    np.testing.assert_array_equal(result, downchirp)


def test_lfm_modulator_baseband_radar_always_uses_upchirp():
    """Radar mode should contain only positive symbols and upchirps."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(8, dtype=np.float32).astype(np.complex64)
    downchirp = -upchirp

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[upchirp, downchirp],
    ):
        result = lfm_modulator_baseband(
            lfm_type="radar",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    np.testing.assert_array_equal(result, upchirp)


def test_lfm_modulator_baseband_concatenates_multiple_symbols():
    """Successive symbols should occupy consecutive output regions."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        4,
        np.array([1, 0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.array([1, 2, 3, 4], dtype=np.complex64)
    downchirp = np.array([-1, -2, -3, -4], dtype=np.complex64)

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[upchirp, downchirp],
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    expected = np.concatenate((upchirp, downchirp))

    np.testing.assert_array_equal(result, expected)


def test_lfm_modulator_baseband_truncates_final_symbol():
    """The final chirp should be truncated to the requested output length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([1, 0]),
    ]
    rng.uniform.return_value = 1.0

    upchirp = np.arange(8, dtype=np.float32).astype(np.complex64)
    downchirp = -upchirp

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[upchirp, downchirp],
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=11,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    expected = np.concatenate(
        (
            upchirp,
            downchirp[:3],
        )
    )

    assert result.shape == (11,)
    np.testing.assert_array_equal(result, expected)


def test_lfm_modulator_baseband_applies_limiting_filter_when_selected():
    """A probability draw below 0.5 should apply a limiting filter."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([1]),
    ]
    rng.uniform.return_value = 0.25

    upchirp = np.ones(8, dtype=np.complex64)
    downchirp = -np.ones(8, dtype=np.complex64)
    filter_taps = np.array([0.25, 0.5, 0.25])
    filtered = np.full(8, 2 + 1j, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirp",
            side_effect=[upchirp, downchirp],
        ),
        patch(
            f"{MODULE_PATH}.random_limiting_filter_design",
            return_value=filter_taps,
        ) as filter_design,
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=filtered,
        ) as convolve_mock,
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    filter_design.assert_called_once_with(
        0.25,
        1.0,
        rng,
    )
    convolve_mock.assert_called_once()

    np.testing.assert_array_equal(
        convolve_mock.call_args.args[0],
        upchirp,
    )
    assert convolve_mock.call_args.args[1] is filter_taps
    assert result is filtered


@pytest.mark.parametrize("probability_draw", [0.5, 0.75, 1.0])
def test_lfm_modulator_baseband_skips_filter_when_not_selected(
    probability_draw,
):
    """A probability draw of at least 0.5 should skip filtering."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([1]),
    ]
    rng.uniform.return_value = probability_draw

    upchirp = np.ones(8, dtype=np.complex64)
    downchirp = -np.ones(8, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.chirp",
            side_effect=[upchirp, downchirp],
        ),
        patch(
            f"{MODULE_PATH}.random_limiting_filter_design",
        ) as filter_design,
        patch(
            f"{MODULE_PATH}.convolve",
        ) as convolve_mock,
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    filter_design.assert_not_called()
    convolve_mock.assert_not_called()
    np.testing.assert_array_equal(result, upchirp)


def test_lfm_modulator_baseband_returns_torchsig_complex_dtype_without_filter():
    """The preallocated unfiltered result should use TorchSig complex dtype."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        8,
        np.array([1]),
    ]
    rng.uniform.return_value = 1.0

    with patch(
        f"{MODULE_PATH}.chirp",
        side_effect=[
            np.ones(8, dtype=np.complex128),
            -np.ones(8, dtype=np.complex128),
        ],
    ):
        result = lfm_modulator_baseband(
            lfm_type="data",
            max_num_samples=8,
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
def test_lfm_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid top-level LFM parameters should be rejected."""
    with pytest.raises(ValueError, match=expected_message):
        lfm_modulator(
            lfm_type="data",
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_lfm_modulator_creates_default_rng():
    """A default random generator should be created when omitted."""
    rng = MagicMock(spec=np.random.Generator)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.lfm_modulator_baseband",
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
        lfm_modulator(
            lfm_type="data",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_lfm_modulator_calculates_resampling_parameters():
    """The wrapper should derive the expected baseband length and rate."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.lfm_modulator_baseband",
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
        lfm_modulator(
            lfm_type="radar",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    # oversampling_rate = 10
    # resample_rate_ideal = 10 / 4 = 2.5
    # ceil(100 / 2.5) = 40
    baseband_modulator.assert_called_once_with(
        "radar",
        40,
        4,
        rng,
    )
    resampler.assert_called_once_with(
        baseband,
        2.5,
    )


def test_lfm_modulator_uses_minimum_one_baseband_sample():
    """The wrapper should request at least one baseband sample."""
    rng = np.random.default_rng(42)
    baseband = np.ones(1, dtype=np.complex64)
    resampled = np.ones(1, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.lfm_modulator_baseband",
            return_value=baseband,
        ) as baseband_modulator,
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        lfm_modulator(
            lfm_type="data",
            bandwidth=1,
            sample_rate=10_000,
            num_samples=1,
            rng=rng,
        )

    baseband_modulator.assert_called_once_with(
        "data",
        1,
        4,
        rng,
    )


def test_lfm_modulator_slices_long_resampled_signal():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(110, dtype=np.complex64)
    sliced = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.lfm_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
            return_value=sliced,
        ) as slice_signal,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad_signal,
    ):
        result = lfm_modulator(
            lfm_type="data",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    slice_signal.assert_called_once_with(
        resampled,
        100,
    )
    pad_signal.assert_not_called()

    np.testing.assert_array_equal(
        result,
        sliced.astype(TorchSigComplexDataType),
    )


@pytest.mark.parametrize("resampled_length", [90, 100])
def test_lfm_modulator_pads_signal_not_longer_than_target(
    resampled_length,
):
    """A signal no longer than the target should use the padding helper."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.lfm_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad_signal,
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
        ) as slice_signal,
    ):
        result = lfm_modulator(
            lfm_type="data",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    pad_signal.assert_called_once_with(
        resampled,
        100,
    )
    slice_signal.assert_not_called()

    assert result.shape == (100,)
    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_lfm_modulator_returns_torchsig_complex_dtype():
    """The top-level modulator should cast its result to TorchSig dtype."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex128)
    resampled = np.ones(100, dtype=np.complex128)

    with (
        patch(
            f"{MODULE_PATH}.lfm_modulator_baseband",
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
        result = lfm_modulator(
            lfm_type="data",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_lfm_signal_generator_initialization():
    """The generator should configure required fields and class name."""
    metadata = {"lfm_type": "radar"}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            LFMSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            LFMSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = LFMSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )
    set_class_name.assert_called_once_with("lfm-radar")

    assert generator.required_metadata_fields == [
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "lfm_type",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_lfm_signal_generator_generate():
    """The generator should sample metadata and return a Signal."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "lfm_type": "data",
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
            f"{MODULE_PATH}.lfm_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = LFMSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
    ]

    modulator.assert_called_once_with(
        "data",
        800,
        10_000,
        150,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=800,
    )

    assert result is expected_signal
