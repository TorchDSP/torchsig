"""Unit tests for the constellation signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.constellation import (
    ConstellationSignalGenerator,
    constellation_modulator,
    constellation_modulator_baseband,
)
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.constellation"


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_constellation_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive baseband output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -10])
def test_constellation_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive nominal oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_creates_default_rng():
    """A default NumPy generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
    ):
        result = constellation_modulator_baseband(
            constellation_name="bpsk",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (8,)


def test_constellation_modulator_baseband_rejects_unknown_pulse_shape():
    """Unsupported pulse-shaping filters should be rejected."""
    with pytest.raises(
        ValueError,
        match="pulse shape invalid not supported",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="invalid",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_srrc_requires_alpha_rolloff():
    """SRRC pulse shaping should require an alpha-rolloff value."""
    with pytest.raises(
        ValueError,
        match="must define an alpha rolloff for SRRC filter",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="srrc",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            alpha_rolloff=None,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("alpha_rolloff", [-1.0, 0.0, 1.0, 1.5])
def test_constellation_modulator_baseband_rejects_invalid_alpha_rolloff(
    alpha_rolloff,
):
    """SRRC alpha rolloff should be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="alpha_rolloff must be between 0 and 1",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="srrc",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            alpha_rolloff=alpha_rolloff,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_raises_for_unknown_constellation():
    """An unknown constellation name should raise a KeyError."""
    with pytest.raises(KeyError):
        constellation_modulator_baseband(
            constellation_name="not-a-constellation",
            pulse_shape_name="rectangular",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_rectangular_pulse_shape():
    """Rectangular pulse shaping should use one tap per sample per symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    normalized_map = np.array([-1 + 0j, 1 + 0j])
    expected_symbols = normalized_map[[0, 1]]

    np.testing.assert_array_equal(
        upfirdn.call_args.args[0],
        np.ones(4),
    )
    np.testing.assert_array_equal(
        upfirdn.call_args.args[1],
        expected_symbols,
    )
    assert upfirdn.call_args.kwargs == {
        "up": 4,
        "down": 1,
    }

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_constellation_modulator_baseband_normalizes_symbol_map():
    """The constellation map should be normalized to average unit power."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    raw_map = np.array([-2 + 0j, 2 + 0j])
    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": raw_map},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    symbols = upfirdn.call_args.args[1]

    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_constellation_modulator_baseband_srrc_designs_expected_filter():
    """SRRC pulse shaping should derive its span and taps."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    pulse_shape = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
    shaped = np.ones(32, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.estimate_filter_length",
            return_value=17,
        ) as estimate_length,
        patch(
            f"{MODULE_PATH}.srrc_taps",
            return_value=pulse_shape,
        ) as taps,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="srrc",
            max_num_samples=32,
            oversampling_rate_nominal=4,
            alpha_rolloff=0.25,
            rng=rng,
        )

    estimate_length.assert_called_once_with(
        0.25,
        120,
        1,
    )

    # ceil((17 - 1) / (2 * 4)) = 2
    taps.assert_called_once_with(
        4,
        2,
        0.25,
    )

    np.testing.assert_array_equal(
        upfirdn.call_args.args[0],
        pulse_shape,
    )
    assert result.shape == (32,)


def test_constellation_modulator_baseband_accounts_for_srrc_filter_span():
    """SRRC transient symbols should be removed from the symbol count."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1, 0, 1])

    shaped = np.ones(32, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.estimate_filter_length",
            return_value=17,
        ),
        patch(
            f"{MODULE_PATH}.srrc_taps",
            return_value=np.ones(17),
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
    ):
        constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="srrc",
            max_num_samples=32,
            oversampling_rate_nominal=4,
            alpha_rolloff=0.25,
            rng=rng,
        )

    # floor(32 / 4) - 2 * span
    # 8 - 2 * 2 = 4 symbols
    rng.integers.assert_called_once_with(
        low=0,
        high=2,
        size=4,
    )


def test_constellation_modulator_baseband_generates_at_least_one_symbol():
    """Very short requests should still generate one symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([1])

    shaped = np.ones(4, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=np.ones(1, dtype=np.complex64),
        ),
    ):
        constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=1,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    rng.integers.assert_called_once_with(
        low=0,
        high=2,
        size=1,
    )


def test_constellation_modulator_baseband_retries_all_zero_ook_symbols():
    """OOK generation should retry when every selected symbol is zero."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        np.array([0, 0]),
        np.array([0, 1]),
    ]

    raw_symbol_map = np.array([0 + 0j, 1 + 0j])
    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"ook-test": raw_symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        constellation_modulator_baseband(
            constellation_name="ook-test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert rng.integers.call_count == 2

    expected_symbol_map = raw_symbol_map / np.sqrt(np.mean(np.abs(raw_symbol_map) ** 2))

    np.testing.assert_allclose(
        upfirdn.call_args.args[1],
        expected_symbol_map[[0, 1]],
    )


def test_constellation_modulator_baseband_pads_short_signal():
    """A short pulse-shaped result should be padded to the requested length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(6, dtype=np.complex64)
    padded = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    pad.assert_called_once_with(
        shaped,
        8,
    )
    slice_tail.assert_not_called()

    np.testing.assert_array_equal(result, padded)


def test_constellation_modulator_baseband_slices_long_signal():
    """A long pulse-shaped result should be sliced from the tail."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(10, dtype=np.complex64)
    sliced = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=sliced,
        ) as slice_tail,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    slice_tail.assert_called_once_with(
        shaped,
        8,
    )
    pad.assert_not_called()

    np.testing.assert_array_equal(result, sliced)


def test_constellation_modulator_baseband_leaves_exact_length_unchanged():
    """An exact-length pulse-shaped result should need no adjustment."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.arange(8, dtype=np.float32).astype(np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    slice_tail.assert_not_called()
    pad.assert_not_called()

    np.testing.assert_array_equal(
        result,
        shaped.astype(TorchSigComplexDataType),
    )


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
def test_constellation_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid top-level modulation parameters should be rejected."""
    with pytest.raises(ValueError, match=expected_message):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_creates_default_rng():
    """A default generator should be created when rng is omitted."""
    rng = MagicMock(spec=np.random.Generator)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
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
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_constellation_modulator_calculates_resampling_parameters():
    """The wrapper should calculate the expected baseband length and rate."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
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
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="srrc",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            alpha_rolloff=0.25,
            rng=rng,
        )

    # oversampling_rate = 10
    # resample_rate_ideal = 10 / 4 = 2.5
    # num_samples_baseband = floor(100 / 2.5) = 40
    baseband_modulator.assert_called_once_with(
        "qpsk",
        "srrc",
        40,
        4,
        0.25,
        rng,
    )

    resampler.assert_called_once_with(
        baseband,
        2.5,
    )


def test_constellation_modulator_uses_minimum_baseband_length():
    """At least four baseband samples should be requested."""
    rng = np.random.default_rng(42)
    baseband = np.ones(4, dtype=np.complex64)
    resampled = np.ones(1, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
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
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1,
            sample_rate=10_000,
            num_samples=1,
            rng=rng,
        )

    baseband_modulator.assert_called_once_with(
        "qpsk",
        "rectangular",
        4,
        4,
        None,
        rng,
    )


def test_constellation_modulator_slices_long_resampled_signal():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(110, dtype=np.complex64)
    sliced = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
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
        result = constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
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

    assert result.shape == (100,)
    assert result.dtype == np.dtype(TorchSigComplexDataType)


@pytest.mark.parametrize("resampled_length", [90, 100])
def test_constellation_modulator_pads_signal_not_longer_than_target(
    resampled_length,
):
    """A resampled signal no longer than the target should use padding."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
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
        result = constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
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


def test_constellation_modulator_rejects_incorrect_final_length():
    """A malfunctioning length helper should trigger output validation."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(90, dtype=np.complex64)
    incorrectly_padded = np.ones(99, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=incorrectly_padded,
        ),
        pytest.raises(
            ValueError,
            match=("constellation mod producing incorrect number of samples: 99 but requested: 100"),
        ),
    ):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )


def test_constellation_signal_generator_initialization():
    """The generator should configure metadata fields and class name."""
    metadata = {"constellation_name": "qpsk"}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            ConstellationSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            ConstellationSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = ConstellationSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )
    set_class_name.assert_called_once_with("qpsk")

    assert generator.required_metadata_fields == [
        "constellation_name",
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_constellation_signal_generator_generate_with_srrc():
    """A pulse-shape draw of zero should select SRRC and an alpha value."""
    metadata = {
        "constellation_name": "qpsk",
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
        0,
    ]
    rng.uniform.return_value = 0.25

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(150, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = ConstellationSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
        call(0, 2),
    ]
    rng.uniform.assert_called_once_with(0.1, 0.5)

    modulator.assert_called_once_with(
        "qpsk",
        "srrc",
        800,
        10_000,
        150,
        0.25,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=800,
        pulse_shape_name="srrc",
        alpha_rolloff=0.25,
        pulse_shape_index=1,
        alpha_rolloff_target=0.25,
    )

    assert result is expected_signal


def test_constellation_signal_generator_generate_with_rectangular():
    """A pulse-shape draw of one should select rectangular shaping."""
    metadata = {
        "constellation_name": "16qam",
        "sample_rate": 20_000,
        "bandwidth_min": 1_000,
        "bandwidth_max": 2_000,
        "signal_duration_in_samples_min": 200,
        "signal_duration_in_samples_max": 400,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        300,
        1_500,
        1,
    ]

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(300, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = ConstellationSignalGenerator.generate(GeneratorStub())

    rng.uniform.assert_not_called()

    modulator.assert_called_once_with(
        "16qam",
        "rectangular",
        1_500,
        20_000,
        300,
        None,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=1_500,
        pulse_shape_name="rectangular",
        alpha_rolloff=None,
        pulse_shape_index=0,
        alpha_rolloff_target=0.0,
    )

    assert result is expected_signal
