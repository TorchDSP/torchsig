"""Unit tests for the AM signal generator module."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.am import AMSignalGenerator, am_modulator
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.am"


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "expected_message"),
    [
        (0, 10_000, "bandwidth must be positive"),
        (-1, 10_000, "bandwidth must be positive"),
        (1_000, 0, "sample_rate must be positive"),
        (1_000, -10_000, "sample_rate must be positive"),
        (5_001, 10_000, "bandwidth must be less than sample_rate/2"),
    ],
)
def test_am_modulator_rejects_invalid_numeric_parameters(
    bandwidth,
    sample_rate,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        am_modulator(
            am_mode="dsb-sc",
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=128,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize(
    "am_mode",
    [
        "",
        "am",
        "DSB",
        "dsb_sc",
        "invalid",
    ],
)
def test_am_modulator_rejects_invalid_mode(am_mode):
    with pytest.raises(
        ValueError,
        match="am_mode must be one of",
    ):
        am_modulator(
            am_mode=am_mode,
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=128,
            rng=np.random.default_rng(42),
        )


def test_am_modulator_creates_default_rng_when_none_is_given():
    rng = MagicMock(spec=np.random.Generator)
    rng.normal.return_value = np.ones(8)
    rng.uniform.return_value = 0.1

    shaped_message = np.arange(8, dtype=np.float32).astype(TorchSigComplexDataType)

    with (
        patch(f"{MODULE_PATH}.np.random.default_rng", return_value=rng) as default_rng,
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(f"{MODULE_PATH}.convolve", return_value=shaped_message),
    ):
        result = am_modulator(
            am_mode="dsb-sc",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
        )

    default_rng.assert_called_once_with()
    rng.normal.assert_called_once_with(0, 1, 8)
    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_am_modulator_dsb_sc_returns_shaped_message():
    rng = MagicMock(spec=np.random.Generator)
    rng.normal.return_value = np.array([1.0, -1.0, 1.0, -1.0])
    rng.uniform.return_value = 0.1

    low_pass_filter = np.array([0.25, 0.5, 0.25])
    shaped_message = np.array(
        [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j],
        dtype=np.complex64,
    )

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=low_pass_filter,
        ) as design_filter,
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=shaped_message,
        ) as convolve_mock,
    ):
        result = am_modulator(
            am_mode="dsb-sc",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=4,
            rng=rng,
        )

    rng.normal.assert_called_once_with(0, 1, 4)
    design_filter.assert_called_once_with(
        cutoff=500,
        transition_bandwidth=225,
        sample_rate=10_000,
    )
    convolve_mock.assert_called_once()
    np.testing.assert_array_equal(
        result,
        shaped_message.astype(TorchSigComplexDataType),
    )


def test_am_modulator_normalizes_message_before_filtering():
    rng = MagicMock(spec=np.random.Generator)
    rng.normal.return_value = np.array([1.0, 2.0, 3.0, 4.0])
    rng.uniform.return_value = 0.1

    captured_message = None

    def capture_message(message, _filter):
        nonlocal captured_message
        captured_message = message.copy()
        return message

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(f"{MODULE_PATH}.convolve", side_effect=capture_message),
    ):
        am_modulator(
            am_mode="dsb-sc",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=4,
            rng=rng,
        )

    assert captured_message is not None
    measured_power = np.mean(np.abs(captured_message) ** 2)
    assert measured_power == pytest.approx(1.0)


def test_am_modulator_dsb_adds_carrier_using_modulation_index():
    rng = MagicMock(spec=np.random.Generator)
    rng.normal.return_value = np.ones(4)

    # First call selects transition bandwidth; second selects modulation index.
    rng.uniform.side_effect = [0.1, 2.0]

    shaped_message = np.array(
        [1 + 0j, -2 + 0j, 3 + 0j, -4 + 0j],
        dtype=np.complex64,
    )

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(f"{MODULE_PATH}.convolve", return_value=shaped_message),
    ):
        result = am_modulator(
            am_mode="dsb",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=4,
            rng=rng,
        )

    modulation_index = 2.0
    carrier_amplitude = np.max(np.abs(shaped_message)) / modulation_index
    expected = modulation_index * shaped_message + carrier_amplitude

    np.testing.assert_allclose(
        result,
        expected.astype(TorchSigComplexDataType),
    )
    assert rng.uniform.call_args_list == [
        call(0.05, 0.25),
        call(0.8, 4),
    ]


def test_am_modulator_lsb_uses_expected_frequency_shifts_and_decimation():
    rng = MagicMock(spec=np.random.Generator)
    rng.normal.return_value = np.ones(16)
    rng.uniform.return_value = 0.1

    low_pass_filter = np.array([0.25, 0.5, 0.25])
    shaped_message = np.arange(16, dtype=np.complex64)
    upconverted = shaped_message + 10
    lsb_at_if = shaped_message + 20
    oversampled = shaped_message + 30
    decimated = np.arange(8, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=low_pass_filter,
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            side_effect=[shaped_message, lsb_at_if],
        ) as convolve_mock,
        patch(
            f"{MODULE_PATH}.frequency_shift",
            side_effect=[upconverted, oversampled],
        ) as frequency_shift_mock,
        patch(
            f"{MODULE_PATH}.polyphase_decimator",
            return_value=decimated,
        ) as decimator_mock,
    ):
        result = am_modulator(
            am_mode="lsb",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    # SSB modes create twice as many samples before decimation.
    rng.normal.assert_called_once_with(0, 1, 16)

    assert convolve_mock.call_args_list[0].args[1] is low_pass_filter
    assert convolve_mock.call_args_list[1] == call(
        upconverted,
        low_pass_filter,
    )

    assert frequency_shift_mock.call_args_list == [
        call(shaped_message, 500, 10_000),
        call(lsb_at_if, -250, 10_000),
    ]
    decimator_mock.assert_called_once_with(oversampled, 2)

    np.testing.assert_array_equal(
        result,
        (decimated * 2).astype(TorchSigComplexDataType),
    )


def test_am_modulator_usb_uses_expected_frequency_shifts_and_decimation():
    rng = MagicMock(spec=np.random.Generator)
    rng.normal.return_value = np.ones(16)
    rng.uniform.return_value = 0.1

    low_pass_filter = np.array([0.25, 0.5, 0.25])
    shaped_message = np.arange(16, dtype=np.complex64)
    downconverted = shaped_message + 10
    usb_at_if = shaped_message + 20
    oversampled = shaped_message + 30
    decimated = np.arange(8, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=low_pass_filter,
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            side_effect=[shaped_message, usb_at_if],
        ) as convolve_mock,
        patch(
            f"{MODULE_PATH}.frequency_shift",
            side_effect=[downconverted, oversampled],
        ) as frequency_shift_mock,
        patch(
            f"{MODULE_PATH}.polyphase_decimator",
            return_value=decimated,
        ) as decimator_mock,
    ):
        result = am_modulator(
            am_mode="usb",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    rng.normal.assert_called_once_with(0, 1, 16)

    assert convolve_mock.call_args_list[1] == call(
        downconverted,
        low_pass_filter,
    )
    assert frequency_shift_mock.call_args_list == [
        call(shaped_message, -500, 10_000),
        call(usb_at_if, 250, 10_000),
    ]
    decimator_mock.assert_called_once_with(oversampled, 2)

    np.testing.assert_array_equal(
        result,
        (decimated * 2).astype(TorchSigComplexDataType),
    )


@pytest.mark.parametrize("am_mode", ["dsb", "dsb-sc", "lsb", "usb"])
def test_am_modulator_always_returns_complex_dtype(am_mode):
    rng = MagicMock(spec=np.random.Generator)
    input_length = 16 if am_mode in {"lsb", "usb"} else 8
    rng.normal.return_value = np.ones(input_length)
    rng.uniform.return_value = 0.1

    shaped_message = np.ones(input_length, dtype=np.float64)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=shaped_message,
        ),
        patch(
            f"{MODULE_PATH}.frequency_shift",
            return_value=shaped_message,
        ),
        patch(
            f"{MODULE_PATH}.polyphase_decimator",
            return_value=np.ones(8),
        ),
    ):
        result = am_modulator(
            am_mode=am_mode,
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_am_signal_generator_initialization_sets_required_fields_and_class_name():
    metadata = {"am_mode": "usb"}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            AMSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            AMSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = AMSignalGenerator(**metadata)

    base_init.assert_called_once_with(generator, **metadata)
    set_class_name.assert_called_once_with("am-usb")
    assert generator.required_metadata_fields == [
        "am_mode",
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_am_signal_generator_generate_uses_sampled_parameters():
    metadata = {
        "am_mode": "dsb-sc",
        "sample_rate": 10_000,
        "bandwidth_min": 800,
        "bandwidth_max": 1_200,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        150,  # Number of samples
        1_000,  # Bandwidth
    ]

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(150, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.am_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_cls,
    ):
        result = AMSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=800, high=1_201),
    ]

    modulator.assert_called_once_with(
        "dsb-sc",
        1_000,
        10_000,
        150,
        rng,
    )
    signal_cls.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=1_000,
    )
    assert result is expected_signal
