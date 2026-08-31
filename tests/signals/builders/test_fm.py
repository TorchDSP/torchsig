"""Unit tests for the FM signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.fm import FMSignalGenerator, fm_modulator
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.fm"


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "num_samples", "expected_message"),
    [
        (0, 10_000, 128, "bandwidth must be positive"),
        (-1, 10_000, 128, "bandwidth must be positive"),
        (1_000, 0, 128, "sample_rate must be positive"),
        (1_000, -1, 128, "sample_rate must be positive"),
        (
            10_001,
            10_000,
            128,
            "bandwidth must be less than or equal to sample_rate",
        ),
        (1_000, 10_000, 0, "num_samples must be positive"),
        (1_000, 10_000, -1, "num_samples must be positive"),
    ],
)
def test_fm_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid FM parameters should raise ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        fm_modulator(
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_fm_modulator_creates_default_rng():
    """A default NumPy generator should be created when rng is omitted."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(8)

    filtered_source = np.ones(8)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=filtered_source,
        ),
    ):
        result = fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (8,)


def test_fm_modulator_samples_modulation_index_from_expected_range():
    """The modulation index should be sampled uniformly from one to ten."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(8)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.ones(8),
        ),
    ):
        fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    rng.uniform.assert_called_once_with(1, 10)


def test_fm_modulator_generates_requested_number_of_message_samples():
    """The source message should contain the requested number of samples."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(17)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.ones(17),
        ),
    ):
        fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=17,
            rng=rng,
        )

    rng.normal.assert_called_once_with(0, 1, 17)


def test_fm_modulator_normalizes_message_to_unit_power():
    """The unfiltered source message should have average unit power."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.array([1.0, 2.0, 3.0, 4.0])

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
        patch(
            f"{MODULE_PATH}.convolve",
            side_effect=capture_message,
        ),
    ):
        fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=4,
            rng=rng,
        )

    assert captured_message is not None
    assert np.mean(np.abs(captured_message) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mod_index", "expected_fdev", "expected_fmax"),
    [
        (1.0, 250.0, 250.0),
        (2.0, 1_000 / 3, 1_000 / 6),
        (4.0, 400.0, 100.0),
        (10.0, 500 / 1.1, (500 / 1.1) / 10),
    ],
)
def test_fm_modulator_uses_carsons_rule(
    mod_index,
    expected_fdev,
    expected_fmax,
):
    """Filter cutoff should follow the implemented Carson's Rule equations."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = mod_index
    rng.normal.return_value = np.ones(8)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ) as filter_design,
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.ones(8),
        ),
    ):
        fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    filter_design.assert_called_once()

    assert filter_design.call_args.kwargs["cutoff"] == pytest.approx(expected_fmax)
    assert filter_design.call_args.kwargs["transition_bandwidth"] == pytest.approx(expected_fmax)
    assert filter_design.call_args.kwargs["sample_rate"] == 10_000


def test_fm_modulator_accepts_bandwidth_above_half_sample_rate():
    """A full two-sided bandwidth may exceed half the sample rate."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 1.0
    rng.normal.return_value = np.ones(8)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ) as filter_design,
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.ones(8),
        ),
    ):
        result = fm_modulator(
            bandwidth=7_500,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    assert result.shape == (8,)
    assert filter_design.call_args.kwargs["cutoff"] == pytest.approx(1_875)
    assert filter_design.call_args.kwargs["transition_bandwidth"] == pytest.approx(1_875)


def test_fm_modulator_passes_normalized_message_and_filter_to_convolve():
    """The normalized source and designed LPF should be convolved."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.array([1.0, -1.0, 1.0, -1.0])

    low_pass_filter = np.array([0.25, 0.5, 0.25])
    filtered_source = np.array([0.1, 0.2, 0.3, 0.4])

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=low_pass_filter,
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=filtered_source,
        ) as convolve_mock,
    ):
        fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=4,
            rng=rng,
        )

    normalized_message = np.array([1.0, -1.0, 1.0, -1.0])

    np.testing.assert_allclose(
        convolve_mock.call_args.args[0],
        normalized_message,
    )
    assert convolve_mock.call_args.args[1] is low_pass_filter


def test_fm_modulator_matches_phase_accumulation_formula():
    """The output should match the implemented cumulative-phase FM equation."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(4)

    filtered_source = np.array([0.1, -0.2, 0.3, -0.4])

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=filtered_source,
        ),
    ):
        result = fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=4,
            rng=rng,
        )

    mod_index = 2.0
    fdev = (1_000 / 2) / (1 + 1 / mod_index)

    expected = np.exp(2j * np.pi * np.cumsum(filtered_source) * fdev / 10_000).astype(TorchSigComplexDataType)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_fm_modulator_returns_requested_shape():
    """The result should contain the requested number of samples."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(32)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.ones(32),
        ),
    ):
        result = fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=32,
            rng=rng,
        )

    assert result.shape == (32,)


def test_fm_modulator_returns_torchsig_complex_dtype():
    """The FM signal should use the configured TorchSig complex dtype."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(8)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.ones(8),
        ),
    ):
        result = fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_fm_modulator_has_unit_magnitude():
    """Ideal FM output samples should lie on the complex unit circle."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 3.0
    rng.normal.return_value = np.ones(8)

    filtered_source = np.linspace(-1, 1, 8)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=filtered_source,
        ),
    ):
        result = fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    np.testing.assert_allclose(
        np.abs(result),
        np.ones(8),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fm_modulator_zero_filtered_source_returns_constant_carrier():
    """A zero source signal should produce a constant unit-valued carrier."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 2.0
    rng.normal.return_value = np.ones(8)

    with (
        patch(
            f"{MODULE_PATH}.low_pass_iterative_design",
            return_value=np.array([1.0]),
        ),
        patch(
            f"{MODULE_PATH}.convolve",
            return_value=np.zeros(8),
        ),
    ):
        result = fm_modulator(
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=8,
            rng=rng,
        )

    np.testing.assert_array_equal(
        result,
        np.ones(8, dtype=TorchSigComplexDataType),
    )


def test_fm_signal_generator_initialization():
    """The generator should configure required fields and class name."""
    metadata = {}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            FMSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = FMSignalGenerator(**metadata)

    base_init.assert_called_once_with(generator)
    set_class_name.assert_called_once_with("fm")

    assert generator.required_metadata_fields == [
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_fm_signal_generator_generate():
    """The generator should sample parameters and create the expected Signal."""
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
            f"{MODULE_PATH}.fm_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = FMSignalGenerator.generate(GeneratorStub())

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
        center_freq=0,
        bandwidth=800,
    )

    assert result is expected_signal
