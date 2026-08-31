"""Unit tests for the OFDM signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.ofdm import (
    OFDMSignalGenerator,
    ofdm_modulator,
    ofdm_modulator_baseband,
)
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.ofdm"


@pytest.mark.parametrize("num_subcarriers", [0, -1, -16])
def test_ofdm_modulator_baseband_rejects_nonpositive_num_subcarriers(
    num_subcarriers,
):
    """Nonpositive subcarrier counts should be rejected."""
    with pytest.raises(
        ValueError,
        match="num_subcarriers must be positive",
    ):
        ofdm_modulator_baseband(
            num_subcarriers=num_subcarriers,
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_ofdm_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        ofdm_modulator_baseband(
            num_subcarriers=16,
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -4])
def test_ofdm_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive nominal oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        ofdm_modulator_baseband(
            num_subcarriers=16,
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_ofdm_modulator_baseband_creates_default_rng():
    """A default NumPy generator should be created when none is provided."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 1), dtype=int),
    ]

    symbol_map = np.array([-1 + 0j, 1 + 0j])

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        result = ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=32,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (32,)


def test_ofdm_modulator_baseband_overrides_supplied_oversampling_rate():
    """The implementation should use its fixed OFDM oversampling rate of four."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 1), dtype=int),
    ]

    symbol_map = np.array([-1 + 0j, 1 + 0j])
    captured_grid = None

    def capture_ifft(grid, axis):
        nonlocal captured_grid
        captured_grid = grid.copy()
        assert axis == 0
        return np.zeros_like(grid)

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            side_effect=capture_ifft,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=32,
            oversampling_rate_nominal=99,
            rng=rng,
        )

    assert captured_grid is not None

    # Fixed internal oversampling rate is four:
    # ifft_size = 4 * 8 = 32.
    assert captured_grid.shape == (32, 1)


def test_ofdm_modulator_baseband_no_cyclic_prefix_branch():
    """A probability draw below 0.5 should omit the cyclic prefix."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 1), dtype=int),
    ]

    symbol_map = np.array([-1 + 0j, 1 + 0j])
    ifft_output = np.arange(32, dtype=np.float32).reshape(32, 1).astype(np.complex64)

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            return_value=ifft_output,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        result = ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    np.testing.assert_array_equal(
        result,
        ifft_output[:, 0],
    )
    rng.uniform.assert_called_once_with(0, 1)


def test_ofdm_modulator_baseband_adds_cyclic_prefix():
    """A probability draw of at least 0.5 should add a cyclic prefix."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.75

    # cp_len=2, modulation index=0, symbol indices all zero.
    rng.integers.side_effect = [
        2,
        0,
        np.zeros((8, 1), dtype=int),
    ]

    symbol_map = np.array([-1 + 0j, 1 + 0j])
    ifft_output = np.arange(32, dtype=np.float32).reshape(32, 1).astype(np.complex64)

    serialized_with_cp = np.concatenate(
        (
            ifft_output[-8:, 0],
            ifft_output[:, 0],
        )
    )

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            return_value=ifft_output,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ) as slice_tail,
    ):
        result = ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=40,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    rng.integers.assert_any_call(2, 4)

    np.testing.assert_array_equal(
        slice_tail.call_args.args[0],
        serialized_with_cp,
    )
    np.testing.assert_array_equal(
        result,
        serialized_with_cp,
    )


def test_ofdm_modulator_baseband_selects_subcarrier_modulation():
    """The selected modulation should come from the OFDM modulation list."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25

    # Select index 1, then generate symbol grid indices.
    rng.integers.side_effect = [
        1,
        np.zeros((8, 1), dtype=int),
    ]

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["bpsk", "qpsk"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {
                "bpsk": np.array([-1 + 0j, 1 + 0j]),
                "qpsk": np.array(
                    [
                        -1 - 1j,
                        -1 + 1j,
                        1 - 1j,
                        1 + 1j,
                    ]
                ),
            },
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    rng.integers.assert_any_call(0, 2)


def test_ofdm_modulator_baseband_normalizes_symbol_map():
    """Subcarrier symbols should be normalized to average unit power."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.array(
            [
                [0],
                [1],
                [0],
                [1],
            ]
        ),
    ]

    raw_symbol_map = np.array([-2 + 0j, 2 + 0j])
    captured_grid = None

    def capture_ifft(grid, axis):
        nonlocal captured_grid
        captured_grid = grid.copy()
        assert axis == 0
        return np.zeros_like(grid)

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": raw_symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            side_effect=capture_ifft,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        ofdm_modulator_baseband(
            num_subcarriers=4,
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    normalized_map = raw_symbol_map / np.sqrt(np.mean(np.abs(raw_symbol_map) ** 2))

    assert captured_grid is not None

    # Positive-frequency carriers use rows 1 and 2.
    np.testing.assert_allclose(
        captured_grid[1:3, 0],
        normalized_map[[0, 1]],
    )

    # Negative-frequency carriers use the final two rows.
    np.testing.assert_allclose(
        captured_grid[-2:, 0],
        normalized_map[[0, 1]],
    )


def test_ofdm_modulator_baseband_leaves_dc_and_guard_bins_zero():
    """Only the configured active subcarriers should be populated."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 1), dtype=int),
    ]

    captured_grid = None

    def capture_ifft(grid, axis):
        nonlocal captured_grid
        captured_grid = grid.copy()
        return np.zeros_like(grid)

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            side_effect=capture_ifft,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert captured_grid is not None
    assert captured_grid[0, 0] == 0

    active_indices = set(range(1, 5)) | set(range(28, 32))
    inactive_indices = set(range(32)) - active_indices

    for index in inactive_indices:
        assert captured_grid[index, 0] == 0


def test_ofdm_modulator_baseband_requests_expected_symbol_grid_shape():
    """The random symbol-index grid should match carriers by OFDM symbols."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 3), dtype=int),
    ]

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=70,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    # No CP means symbol length = 8 * 4 = 32.
    # ceil(70 / 32) = 3 symbols.
    rng.integers.assert_any_call(
        0,
        2,
        (8, 3),
    )


def test_ofdm_modulator_baseband_calls_ifft_on_subcarrier_axis():
    """The time-frequency grid should be transformed along axis zero."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 1), dtype=int),
    ]

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            return_value=np.zeros((32, 1), dtype=np.complex64),
        ) as ifft,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ),
    ):
        ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert ifft.call_args.kwargs == {"axis": 0}
    assert ifft.call_args.args[0].shape == (32, 1)


def test_ofdm_modulator_baseband_serializes_symbols_in_time_order():
    """Each OFDM symbol should be serialized before the next symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((4, 2), dtype=int),
    ]

    first_symbol = np.arange(16, dtype=np.float32)
    second_symbol = np.arange(100, 116, dtype=np.float32)
    ifft_output = np.column_stack(
        (
            first_symbol,
            second_symbol,
        )
    ).astype(np.complex64)

    expected_serialized = np.concatenate(
        (
            first_symbol,
            second_symbol,
        )
    ).astype(np.complex64)

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            return_value=ifft_output,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            side_effect=lambda signal, length: signal[:length],
        ) as slice_tail,
    ):
        result = ofdm_modulator_baseband(
            num_subcarriers=4,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    np.testing.assert_array_equal(
        slice_tail.call_args.args[0],
        expected_serialized,
    )
    np.testing.assert_array_equal(
        result,
        expected_serialized,
    )


def test_ofdm_modulator_baseband_enforces_requested_length():
    """The serialized OFDM waveform should be sliced to the target length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25
    rng.integers.side_effect = [
        0,
        np.zeros((8, 1), dtype=int),
    ]

    full_signal = np.arange(32, dtype=np.float32).astype(np.complex64)
    sliced = full_signal[:20]

    with (
        patch.object(
            __import__(
                MODULE_PATH,
                fromlist=["TorchSigSignalLists"],
            ).TorchSigSignalLists,
            "ofdm_subcarrier_modulations",
            ["test"],
        ),
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.np.fft.ifft",
            return_value=full_signal.reshape(32, 1),
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=sliced,
        ) as slice_tail,
    ):
        result = ofdm_modulator_baseband(
            num_subcarriers=8,
            max_num_samples=20,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    slice_tail.assert_called_once()

    np.testing.assert_array_equal(
        slice_tail.call_args.args[0],
        full_signal,
    )
    assert slice_tail.call_args.args[1] == 20
    assert result is sliced


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
def test_ofdm_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid top-level OFDM parameters should be rejected."""
    with pytest.raises(ValueError, match=expected_message):
        ofdm_modulator(
            num_subcarriers=64,
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_ofdm_modulator_creates_default_rng():
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
            f"{MODULE_PATH}.ofdm_modulator_baseband",
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
        ofdm_modulator(
            num_subcarriers=64,
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_ofdm_modulator_calculates_resampling_parameters():
    """The wrapper should derive the expected baseband length and rate."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.ofdm_modulator_baseband",
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
        ofdm_modulator(
            num_subcarriers=64,
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    # oversampling_rate = 10
    # resample_rate_ideal = 10 / 4 = 2.5
    # ceil(100 / 2.5) = 40
    baseband_modulator.assert_called_once_with(
        64,
        40,
        4,
        rng,
    )
    resampler.assert_called_once_with(
        baseband,
        2.5,
    )


def test_ofdm_modulator_slices_long_resampled_signal():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(110, dtype=np.complex64)
    sliced = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.ofdm_modulator_baseband",
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
        result = ofdm_modulator(
            num_subcarriers=64,
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
def test_ofdm_modulator_pads_signal_not_longer_than_target(
    resampled_length,
):
    """Signals no longer than the target should use the padding helper."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.ofdm_modulator_baseband",
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
        result = ofdm_modulator(
            num_subcarriers=64,
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


def test_ofdm_modulator_returns_torchsig_complex_dtype():
    """The wrapper should cast its output to the TorchSig complex dtype."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex128)
    resampled = np.ones(100, dtype=np.complex128)

    with (
        patch(
            f"{MODULE_PATH}.ofdm_modulator_baseband",
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
        result = ofdm_modulator(
            num_subcarriers=64,
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_ofdm_signal_generator_initialization():
    """The generator should configure required fields and class name."""
    metadata = {"num_subcarriers": 64}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            OFDMSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            OFDMSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = OFDMSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )
    set_class_name.assert_called_once_with("ofdm-64")

    assert generator.required_metadata_fields == [
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "num_subcarriers",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_ofdm_signal_generator_generate():
    """The generator should sample parameters and construct a Signal."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "num_subcarriers": 64,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        150,
        800,
        16,
    ]
    rng.uniform.return_value = 0.75

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(150, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.ofdm_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = OFDMSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
        call(2, 32),
    ]

    modulator.assert_called_once_with(
        64,
        800,
        10_000,
        150,
        rng,
        16,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=800,
        has_cyclic_prefix=True,
        cyclic_prefix_len=16,
    )

    assert result is expected_signal


def test_ofdm_signal_generator_labels_absent_cyclic_prefix():
    """An omitted cyclic prefix should be represented by false and zero labels."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "num_subcarriers": 64,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [150, 800]
    rng.uniform.return_value = 0.25

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    with patch(f"{MODULE_PATH}.ofdm_modulator", return_value=np.ones(150)):
        signal = OFDMSignalGenerator.generate(GeneratorStub())

    assert signal.has_cyclic_prefix is False
    assert signal.cyclic_prefix_len == 0
