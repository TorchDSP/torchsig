"""Unit tests for the FSK signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import scipy.signal as sp

from torchsig.signals.builders.fsk import (
    MIN_SAMPLES_PER_SYMBOL,
    FSKSignalGenerator,
    fsk_bandwidth_symbol_product,
    fsk_modulator,
    fsk_modulator_baseband,
    fsk_random_params,
    fsk_symbol_timing,
    gaussian_taps,
    get_fsk_freq_map,
    get_fsk_mod_index,
)
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.fsk"

# Every relevant class.
ALL_CLASSES = [
    ("fsk", 2),
    ("fsk", 4),
    ("fsk", 8),
    ("fsk", 16),
    ("gfsk", 2),
    ("gfsk", 4),
    ("gfsk", 8),
    ("gfsk", 16),
    ("msk", 2),
    ("gmsk", 2),
]


def occupied_bandwidth(iq, sample_rate, containment=0.99, nperseg=8192):
    """Estimate narrowest contiguous band holding `containment` of the power, in Hz."""
    nperseg = int(min(nperseg, len(iq)))
    freq, psd = sp.welch(
        iq,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=False,
        detrend=False,
        scaling="spectrum",
    )
    order = np.argsort(freq)
    freq, psd = freq[order], psd[order]

    cumulative = np.concatenate(([0.0], np.cumsum(psd)))
    cumulative /= cumulative[-1]

    best = np.inf
    for lo in range(len(psd)):
        target = cumulative[lo] + containment
        if target > 1.0:
            break
        hi = min(int(np.searchsorted(cumulative, target)), len(psd) - 1)
        best = min(best, freq[hi] - freq[lo])
    return float(best)


# ---------------------------------------------------------------- freq map


@pytest.mark.parametrize("constellation_size", [1, 2, 4, 8, 16])
def test_get_fsk_freq_map_returns_expected_values(constellation_size):
    """The frequency map should contain the documented evenly spaced values."""
    result = get_fsk_freq_map(constellation_size)

    expected = np.linspace(
        -1 + (1 / constellation_size),
        1 - (1 / constellation_size),
        constellation_size,
        endpoint=True,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_get_fsk_freq_map_is_symmetric(constellation_size):
    """Even-order FSK maps should be symmetric around zero."""
    result = get_fsk_freq_map(constellation_size)

    np.testing.assert_allclose(result, -result[::-1])


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_get_fsk_freq_map_has_expected_size(constellation_size):
    """The frequency map should contain the requested number of points."""
    result = get_fsk_freq_map(constellation_size)

    assert result.shape == (constellation_size,)


# ------------------------------------------------------------- mod index


def test_get_fsk_mod_index_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3

    with patch(
        f"{MODULE_PATH}.np.random.default_rng",
        return_value=rng,
    ) as default_rng:
        result = get_fsk_mod_index("gfsk")

    default_rng.assert_called_once_with()
    assert result == 0.3


def test_get_fsk_mod_index_gfsk():
    """GFSK should draw its modulation index from the Bluetooth range."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3

    result = get_fsk_mod_index("gfsk", rng)

    assert result == 0.3
    rng.uniform.assert_called_once_with(0.1, 0.5)


@pytest.mark.parametrize("fsk_type", ["msk", "gmsk"])
def test_get_fsk_mod_index_msk_variants(fsk_type):
    """MSK and GMSK should use a fixed modulation index of 0.5."""
    rng = MagicMock(spec=np.random.Generator)

    result = get_fsk_mod_index(fsk_type, rng)

    assert result == 0.5
    rng.uniform.assert_not_called()


def test_get_fsk_mod_index_fsk_orthogonal_branch():
    """FSK should return 1.0 when the orthogonal branch is selected."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25

    result = get_fsk_mod_index("fsk", rng)

    assert result == 1.0
    rng.uniform.assert_called_once_with(0, 1)


def test_get_fsk_mod_index_fsk_nonorthogonal_branch():
    """Non-orthogonal FSK should draw an index from the configured range."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.side_effect = [0.75, 0.9]

    result = get_fsk_mod_index("fsk", rng)

    assert result == 0.9
    assert rng.uniform.call_args_list == [
        call(0, 1),
        call(0.7, 1.01),
    ]


def test_get_fsk_mod_index_rejects_unknown_type():
    """Unsupported FSK types should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="Unexpected fsk_type: invalid",
    ):
        get_fsk_mod_index(
            "invalid",
            np.random.default_rng(42),
        )


# ----------------------------------------------------------- gaussian taps


@pytest.mark.parametrize("bt", [-1.0, -0.1, 1.1, 2.0])
def test_gaussian_taps_rejects_invalid_bt(bt):
    """The Gaussian time-bandwidth product must lie between zero and one."""
    with pytest.raises(
        ValueError,
        match="bt must be between 0.0 and 1.0",
    ):
        gaussian_taps(
            samples_per_symbol=4,
            bt=bt,
            rng=np.random.default_rng(42),
        )


def test_gaussian_taps_validates_bt_before_touching_rng():
    """Validation should not consume a draw from the generator."""
    rng = MagicMock(spec=np.random.Generator)

    with pytest.raises(ValueError, match="bt must be between 0.0 and 1.0"):
        gaussian_taps(samples_per_symbol=4, bt=1.5, rng=rng)

    rng.integers.assert_not_called()


@pytest.mark.parametrize("bt", [0.0, 0.1, 0.5, 1.0])
def test_gaussian_taps_accepts_boundary_and_interior_bt(bt):
    """The documented closed interval for BT should be accepted."""
    result = gaussian_taps(
        samples_per_symbol=4,
        bt=bt,
        span=2,
    )

    assert result.ndim == 1
    assert np.all(np.isfinite(result))


def test_gaussian_taps_creates_default_rng_when_span_omitted():
    """A default generator should be created only when the span is not given."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 2

    with patch(
        f"{MODULE_PATH}.np.random.default_rng",
        return_value=rng,
    ) as default_rng:
        gaussian_taps(
            samples_per_symbol=4,
            bt=0.3,
        )

    default_rng.assert_called_once_with()
    rng.integers.assert_called_once_with(1, 5)


def test_gaussian_taps_explicit_span_ignores_rng():
    """An explicit span should bypass the generator entirely."""
    rng = MagicMock(spec=np.random.Generator)

    result = gaussian_taps(
        samples_per_symbol=4,
        bt=0.3,
        rng=rng,
        span=3,
    )

    rng.integers.assert_not_called()
    assert result.shape == (2 * 3 * 4 + 1,)


@pytest.mark.parametrize("filter_span", [1, 2, 3, 4])
def test_gaussian_taps_has_expected_length_from_rng(filter_span):
    """The tap count should follow the selected two-sided filter span."""
    samples_per_symbol = 4
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = filter_span

    result = gaussian_taps(
        samples_per_symbol=samples_per_symbol,
        bt=0.3,
        rng=rng,
    )

    expected_length = 2 * filter_span * samples_per_symbol + 1

    assert result.shape == (expected_length,)
    rng.integers.assert_called_once_with(1, 5)


@pytest.mark.parametrize("filter_span", [1, 2, 3, 4])
def test_gaussian_taps_has_expected_length_from_span(filter_span):
    """The tap count should follow an explicitly supplied filter span."""
    samples_per_symbol = 8

    result = gaussian_taps(
        samples_per_symbol=samples_per_symbol,
        bt=0.3,
        span=filter_span,
    )

    assert result.shape == (2 * filter_span * samples_per_symbol + 1,)


def test_gaussian_taps_sum_to_one():
    """Gaussian taps should be normalized to unit sum."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 3

    result = gaussian_taps(
        samples_per_symbol=4,
        bt=0.3,
        rng=rng,
    )

    assert np.sum(result) == pytest.approx(1.0)


def test_gaussian_taps_are_symmetric():
    """The Gaussian filter should be symmetric around its center."""
    result = gaussian_taps(samples_per_symbol=4, bt=0.3, span=3)

    np.testing.assert_allclose(result, result[::-1])


def test_gaussian_taps_are_nonnegative():
    """All Gaussian pulse-shaping coefficients should be nonnegative."""
    result = gaussian_taps(samples_per_symbol=4, bt=0.3, span=2)

    assert np.all(result >= 0)


@pytest.mark.parametrize("bt", [0.1, 0.3, 0.5])
def test_gaussian_taps_narrower_bt_spreads_energy(bt):
    """A smaller BT should give a wider pulse, i.e. a smaller peak tap."""
    wide = gaussian_taps(samples_per_symbol=8, bt=bt, span=4)
    narrow = gaussian_taps(samples_per_symbol=8, bt=bt + 0.2, span=4)

    assert wide.max() < narrow.max()


# ------------------------------------------------- bandwidth symbol product


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_bandwidth_symbol_product_orthogonal_fsk_is_near_constellation_size(
    constellation_size,
):
    """Orthogonal FSK occupies about constellation_size times the symbol rate."""
    gamma = fsk_bandwidth_symbol_product(constellation_size, mod_idx=1.0)

    assert gamma == pytest.approx(constellation_size, rel=0.15)


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_bandwidth_symbol_product_increases_with_mod_index(constellation_size):
    """Wider tone spacing must widen the occupied bandwidth."""
    values = [fsk_bandwidth_symbol_product(constellation_size, mod_idx=h) for h in (0.1, 0.3, 0.5, 0.7, 1.0)]

    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("mod_idx", [0.1, 0.3, 0.5])
def test_bandwidth_symbol_product_gaussian_is_narrower_than_rectangular(mod_idx):
    """Gaussian shaping must reduce the bandwidth at the same mod index."""
    rectangular = fsk_bandwidth_symbol_product(4, mod_idx)
    gaussian = fsk_bandwidth_symbol_product(4, mod_idx, bt=0.3, gaussian_span=2)

    assert gaussian < rectangular


@pytest.mark.parametrize("bt", [0.1, 0.2, 0.3, 0.4])
def test_bandwidth_symbol_product_increases_with_bt(bt):
    """A larger time-bandwidth product must widen the spectrum."""
    lower = fsk_bandwidth_symbol_product(4, 0.3, bt=bt, gaussian_span=4)
    upper = fsk_bandwidth_symbol_product(4, 0.3, bt=bt + 0.1, gaussian_span=4)

    assert upper > lower


def test_bandwidth_symbol_product_applies_truncation_floor_to_bt():
    """A short Gaussian span cannot smooth below its own window width.

    bt_effective is floored at 1 / (3 * span), so span 1 with bt 0.1 must behave
    like a larger BT than span 4 with the same bt.
    """
    truncated = fsk_bandwidth_symbol_product(4, 0.3, bt=0.1, gaussian_span=1)
    full = fsk_bandwidth_symbol_product(4, 0.3, bt=0.1, gaussian_span=4)

    assert truncated > full


def test_bandwidth_symbol_product_span_floor_inactive_for_large_bt():
    """Above the floor the span should not change the result."""
    short = fsk_bandwidth_symbol_product(4, 0.3, bt=0.5, gaussian_span=1)
    long = fsk_bandwidth_symbol_product(4, 0.3, bt=0.5, gaussian_span=4)

    assert short == pytest.approx(long)


@pytest.mark.parametrize("gaussian_span", [None, 0])
def test_bandwidth_symbol_product_tolerates_missing_span(gaussian_span):
    """A missing or zero span must not divide by zero."""
    gamma = fsk_bandwidth_symbol_product(4, 0.3, bt=0.2, gaussian_span=gaussian_span)

    assert np.isfinite(gamma)
    assert gamma > 0


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_bandwidth_symbol_product_is_positive_over_drawn_parameters(fsk_type, constellation_size):
    """Gamma must stay positive and finite across the sampled parameter space."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = fsk_random_params(fsk_type, rng)
        gamma = fsk_bandwidth_symbol_product(constellation_size, *params)
        assert np.isfinite(gamma)
        assert gamma > 0


# ------------------------------------------------------------ random params


def test_fsk_random_params_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3
    rng.integers.return_value = 2

    with patch(
        f"{MODULE_PATH}.np.random.default_rng",
        return_value=rng,
    ) as default_rng:
        fsk_random_params("gfsk")

    default_rng.assert_called_once_with()


@pytest.mark.parametrize("fsk_type", ["fsk", "msk"])
def test_fsk_random_params_rectangular_types_have_no_pulse_parameters(fsk_type):
    """Only the Gaussian variants should carry bt and span."""
    mod_idx, bt, gaussian_span = fsk_random_params(fsk_type, np.random.default_rng(42))

    assert bt is None
    assert gaussian_span is None
    assert 0.0 < mod_idx <= 1.01


@pytest.mark.parametrize("fsk_type", ["gfsk", "gmsk"])
def test_fsk_random_params_gaussian_types_draw_bt_and_span(fsk_type):
    """GFSK and GMSK should draw a time-bandwidth product and a filter span."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3
    rng.integers.return_value = 3

    mod_idx, bt, gaussian_span = fsk_random_params(fsk_type, rng)

    assert bt == 0.3
    assert gaussian_span == 3
    assert isinstance(bt, float)
    assert isinstance(gaussian_span, int)
    rng.integers.assert_called_once_with(1, 5)
    if fsk_type == "gmsk":
        assert mod_idx == 0.5


def test_fsk_random_params_draw_order_matches_documented_sequence():
    """Draws should be modulation index, then bt, then span."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.4
    rng.integers.return_value = 2

    with patch(
        f"{MODULE_PATH}.get_fsk_mod_index",
        return_value=0.25,
    ) as get_mod_index:
        mod_idx, bt, gaussian_span = fsk_random_params("gfsk", rng)

    get_mod_index.assert_called_once_with("gfsk", rng)
    rng.uniform.assert_called_once_with(0.1, 0.5)
    assert (mod_idx, bt, gaussian_span) == (0.25, 0.4, 2)


@pytest.mark.parametrize("fsk_type", ["fsk", "gfsk", "msk", "gmsk"])
def test_fsk_random_params_is_reproducible(fsk_type):
    """The same seed should reproduce the same shaping parameters."""
    first = fsk_random_params(fsk_type, np.random.default_rng(7))
    second = fsk_random_params(fsk_type, np.random.default_rng(7))

    assert first == second


# ----------------------------------------------------------- symbol timing


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_fsk_symbol_timing_for_orthogonal_fsk(
    constellation_size,
):
    """Orthogonal FSK should produce constellation_size * oversampling_rate_nominal
    samples per symbol expected.
    """
    oversampling_rate_nominal = 4
    expected_sps = constellation_size * oversampling_rate_nominal

    _, samples_per_symbol = fsk_symbol_timing(constellation_size, oversampling_rate_nominal, mod_idx=1.0)

    assert abs(samples_per_symbol - expected_sps) <= max(1, round(0.05 * expected_sps))


def test_fsk_symbol_timing_returns_gamma_from_the_bandwidth_model():
    """The returned gamma should come from fsk_bandwidth_symbol_product."""
    with patch(
        f"{MODULE_PATH}.fsk_bandwidth_symbol_product",
        return_value=3.0,
    ) as product:
        gamma, samples_per_symbol = fsk_symbol_timing(4, 4, 0.4, bt=0.2, gaussian_span=3)

    product.assert_called_once_with(4, 0.4, 0.2, 3)
    assert gamma == 3.0
    assert samples_per_symbol == 12


@pytest.mark.parametrize("gamma", [0.05, 0.1, 0.2])
def test_fsk_symbol_timing_applies_minimum_samples_per_symbol(gamma):
    """Very narrow modulations should still be sampled at the floor."""
    with patch(
        f"{MODULE_PATH}.fsk_bandwidth_symbol_product",
        return_value=gamma,
    ):
        _, samples_per_symbol = fsk_symbol_timing(2, 4, 0.1)

    assert samples_per_symbol == MIN_SAMPLES_PER_SYMBOL


@pytest.mark.parametrize("oversampling_rate_nominal", [2, 4, 8])
def test_fsk_symbol_timing_scales_with_nominal_oversampling(
    oversampling_rate_nominal,
):
    """samples_per_symbol should track the nominal oversampling design point."""
    with patch(
        f"{MODULE_PATH}.fsk_bandwidth_symbol_product",
        return_value=5.0,
    ):
        _, samples_per_symbol = fsk_symbol_timing(4, oversampling_rate_nominal, 1.0)

    assert samples_per_symbol == round(5.0 * oversampling_rate_nominal)


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_symbol_timing_keeps_baseband_inside_nyquist(fsk_type, constellation_size):
    """Gamma / samples_per_symbol is the baseband occupancy; it must fit."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        params = fsk_random_params(fsk_type, rng)
        gamma, samples_per_symbol = fsk_symbol_timing(constellation_size, 4, *params)
        assert samples_per_symbol >= MIN_SAMPLES_PER_SYMBOL
        assert gamma / samples_per_symbol < 0.5


# -------------------------------------------------------- baseband modulator


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_fsk_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -10])
def test_fsk_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive nominal oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_fsk_modulator_baseband_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(1.0, None, None),
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(16),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(32, dtype=np.complex64),
        ),
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=32,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (32,)


def test_fsk_modulator_baseband_draws_parameters_when_not_supplied():
    """Without `params` the baseband modulator should draw its own."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(0.75, None, None),
        ) as random_params,
        patch(
            f"{MODULE_PATH}.get_fsk_freq_map",
            return_value=np.array([-0.5, 0.5]),
        ) as get_freq_map,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(8),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(16, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=2,
            fsk_type="fsk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    random_params.assert_called_once_with("fsk", rng)
    get_freq_map.assert_called_once_with(2)


def test_fsk_modulator_baseband_reuses_supplied_parameters():
    """Supplied `params` must be used verbatim and consume no draws.

    fsk_modulator sizes the resampling ratio from the parameters it drew, so the
    baseband stage has to modulate with those same values.
    """
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])
    params = (0.42, 0.3, 2)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
        ) as random_params,
        patch(
            f"{MODULE_PATH}.gaussian_taps",
            return_value=np.array([0.25, 0.5, 0.25]),
        ) as gaussian,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(16),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(64, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="gfsk",
            max_num_samples=64,
            oversampling_rate_nominal=4,
            rng=rng,
            params=params,
        )

    random_params.assert_not_called()
    rng.uniform.assert_not_called()

    expected_sps = fsk_symbol_timing(4, 4, *params)[1]
    gaussian.assert_called_once_with(expected_sps, 0.3, span=2)


def test_fsk_modulator_baseband_uses_symbol_timing_for_samples_per_symbol():
    """The pulse length should come from fsk_symbol_timing, not mod_order."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    with (
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(2.5, 10),
        ) as symbol_timing,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(64),
        ) as upfirdn,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(64, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=64,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(0.8, None, None),
        )

    symbol_timing.assert_called_once_with(4, 4, 0.8, None, None)

    np.testing.assert_array_equal(upfirdn.call_args.args[0], np.ones(10))
    assert upfirdn.call_args.kwargs == {"up": 10, "down": 1}


def test_fsk_modulator_baseband_rectangular_pulse_shape():
    """Plain FSK should use a rectangular pulse without Gaussian filtering."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    with (
        patch(
            f"{MODULE_PATH}.gaussian_taps",
        ) as gaussian,
        patch(
            f"{MODULE_PATH}.sp.convolve",
        ) as convolve,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(64),
        ) as upfirdn,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(64, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=64,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(1.0, None, None),
        )

    gaussian.assert_not_called()
    convolve.assert_not_called()

    expected_sps = fsk_symbol_timing(4, 4, 1.0)[1]
    np.testing.assert_array_equal(
        upfirdn.call_args.args[0],
        np.ones(expected_sps),
    )


@pytest.mark.parametrize("fsk_type", ["gfsk", "gmsk"])
def test_fsk_modulator_baseband_gaussian_filter_branch(fsk_type):
    """GFSK and GMSK should convolve the rectangular pulse with Gaussian taps."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])
    params = (0.5, 0.3, 2)

    taps = np.array([0.25, 0.5, 0.25])

    with (
        patch(
            f"{MODULE_PATH}.gaussian_taps",
            return_value=taps,
        ) as gaussian,
        patch(
            f"{MODULE_PATH}.sp.convolve",
            return_value=np.ones(10),
        ) as convolve,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(32),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(32, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=2,
            fsk_type=fsk_type,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
            params=params,
        )

    expected_sps = fsk_symbol_timing(2, 4, *params)[1]

    gaussian.assert_called_once_with(expected_sps, 0.3, span=2)
    np.testing.assert_array_equal(convolve.call_args.args[0], taps)
    np.testing.assert_array_equal(
        convolve.call_args.args[1],
        np.ones(expected_sps),
    )


def test_fsk_modulator_baseband_scales_frequency_map_by_symbol_timing():
    """Symbols should be scaled by constellation_size / samples_per_symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 3])

    frequency_map = np.array([-0.75, -0.25, 0.25, 0.75])

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_freq_map",
            return_value=frequency_map,
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(64),
        ) as upfirdn,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(64, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=64,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(1.0, None, None),
        )

    expected_symbols = frequency_map[[0, 3]] * 4 / 16

    np.testing.assert_allclose(upfirdn.call_args.args[1], expected_symbols)


def test_fsk_modulator_baseband_rounds_symbol_count_up():
    """The symbol count should round up so the record is never short."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1, 2, 3, 0])

    with (
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(96),
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=np.ones(80, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=80,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(1.0, None, None),
        )

    # samples_per_symbol = 16, rectangular pulse length 16
    # ceil((80 - 16 + 1) / 16) = ceil(65 / 16) = 5, where floor gave 4
    rng.integers.assert_called_once_with(0, 4, 5)


def test_fsk_modulator_baseband_generates_at_least_one_symbol():
    """Short signals should still generate at least one symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    with (
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(16),
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=np.ones(4, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=4,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(1.0, None, None),
        )

    rng.integers.assert_called_once_with(0, 4, 1)


def test_fsk_modulator_baseband_matches_phase_formula():
    """The modulated output should match cumulative phase integration."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    modulation_index = 0.5
    filtered = np.array([0.1, -0.2, 0.3, -0.4])

    with (
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
    ):
        result = fsk_modulator_baseband(
            constellation_size=1,
            fsk_type="msk",
            max_num_samples=4,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(modulation_index, None, None),
        )

    expected = np.exp(np.cumsum(filtered * 1j * modulation_index * np.pi))

    pad.assert_not_called()
    slice_tail.assert_not_called()

    np.testing.assert_allclose(result, expected)


def test_fsk_modulator_baseband_has_unit_magnitude():
    """Continuous-phase FSK samples should lie on the unit circle."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.linspace(-0.25, 0.25, 16)

    with patch(
        f"{MODULE_PATH}.sp.upfirdn",
        return_value=filtered,
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="msk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(0.5, None, None),
        )

    np.testing.assert_allclose(
        np.abs(result),
        np.ones(16),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fsk_modulator_baseband_slices_long_signal():
    """An oversized modulated signal should be sliced from its tail."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.ones(20)
    sliced = np.ones(16, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=sliced,
        ) as slice_tail,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(1.0, None, None),
        )

    slice_tail.assert_called_once()

    np.testing.assert_allclose(
        slice_tail.call_args.args[0],
        np.exp(np.cumsum(filtered * 1j * np.pi)),
    )
    assert slice_tail.call_args.args[1] == 16

    pad.assert_not_called()
    assert result is sliced


def test_fsk_modulator_baseband_pads_short_signal():
    """A short modulated signal should be padded to the target length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    padded = np.ones(16, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(8),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
            params=(1.0, None, None),
        )

    pad.assert_called_once()
    assert pad.call_args.args[1] == 16
    slice_tail.assert_not_called()
    assert result is padded


# ------------------------------------------------------------- modulator


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
def test_fsk_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid top-level modulation parameters should be rejected."""
    with pytest.raises(ValueError, match=expected_message):
        fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_fsk_modulator_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(1.0, None, None),
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(40, dtype=np.complex64),
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
        fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_fsk_modulator_derives_resampling_rate_from_symbol_timing():
    """resample_rate_ideal should be oversampling_rate * gamma / sps."""
    rng = np.random.default_rng(42)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(0.5, 0.25, 2),
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(2.0, 8),
        ) as symbol_timing,
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(40, dtype=np.complex64),
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
        fsk_modulator(
            constellation_size=4,
            fsk_type="gfsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    symbol_timing.assert_called_once_with(4, 4, 0.5, 0.25, 2)

    # oversampling_rate = 10, gamma = 2, samples_per_symbol = 8
    # resample_rate_ideal = 10 * 2 / 8 = 2.5
    # max_num_samples = ceil(100 / 2.5) + 8 = 48
    assert resampler.call_args.args[1] == pytest.approx(2.5)
    baseband_modulator.assert_called_once_with(
        4,
        "gfsk",
        48,
        4,
        rng,
        params=(0.5, 0.25, 2),
    )


def test_fsk_modulator_forwards_its_own_drawn_parameters():
    """The parameters used for sizing must be the ones handed to the baseband."""
    rng = np.random.default_rng(11)
    params = (0.31, 0.22, 3)
    resampled = np.ones(64, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=params,
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(64, dtype=np.complex64),
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
        fsk_modulator(
            constellation_size=4,
            fsk_type="gfsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=64,
            rng=rng,
        )

    assert baseband_modulator.call_args.kwargs["params"] is params


def test_fsk_modulator_requests_margin_above_the_target_length():
    """The baseband request should exceed num_samples / rate by one symbol.

    The margin means the resampled record is sliced rather than zero padded, so
    the signal fills the whole labelled duration.
    """
    rng = np.random.default_rng(42)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(1.0, None, None),
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(56, dtype=np.complex64),
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
        fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    # resample_rate_ideal = 10 * 4 / 16 = 2.5, ceil(100 / 2.5) + 16 = 56
    assert baseband_modulator.call_args.args[2] == 56


def test_fsk_modulator_uses_minimum_baseband_length():
    """The baseband request should never fall below one symbol."""
    rng = np.random.default_rng(42)
    resampled = np.ones(1, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(0.5, None, None),
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(1.3, 5),
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(6, dtype=np.complex64),
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
        fsk_modulator(
            constellation_size=2,
            fsk_type="msk",
            bandwidth=1,
            sample_rate=10_000,
            num_samples=1,
            rng=rng,
        )

    assert baseband_modulator.call_args.args[2] >= 5


def test_fsk_modulator_applies_resampling_amplitude_correction():
    """The resampled signal should be scaled by the inverse resample rate."""
    rng = np.random.default_rng(42)
    resampled = np.full(100, 10 + 5j, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(1.0, None, None),
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(56, dtype=np.complex64),
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled.copy(),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            side_effect=lambda signal, _length: signal,
        ),
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    expected = resampled / 2.5

    np.testing.assert_allclose(
        result,
        expected.astype(TorchSigComplexDataType),
    )


def test_fsk_modulator_slices_long_resampled_signal():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    resampled = np.ones(110, dtype=np.complex64)
    sliced = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(1.0, None, None),
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(56, dtype=np.complex64),
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled.copy(),
        ),
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
            return_value=sliced,
        ) as slice_signal,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad_signal,
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    np.testing.assert_allclose(
        slice_signal.call_args.args[0],
        resampled / 2.5,
    )
    assert slice_signal.call_args.args[1] == 100

    pad_signal.assert_not_called()
    assert result.dtype == np.dtype(TorchSigComplexDataType)


@pytest.mark.parametrize("resampled_length", [90, 100])
def test_fsk_modulator_pads_signal_not_longer_than_target(
    resampled_length,
):
    """A signal no longer than the target should use the padding helper."""
    rng = np.random.default_rng(42)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_random_params",
            return_value=(1.0, None, None),
        ),
        patch(
            f"{MODULE_PATH}.fsk_symbol_timing",
            return_value=(4.0, 16),
        ),
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=np.ones(56, dtype=np.complex64),
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled.copy(),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad_signal,
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
        ) as slice_signal,
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    np.testing.assert_allclose(
        pad_signal.call_args.args[0],
        resampled / 2.5,
    )
    assert pad_signal.call_args.args[1] == 100

    slice_signal.assert_not_called()
    assert result.shape == (100,)


def test_fsk_modulator_returns_torchsig_complex_dtype():
    """The wrapper should return the configured TorchSig complex dtype."""
    result = fsk_modulator(
        constellation_size=4,
        fsk_type="fsk",
        bandwidth=1_000,
        sample_rate=10_000,
        num_samples=4096,
        rng=np.random.default_rng(42),
    )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


# ------------------------------------------------- end to end signal checks


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_modulator_returns_exactly_num_samples(fsk_type, constellation_size):
    """The record length must match the requested duration exactly."""
    for num_samples in (1024, 4099, 32768):
        result = fsk_modulator(
            constellation_size=constellation_size,
            fsk_type=fsk_type,
            bandwidth=1e6,
            sample_rate=10e6,
            num_samples=num_samples,
            rng=np.random.default_rng(1),
        )

        assert result.shape == (num_samples,)


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_modulator_record_contains_no_zero_padding(fsk_type, constellation_size):
    """Rounding the symbol count up should remove the zero padded case.

    A padded record is shorter than the duration its label claims.
    """
    result = fsk_modulator(
        constellation_size=constellation_size,
        fsk_type=fsk_type,
        bandwidth=1e6,
        sample_rate=10e6,
        num_samples=32768,
        rng=np.random.default_rng(2),
    )

    assert not np.any(np.abs(result) < 1e-12)


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_modulator_occupied_bandwidth_matches_request(fsk_type, constellation_size):
    """The realized bandwidth should match the requested bandwidth.

    This is the property the closed-form bandwidth model exists to provide: the
    value passed here is what the caller writes into the Signal metadata and
    therefore into the bounding box. Gamma is an empirical fit, so approximate.
    """
    sample_rate = 10e6
    bandwidth = 1e6

    for seed in (0, 1, 2, 3):
        iq = fsk_modulator(
            constellation_size=constellation_size,
            fsk_type=fsk_type,
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=65536,
            rng=np.random.default_rng(seed),
        )

        ratio = occupied_bandwidth(iq, sample_rate) / bandwidth

        assert 0.75 < ratio < 1.30, f"{constellation_size}{fsk_type} seed {seed}: ratio {ratio:.3f}"


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_modulator_bandwidth_scales_with_the_request(fsk_type, constellation_size):
    """Doubling the requested bandwidth should double the occupied bandwidth."""
    sample_rate = 10e6
    narrow = fsk_modulator(
        constellation_size=constellation_size,
        fsk_type=fsk_type,
        bandwidth=0.5e6,
        sample_rate=sample_rate,
        num_samples=65536,
        rng=np.random.default_rng(5),
    )
    wide = fsk_modulator(
        constellation_size=constellation_size,
        fsk_type=fsk_type,
        bandwidth=1.0e6,
        sample_rate=sample_rate,
        num_samples=65536,
        rng=np.random.default_rng(5),
    )

    measured = occupied_bandwidth(wide, sample_rate) / occupied_bandwidth(narrow, sample_rate)

    assert measured == pytest.approx(2.0, rel=0.15)


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_modulator_is_reproducible_for_a_seed(fsk_type, constellation_size):
    """The same seed should produce bit-identical samples."""
    kwargs = dict(
        constellation_size=constellation_size,
        fsk_type=fsk_type,
        bandwidth=1e6,
        sample_rate=10e6,
        num_samples=8192,
    )

    first = fsk_modulator(rng=np.random.default_rng(99), **kwargs)
    second = fsk_modulator(rng=np.random.default_rng(99), **kwargs)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(("fsk_type", "constellation_size"), ALL_CLASSES)
def test_fsk_modulator_output_is_finite(fsk_type, constellation_size):
    """No configuration should produce NaN or infinite samples."""
    result = fsk_modulator(
        constellation_size=constellation_size,
        fsk_type=fsk_type,
        bandwidth=1e6,
        sample_rate=10e6,
        num_samples=8192,
        rng=np.random.default_rng(4),
    )

    assert np.all(np.isfinite(result.view(np.float32)))


# ------------------------------------------------------------- generator


def test_fsk_signal_generator_initialization():
    """The generator should configure required fields and its class name."""
    metadata = {
        "fsk_type": "gfsk",
        "constellation_size": 4,
    }

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            FSKSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            FSKSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = FSKSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )
    set_class_name.assert_called_once_with("4gfsk")

    assert generator.required_metadata_fields == [
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "fsk_type",
        "constellation_size",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_fsk_signal_generator_generate():
    """The generator should sample parameters and construct a Signal."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "fsk_type": "gmsk",
        "constellation_size": 2,
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
            f"{MODULE_PATH}.fsk_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = FSKSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
    ]

    modulator.assert_called_once_with(
        2,
        "gmsk",
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


def test_fsk_signal_generator_generate_labels_the_requested_bandwidth():
    """The Signal bandwidth must be the value handed to the modulator.

    The bounding box is derived from this field, so a mismatch between it and the
    modulator argument is what the bandwidth model is there to prevent.
    """
    metadata = {
        "sample_rate": 10_000_000,
        "bandwidth_min": 1_000_000,
        "bandwidth_max": 1_000_000,
        "fsk_type": "gfsk",
        "constellation_size": 4,
        "signal_duration_in_samples_min": 8_192,
        "signal_duration_in_samples_max": 8_192,
    }

    rng = np.random.default_rng(0)

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    with patch(
        f"{MODULE_PATH}.Signal",
        side_effect=lambda **kwargs: kwargs,
    ):
        signal = FSKSignalGenerator.generate(GeneratorStub())

    assert signal["bandwidth"] == 1_000_000
    assert signal["center_freq"] == 0
    assert signal["data"].shape == (8_192,)

    ratio = occupied_bandwidth(signal["data"], metadata["sample_rate"]) / signal["bandwidth"]

    assert 0.75 < ratio < 1.30
