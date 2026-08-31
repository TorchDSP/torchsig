import numpy as np
import pytest

from torchsig.signals.signal_utils import (
    check_signal_class,
    random_limiting_filter_design,
)


@pytest.mark.parametrize(
    ("name", "possible_names"),
    [
        ("4fsk", ["fsk", "msk"]),
        ("am-dsb", ["am-"]),
        ("ofdm-64", ["ofdm"]),
        ("16qam", ["qam"]),
        ("tone", ["tone"]),
    ],
)
def test_check_signal_class_returns_true_for_substring_matches(name, possible_names):
    assert check_signal_class(name, possible_names)


@pytest.mark.parametrize(
    ("name", "possible_names"),
    [
        ("4fsk", ["psk", "qam"]),
        ("am-dsb", ["fm"]),
        ("ofdm-64", ["fsk"]),
        ("tone", ["chirp"]),
    ],
)
def test_check_signal_class_returns_false_without_matches(name, possible_names):
    assert not check_signal_class(name, possible_names)


@pytest.mark.parametrize("bad_name", [None, 1, 1.0, [], {}])
def test_check_signal_class_raises_type_error_for_non_string_name(bad_name):
    with pytest.raises(TypeError, match="name must be a string"):
        check_signal_class(bad_name, ["fsk"])


@pytest.mark.parametrize(
    "bad_possible_names",
    [
        None,
        "fsk",
        ("fsk",),
        ["fsk", 1],
        [None],
        {},
    ],
)
def test_check_signal_class_raises_type_error_for_invalid_possible_names(
    bad_possible_names,
):
    with pytest.raises(TypeError, match="possible_names must be a list of strings"):
        check_signal_class("4fsk", bad_possible_names)


def test_random_limiting_filter_design_returns_numpy_array():
    rng = np.random.default_rng(42)

    taps = random_limiting_filter_design(
        bandwidth=1_000,
        sample_rate=10_000,
        rng=rng,
    )

    assert isinstance(taps, np.ndarray)
    assert taps.ndim == 1
    assert taps.size > 0
    assert np.all(np.isfinite(taps))


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "error_msg"),
    [
        (0, 10_000, "bandwidth must be positive"),
        (-1, 10_000, "bandwidth must be positive"),
        (1_000, 0, "sample_rate must be positive"),
        (1_000, -1, "sample_rate must be positive"),
        (6_000, 10_000, "bandwidth must be less than sample_rate/2"),
    ],
)
def test_random_limiting_filter_design_validates_inputs(
    bandwidth,
    sample_rate,
    error_msg,
):
    with pytest.raises(ValueError, match=error_msg):
        random_limiting_filter_design(bandwidth=bandwidth, sample_rate=sample_rate)


def test_random_limiting_filter_design_uses_provided_rng_deterministically():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    taps_a = random_limiting_filter_design(1_000, 10_000, rng_a)
    taps_b = random_limiting_filter_design(1_000, 10_000, rng_b)

    np.testing.assert_allclose(taps_a, taps_b)
