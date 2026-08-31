"""Unit tests for constellation symbol maps."""

import numpy as np
import pytest

from torchsig.signals.builders.constellation_maps import (
    all_symbol_maps,
    remove_corners,
)

EXPECTED_MAP_SIZES = {
    "ook": 2,
    "bpsk": 2,
    "qpsk": 4,
    "8psk": 8,
    "16psk": 16,
    "32psk": 32,
    "64psk": 64,
    "4ask": 4,
    "8ask": 8,
    "16ask": 16,
    "32ask": 32,
    "64ask": 64,
    "16qam": 16,
    "32qam": 32,
    "64qam": 64,
    "256qam": 256,
    "1024qam": 1024,
    "32qam_cross": 32,
    "128qam_cross": 128,
    "512qam_cross": 512,
    "16apsk": 16,
    "32apsk": 32,
}


def test_all_symbol_maps_contains_expected_modulations():
    """All documented modulation maps should be available."""
    assert set(all_symbol_maps) == set(EXPECTED_MAP_SIZES)


@pytest.mark.parametrize(
    ("modulation", "expected_size"),
    EXPECTED_MAP_SIZES.items(),
)
def test_symbol_map_has_expected_size(modulation, expected_size):
    """Each modulation should contain the expected number of symbols."""
    symbol_map = all_symbol_maps[modulation]

    assert len(symbol_map) == expected_size


@pytest.mark.parametrize("modulation", EXPECTED_MAP_SIZES)
def test_symbol_maps_have_unique_symbols(modulation):
    """Each constellation should contain no duplicate symbols."""
    symbol_map = np.asarray(all_symbol_maps[modulation])

    assert np.unique(symbol_map).size == symbol_map.size


@pytest.mark.parametrize("modulation", EXPECTED_MAP_SIZES)
def test_symbol_maps_contain_finite_values(modulation):
    """Constellation points should not contain NaN or infinity."""
    symbol_map = np.asarray(all_symbol_maps[modulation])

    assert np.all(np.isfinite(symbol_map.real))
    assert np.all(np.isfinite(symbol_map.imag))


def test_ook_contains_zero_and_one():
    """OOK should use the expected off and on symbols."""
    expected = np.array([0 + 0j, 1 + 0j])

    np.testing.assert_array_equal(
        np.sort_complex(all_symbol_maps["ook"]),
        np.sort_complex(expected),
    )


def test_bpsk_contains_negative_and_positive_one():
    """BPSK should contain symbols at -1 and 1 on the real axis."""
    expected = np.array([-1 + 0j, 1 + 0j])

    np.testing.assert_array_equal(
        np.sort_complex(all_symbol_maps["bpsk"]),
        np.sort_complex(expected),
    )


def test_qpsk_contains_expected_corner_points():
    """QPSK should contain the four corners of the unit square."""
    expected = np.array(
        [
            -1 - 1j,
            -1 + 1j,
            1 - 1j,
            1 + 1j,
        ]
    )

    np.testing.assert_array_equal(
        np.sort_complex(all_symbol_maps["qpsk"]),
        np.sort_complex(expected),
    )


@pytest.mark.parametrize(
    "modulation",
    ["8psk", "16psk", "32psk", "64psk"],
)
def test_psk_symbols_have_unit_magnitude(modulation):
    """All PSK symbols should lie on the unit circle."""
    symbol_map = all_symbol_maps[modulation]

    np.testing.assert_allclose(
        np.abs(symbol_map),
        np.ones(len(symbol_map)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("modulation", "order"),
    [
        ("8psk", 8),
        ("16psk", 16),
        ("32psk", 32),
        ("64psk", 64),
    ],
)
def test_psk_symbol_angles_are_evenly_spaced(modulation, order):
    """PSK symbols should be uniformly distributed in phase."""
    symbol_map = all_symbol_maps[modulation]

    angles = np.mod(np.angle(symbol_map), 2 * np.pi)
    sorted_angles = np.sort(angles)

    angular_differences = np.diff(np.concatenate((sorted_angles, [sorted_angles[0] + 2 * np.pi])))

    expected_difference = 2 * np.pi / order

    np.testing.assert_allclose(
        angular_differences,
        np.full(order, expected_difference),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("modulation", "order"),
    [
        ("4ask", 4),
        ("8ask", 8),
        ("16ask", 16),
        ("32ask", 32),
        ("64ask", 64),
    ],
)
def test_ask_maps_are_evenly_spaced_on_real_axis(modulation, order):
    """ASK symbols should be evenly spaced from -1 to 1."""
    symbol_map = np.asarray(all_symbol_maps[modulation])

    expected = np.linspace(-1, 1, order).astype(np.complex128)

    np.testing.assert_allclose(
        np.sort_complex(symbol_map),
        np.sort_complex(expected),
    )
    np.testing.assert_array_equal(
        symbol_map.imag,
        np.zeros(order),
    )


@pytest.mark.parametrize(
    ("modulation", "side_length"),
    [
        ("16qam", 4),
        ("64qam", 8),
        ("256qam", 16),
        ("1024qam", 32),
    ],
)
def test_square_qam_maps_use_expected_grid(modulation, side_length):
    """Square QAM maps should use a complete Cartesian grid."""
    symbol_map = np.asarray(all_symbol_maps[modulation])

    expected_axis = np.linspace(-1, 1, side_length)

    np.testing.assert_allclose(
        np.unique(symbol_map.real),
        expected_axis,
    )
    np.testing.assert_allclose(
        np.unique(symbol_map.imag),
        expected_axis,
    )


@pytest.mark.parametrize(
    "modulation",
    [
        "bpsk",
        "qpsk",
        "4ask",
        "8ask",
        "16ask",
        "32ask",
        "64ask",
        "16qam",
        "32qam",
        "64qam",
        "256qam",
        "1024qam",
        "32qam_cross",
        "128qam_cross",
        "512qam_cross",
    ],
)
def test_symmetric_constellations_have_zero_mean(modulation):
    """Symmetric constellations should be centered at the origin."""
    symbol_map = np.asarray(all_symbol_maps[modulation])

    assert np.mean(symbol_map.real) == pytest.approx(0.0, abs=1e-12)
    assert np.mean(symbol_map.imag) == pytest.approx(0.0, abs=1e-12)


def test_remove_corners_removes_outermost_square_corners():
    """The four outer corners should be removed from a 6-by-6 grid."""
    axis = np.linspace(-1, 1, 6)
    constellation = np.add(
        *map(
            np.ravel,
            np.meshgrid(axis, 1j * axis),
        )
    )

    result = remove_corners(constellation)

    corners = {
        -1 - 1j,
        -1 + 1j,
        1 - 1j,
        1 + 1j,
    }

    assert len(constellation) == 36
    assert len(result) == 32
    assert corners.isdisjoint(result)


def test_remove_corners_preserves_noncorner_points():
    """Points outside the removed corner regions should remain."""
    axis = np.linspace(-1, 1, 6)
    constellation = np.add(
        *map(
            np.ravel,
            np.meshgrid(axis, 1j * axis),
        )
    )

    result = np.asarray(remove_corners(constellation))

    expected_points = np.array(
        [
            0.6 + 1j,
            -0.6 + 1j,
            1 + 0.6j,
            1 - 0.6j,
            0.2 + 0.2j,
            -0.2 - 0.2j,
        ]
    )

    for expected_point in expected_points:
        assert np.any(np.isclose(result, expected_point))


def test_remove_corners_does_not_modify_input():
    """Removing corners should not mutate the original constellation."""
    axis = np.linspace(-1, 1, 6)
    constellation = np.add(
        *map(
            np.ravel,
            np.meshgrid(axis, 1j * axis),
        )
    )
    original = constellation.copy()

    remove_corners(constellation)

    np.testing.assert_array_equal(constellation, original)


def test_remove_corners_returns_list():
    """The helper currently returns a Python list."""
    constellation = np.array(
        [
            -1 - 1j,
            -1 + 0j,
            -1 + 1j,
            0 - 1j,
            0 + 0j,
            0 + 1j,
            1 - 1j,
            1 + 0j,
            1 + 1j,
        ]
    )

    result = remove_corners(constellation)

    assert isinstance(result, list)


@pytest.mark.parametrize(
    ("modulation", "full_grid_size"),
    [
        ("32qam_cross", 36),
        ("128qam_cross", 144),
        ("512qam_cross", 576),
    ],
)
def test_cross_qam_maps_remove_points_from_full_grid(
    modulation,
    full_grid_size,
):
    """Cross-QAM maps should contain fewer points than their source grids."""
    symbol_map = all_symbol_maps[modulation]

    assert len(symbol_map) < full_grid_size


@pytest.mark.parametrize(
    "modulation",
    [
        "32qam_cross",
        "128qam_cross",
        "512qam_cross",
    ],
)
def test_cross_qam_maps_do_not_contain_outer_corners(modulation):
    """Cross-QAM maps should exclude all four points at ±1 ± 1j."""
    symbol_map = set(all_symbol_maps[modulation])

    outer_corners = {
        -1 - 1j,
        -1 + 1j,
        1 - 1j,
        1 + 1j,
    }

    assert outer_corners.isdisjoint(symbol_map)


@pytest.mark.parametrize(
    "modulation",
    [
        "32qam_cross",
        "128qam_cross",
        "512qam_cross",
    ],
)
def test_cross_qam_maps_retain_axis_arms(modulation):
    """Cross-QAM maps should retain points along all four outer arms."""
    symbol_map = np.asarray(all_symbol_maps[modulation])

    assert np.max(symbol_map.real) == pytest.approx(1.0)
    assert np.min(symbol_map.real) == pytest.approx(-1.0)
    assert np.max(symbol_map.imag) == pytest.approx(1.0)
    assert np.min(symbol_map.imag) == pytest.approx(-1.0)

    # At each extreme, at least one point should remain that is not a corner.
    assert np.any(np.isclose(symbol_map.real, 1.0) & (np.abs(symbol_map.imag) < 1.0))
    assert np.any(np.isclose(symbol_map.real, -1.0) & (np.abs(symbol_map.imag) < 1.0))
    assert np.any(np.isclose(symbol_map.imag, 1.0) & (np.abs(symbol_map.real) < 1.0))
    assert np.any(np.isclose(symbol_map.imag, -1.0) & (np.abs(symbol_map.real) < 1.0))
