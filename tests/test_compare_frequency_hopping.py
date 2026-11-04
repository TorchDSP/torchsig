"""Tests for the experimental FHSS comparison script."""

import numpy as np
import pytest

from examples.scripts import compare_frequency_hopping as comparison


def test_make_comparison_is_deterministic_and_matched():
    first = comparison.make_comparison(256, 1_000.0, 400.0, seed=7)
    second = comparison.make_comparison(256, 1_000.0, 400.0, seed=7)

    assert first.original.shape == first.hopped.shape == (256,)
    np.testing.assert_array_equal(first.hopped, second.hopped)
    np.testing.assert_array_equal(first.frequency_offsets, second.frequency_offsets)
    assert set(np.unique(first.original)) == {-1, 1}
    assert not np.array_equal(first.original, first.hopped)
    assert first.max_reconstruction_error < 1e-6
    estimated = np.angle(first.phasor[1:] * np.conj(first.phasor[:-1])) * 1_000.0 / (2 * np.pi)
    np.testing.assert_allclose(estimated, first.frequency_offsets[:-1], rtol=1e-5, atol=1e-5)


def test_make_comparison_rejects_too_few_samples():
    with pytest.raises(ValueError, match="num_samples must be at least 16"):
        comparison.make_comparison(15, 1_000.0, 400.0, seed=7)


def test_plot_comparison_saves_nonempty_image(tmp_path):
    result = comparison.make_comparison(256, 1_000.0, 400.0, seed=7)
    figure = comparison.plot_comparison(result, 1_000.0, 400.0, seed=7)
    output = tmp_path / "comparison.png"

    figure.savefig(output)

    assert output.stat().st_size > 0
