"""Tests for RF propagation utilities.

This module tests:
- free_space_path_loss_db: Free Space Path Loss calculation
- Edge cases: zero distance, various frequencies, various distances
- Numerical accuracy
"""

import numpy as np
import pytest

from torchsig.geo.utils.propagation import SPEED_OF_LIGHT_M_PER_S, free_space_path_loss_db


class TestPropagationConstants:
    """Tests for propagation constants."""

    def test_speed_of_light_value(self):
        """Test that SPEED_OF_LIGHT_M_PER_S is the correct value."""
        # Speed of light in vacuum is approximately 299,792,458 m/s
        assert SPEED_OF_LIGHT_M_PER_S == pytest.approx(299_792_458.0, rel=1e-6)

    def test_speed_of_light_is_positive(self):
        """Test that speed of light is positive."""
        assert SPEED_OF_LIGHT_M_PER_S > 0


class TestFreeSpacePathLoss:
    """Tests for free_space_path_loss_db function."""

    def test_basic_calculation(self):
        """Test basic FSPL calculation."""
        distance = 1000.0  # 1 km
        frequency = 2.4e9  # 2.4 GHz
        loss = free_space_path_loss_db(distance, frequency)

        # Known FSPL value for 1 km at 2.4 GHz is approximately 100.22 dB
        assert 100.0 < loss < 100.5

    def test_zero_distance(self):
        """Test FSPL at zero distance raises ValueError."""
        distance = 0.0
        frequency = 2.4e9
        # Zero distance is physically meaningless, should raise ValueError
        with pytest.raises(ValueError, match="free_space_path_loss_db.distance must be positive"):
            free_space_path_loss_db(distance, frequency)

    def test_zero_frequency(self):
        """Test FSPL at zero frequency raises ValueError."""
        distance = 1000.0
        frequency = 0.0
        # Zero frequency would cause division by zero in wavelength calculation
        # The function should handle this by raising ValueError
        with pytest.raises(ValueError, match="free_space_path_loss_db.frequency must be positive"):
            free_space_path_loss_db(distance, frequency)

    def test_very_small_frequency(self):
        """Test FSPL at very low frequency."""
        distance = 1000.0
        frequency = 1e3  # 1 kHz
        loss = free_space_path_loss_db(distance, frequency)

        # At low frequencies, wavelength is very large, so loss should be small
        assert loss < 20.0  # Very low loss at such low frequency

    def test_very_high_frequency(self):
        """Test FSPL at very high frequency."""
        distance = 1000.0
        frequency = 1e12  # 1 THz
        loss = free_space_path_loss_db(distance, frequency)

        # At high frequencies, wavelength is very small, so loss should be large
        assert loss > 140.0

    def test_distance_dependence(self):
        """Test that FSPL increases with distance."""
        frequency = 2.4e9

        loss_1km = free_space_path_loss_db(1000.0, frequency)
        loss_10km = free_space_path_loss_db(10000.0, frequency)
        loss_100km = free_space_path_loss_db(100000.0, frequency)

        # Loss should increase with distance
        assert loss_10km > loss_1km
        assert loss_100km > loss_10km

    def test_frequency_dependence(self):
        """Test that FSPL increases with frequency."""
        distance = 1000.0

        loss_1GHz = free_space_path_loss_db(distance, 1e9)
        loss_2_4GHz = free_space_path_loss_db(distance, 2.4e9)
        loss_5GHz = free_space_path_loss_db(distance, 5e9)

        # Loss should increase with frequency
        assert loss_2_4GHz > loss_1GHz
        assert loss_5GHz > loss_2_4GHz

    @pytest.mark.parametrize(
        "distance,frequency,expected_approx",
        [
            (1.0, 1e9, 20 * np.log10(4 * np.pi * 1.0 / (SPEED_OF_LIGHT_M_PER_S / 1e9))),  # 1m at 1GHz
            (100.0, 1e9, 20 * np.log10(4 * np.pi * 100.0 / (SPEED_OF_LIGHT_M_PER_S / 1e9))),  # 100m at 1GHz
            (1000.0, 2.4e9, 20 * np.log10(4 * np.pi * 1000.0 / (SPEED_OF_LIGHT_M_PER_S / 2.4e9))),  # 1km at 2.4GHz
        ],
    )
    def test_formula_correctness(self, distance, frequency, expected_approx):
        """Test that FSPL formula is correctly implemented."""
        loss = free_space_path_loss_db(distance, frequency)
        assert loss == pytest.approx(expected_approx, rel=1e-10)

    def test_propagation_constant_default(self):
        """Test default propagation constant (1.0 for vacuum)."""
        distance = 1000.0
        frequency = 2.4e9
        loss_default = free_space_path_loss_db(distance, frequency, propagation_constant=1.0)
        loss_explicit = free_space_path_loss_db(distance, frequency)

        assert loss_default == loss_explicit

    def test_propagation_constant_fiber(self):
        """Test propagation constant for fiber (n=1.5, speed = c/1.5)."""
        distance = 1000.0
        frequency = 2.4e9

        # In fiber with n=1.5, speed = c/1.5, so propagation_constant = 1/1.5 = 2/3
        propagation_constant = 2 / 3
        loss_vacuum = free_space_path_loss_db(distance, frequency, propagation_constant=1.0)
        loss_fiber = free_space_path_loss_db(distance, frequency, propagation_constant=propagation_constant)

        # In fiber, wavelength is shorter (lambda = v/f = c/(n*f)), so loss is higher
        assert loss_fiber > loss_vacuum

    @pytest.mark.parametrize("propagation_constant", [0.5, 0.6667, 0.8, 1.0, 1.2])
    def test_various_propagation_constants(self, propagation_constant):
        """Test FSPL with various propagation constants."""
        distance = 1000.0
        frequency = 2.4e9

        # Should not raise
        loss = free_space_path_loss_db(distance, frequency, propagation_constant=propagation_constant)
        assert isinstance(loss, float)

    def test_zero_propagation_constant_raises(self):
        """Test FSPL with zero propagation constant raises ValueError."""
        distance = 1000.0
        frequency = 2.4e9
        with pytest.raises(ValueError, match="free_space_path_loss_db.propagation_constant must be positive"):
            free_space_path_loss_db(distance, frequency, propagation_constant=0.0)

    def test_negative_propagation_constant_raises(self):
        """Test FSPL with negative propagation constant raises ValueError."""
        distance = 1000.0
        frequency = 2.4e9
        with pytest.raises(ValueError, match="free_space_path_loss_db.propagation_constant must be positive"):
            free_space_path_loss_db(distance, frequency, propagation_constant=-1.0)

    def test_nan_distance_raises(self):
        """Test FSPL with NaN distance raises ValueError."""
        import numpy as np

        distance = np.nan
        frequency = 2.4e9
        with pytest.raises(ValueError, match="free_space_path_loss_db.distance must be finite"):
            free_space_path_loss_db(distance, frequency)

    def test_nan_frequency_raises(self):
        """Test FSPL with NaN frequency raises ValueError."""
        import numpy as np

        distance = 1000.0
        frequency = np.nan
        with pytest.raises(ValueError, match="free_space_path_loss_db.frequency must be finite"):
            free_space_path_loss_db(distance, frequency)

    def test_inf_distance_raises(self):
        """Test FSPL with infinite distance raises ValueError."""
        import numpy as np

        distance = np.inf
        frequency = 2.4e9
        with pytest.raises(ValueError, match="free_space_path_loss_db.distance must be finite"):
            free_space_path_loss_db(distance, frequency)

    @pytest.mark.parametrize(
        "distance,frequency",
        [
            (0.1, 1e9),  # Very short distance
            (1e6, 1e9),  # 1000 km
            (100.0, 1e6),  # 100m at 1 MHz
            (500.0, 60e9),  # 500m at 60 GHz (mmWave)
        ],
    )
    def test_various_scenarios(self, distance, frequency):
        """Test FSPL with various realistic scenarios."""
        loss = free_space_path_loss_db(distance, frequency)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_returns_float(self):
        """Test that function returns a float."""
        distance = 1000.0
        frequency = 2.4e9
        loss = free_space_path_loss_db(distance, frequency)
        assert isinstance(loss, float)

    def test_negative_distance(self):
        """Test FSPL with negative distance raises ValueError."""
        distance = -1000.0
        frequency = 2.4e9
        with pytest.raises(ValueError, match="free_space_path_loss_db.distance must be positive"):
            free_space_path_loss_db(distance, frequency)

    def test_negative_frequency(self):
        """Test FSPL with negative frequency raises ValueError."""
        distance = 1000.0
        frequency = -2.4e9
        # Negative frequency is not physically realizable, should raise ValueError
        with pytest.raises(ValueError, match="free_space_path_loss_db.frequency must be positive"):
            free_space_path_loss_db(distance, frequency)

    def test_very_large_values(self):
        """Test FSPL with very large distance and frequency."""
        distance = 1e9  # 1 million km
        frequency = 1e15  # 1 PHz

        # Should compute without error (though result may be extreme)
        loss = free_space_path_loss_db(distance, frequency)
        assert isinstance(loss, float)

    def test_consistency(self):
        """Test that multiple calls with same parameters return same result."""
        distance = 1000.0
        frequency = 2.4e9

        loss1 = free_space_path_loss_db(distance, frequency)
        loss2 = free_space_path_loss_db(distance, frequency)
        loss3 = free_space_path_loss_db(distance, frequency)

        assert loss1 == loss2 == loss3
