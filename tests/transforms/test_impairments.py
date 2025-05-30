"""Unit Tests for transforms/impairments_narrowband and impairments_wideband.py
"""
from torchsig.transforms.impairments_wideband import WidebandImpairments
from torchsig.transforms.base_transforms import Transform

import numpy as np
import pytest


@pytest.mark.parametrize("params, is_error", [
    ( {'level' : 0}, False ),
    ( {'level' : 1}, False ),
    ( {'level' : 2}, False ),
    ( {'level' : 42}, True ),
])
def test_WidebandImpairments(params: dict, is_error: bool) -> None:
    """Test WidebandImpairments with pytest.

    Args:
        params (dict): Parameter specifying impairment level.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """   
    level = params['level']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = WidebandImpairments(
                level = level,
                seed = 42
            )
    else:
        T = WidebandImpairments(
            level = level,
            seed = 42
        )

        assert isinstance(T, WidebandImpairments) 
        assert isinstance(T.level, int) 
        assert isinstance(T.random_generator, np.random.Generator)
        for t in T.signal_transforms.transforms:
            assert isinstance(t, Transform)
        for t in T.dataset_transforms.transforms:
            assert isinstance(t, Transform)

