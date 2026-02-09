"""Unit Tests for transforms impairments"""

from torchsig.transforms.impairments import Impairments
from torchsig.transforms.base_transforms import Transform

import numpy as np
import pytest


@pytest.mark.parametrize(
    "params, is_error",
    [
        ({"level": 0}, False),
        ({"level": 1}, False),
        ({"level": 2}, False),
        ({"level": 42}, True),
    ],
)
def test_Impairments(params: dict, is_error: bool) -> None:
    """Test Impairments with pytest.

    Args:
        params (dict): Parameter specifying impairment level.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    level = params["level"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Impairments(level=level, seed=42)
    else:
        T = Impairments(level=level, seed=42)

        assert isinstance(T, Impairments)
        assert isinstance(T.level, int)
        assert isinstance(T.random_generator, np.random.Generator)
        for t in T.signal_transforms.transforms:
            assert isinstance(t, Transform)
        for t in T.dataset_transforms.transforms:
            assert isinstance(t, Transform)
