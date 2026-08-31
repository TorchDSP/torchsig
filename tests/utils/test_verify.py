import numpy as np
import pytest

from torchsig.transforms.base_transforms import Transform
from torchsig.utils.verify import (
    verify_bounds,
    verify_dict,
    verify_distribution_list,
    verify_float,
    verify_int,
    verify_list,
    verify_numpy_array,
    verify_str,
    verify_transforms,
)


def test_verify_bounds_returns_value_when_in_range():
    assert verify_bounds(5, "x", low=0, high=10) == 5


def test_verify_bounds_clips_low():
    assert verify_bounds(-1, "x", low=0, clip_low=True) == 0


def test_verify_bounds_clips_high():
    assert verify_bounds(11, "x", high=10, clip_high=True) == 10


def test_verify_bounds_raises_when_below_range():
    with pytest.raises(ValueError, match="out of bounds"):
        verify_bounds(-1, "x", low=0)


def test_verify_bounds_raises_when_above_range():
    with pytest.raises(ValueError, match="out of bounds"):
        verify_bounds(11, "x", high=10)


def test_verify_bounds_respects_exclusive_bounds():
    with pytest.raises(ValueError):
        verify_bounds(0, "x", low=0, exclude_low=True)

    with pytest.raises(ValueError):
        verify_bounds(10, "x", high=10, exclude_high=True)


def test_verify_int_accepts_valid_value():
    assert verify_int(5, "x") == 5


def test_verify_int_rejects_non_int():
    with pytest.raises(TypeError):
        verify_int(1.2, "x")


def test_verify_int_passes_bounds_to_verify_bounds():
    with pytest.raises(ValueError):
        verify_int(-1, "x")


def test_verify_float_accepts_float():
    assert verify_float(1.5, "x") == 1.5


def test_verify_float_converts_int():
    assert verify_float(3, "x") == 3.0


def test_verify_float_rejects_invalid_type():
    with pytest.raises(TypeError):
        verify_float("1.0", "x")


def test_verify_float_checks_bounds():
    with pytest.raises(ValueError):
        verify_float(-1.0, "x")


def test_verify_str_lowercases_and_strips():
    assert verify_str("  HELLO ", "name") == "hello"


def test_verify_str_upper():
    assert verify_str("abc", "name", str_format="upper") == "ABC"


def test_verify_str_title():
    assert verify_str("hello world", "name", str_format="title") == "Hello World"


def test_verify_str_valid_values():
    assert verify_str("A", "mode", valid=["a", "b"]) == "a"


def test_verify_str_invalid_value():
    with pytest.raises(ValueError):
        verify_str("c", "mode", valid=["a", "b"])


def test_verify_str_invalid_type():
    with pytest.raises(TypeError):
        verify_str(123, "mode")


def test_verify_distribution_list_allows_none():
    assert verify_distribution_list(None, 2, "dist", "classes") is None


def test_verify_distribution_list_returns_valid_distribution():
    dist = np.array([0.2, 0.8])

    result = verify_distribution_list(dist, 2, "dist", "classes")

    np.testing.assert_array_equal(result, dist)


def test_verify_distribution_list_normalizes(capsys):
    dist = np.array([2.0, 2.0])

    result = verify_distribution_list(dist, 2, "dist", "classes")

    np.testing.assert_allclose(result, [0.5, 0.5])

    assert "automatically normalizing" in capsys.readouterr().out


def test_verify_distribution_list_invalid_length():
    with pytest.raises(ValueError):
        verify_distribution_list(np.array([1.0]), 2, "dist", "classes")


def test_verify_list_accepts_list():
    assert verify_list([1, 2], "x") == [1, 2]


def test_verify_list_converts_tuple():
    assert verify_list((1, 2), "x") == [1, 2]


def test_verify_list_converts_numpy_array():
    assert verify_list(np.array([1, 2]), "x") == [1, 2]


def test_verify_list_rejects_non_list():
    with pytest.raises(TypeError):
        verify_list(5, "x")


def test_verify_list_detects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        verify_list([1, 2, 1], "x", no_duplicates=True)


def test_verify_list_checks_item_type():
    with pytest.raises(TypeError):
        verify_list([1, "2"], "x", data_type=int)


def test_verify_numpy_array_accepts_array():
    arr = np.array([1.0, 2.0])

    result = verify_numpy_array(arr, "x")

    assert result is arr


def test_verify_numpy_array_rejects_non_array():
    with pytest.raises(TypeError):
        verify_numpy_array(3, "x")


def test_verify_numpy_array_checks_min_length():
    with pytest.raises(ValueError):
        verify_numpy_array(np.array([1]), "x", min_length=2)


def test_verify_numpy_array_checks_max_length():
    with pytest.raises(ValueError):
        verify_numpy_array(np.array([1, 2, 3]), "x", max_length=2)


def test_verify_numpy_array_checks_exact_length():
    with pytest.raises(ValueError):
        verify_numpy_array(np.array([1, 2]), "x", exact_length=3)


def test_verify_numpy_array_checks_dtype():
    with pytest.raises(ValueError):
        verify_numpy_array(np.array(["a"]), "x", data_type=float)


def test_verify_numpy_array_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        verify_numpy_array(np.array([1.0, np.nan]), "x")


def test_verify_numpy_array_rejects_inf():
    with pytest.raises(ValueError, match="np.inf"):
        verify_numpy_array(np.array([1.0, np.inf]), "x")


def test_verify_dict_accepts_valid_dict():
    d = {"a": 1}

    assert verify_dict(d, "d") is d


def test_verify_dict_rejects_non_dict():
    with pytest.raises(TypeError):
        verify_dict([], "d")


def test_verify_dict_missing_key():
    with pytest.raises(ValueError):
        verify_dict({}, "d", required_keys=["a"], required_types=[int])


def test_verify_dict_wrong_type():
    with pytest.raises(ValueError):
        verify_dict({"a": "1"}, "d", required_keys=["a"], required_types=[int])


class DummyTransform(Transform):
    def __call__(self, x):
        return x


def test_verify_transforms_none():
    assert verify_transforms(None) == []


def test_verify_transforms_single():
    t = DummyTransform()

    assert verify_transforms(t) == [t]


def test_verify_transforms_list():
    t = DummyTransform()

    assert verify_transforms([t]) == [t]


def test_verify_transforms_rejects_non_callable():
    with pytest.raises(TypeError):
        verify_transforms([object()])


def test_verify_transforms_rejects_non_list():
    with pytest.raises(TypeError):
        verify_transforms(123)
