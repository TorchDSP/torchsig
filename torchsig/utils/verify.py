""" Data verification and error checking utils
"""

from __future__ import annotations

__all__ = [
    "verify_int",
    "verify_float",
    "verify_str",
    "verify_distribution_list",
    "verify_list",
    "verify_numpy_array",
    "verify_transforms",
    "verify_target_transforms",
    "verify_dict"
]

# TorchSig

# Third Party
import numpy as np

# Built In
from typing import List, Callable, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from torchsig.transforms.base_transforms import Transform
    from torchsig.transforms.target_transforms import TargetTransform

def verify_bounds(
    a: float | int,
    name: str,
    low: float | int = None, 
    high: float | int = None, 
    clip_low: bool = False, 
    clip_high: bool = False,
    exclude_low: bool = False, # use less than or equal
    exclude_high: bool = False, # use greater than or equal
):
    """
    Verifies if the value `a` is within the specified bounds (low, high). 
    If `a` is outside the bounds, raises a ValueError. Optionally, clips the value 
    of `a` to the bounds if it is outside the specified range.

    Args:
        a (float | int): The value to be checked.
        name (str): The name of the value to be used in error messages.
        low (float | int, optional): The lower bound of the value. Defaults to None.
        high (float | int, optional): The upper bound of the value. Defaults to None.
        clip_low (bool, optional): If True, the value will be clipped to `low` if it is below `low`. Defaults to False.
        clip_high (bool, optional): If True, the value will be clipped to `high` if it exceeds `high`. Defaults to False.
        exclude_low (bool, optional): If True, `a` must be strictly greater than `low`. Defaults to False.
        exclude_high (bool, optional): If True, `a` must be strictly less than `high`. Defaults to False.

    Raises:
        ValueError: If `a` is out of bounds and `clip_low` or `clip_high` are not enabled.

    Returns:
        float | int: The value `a`, either adjusted to the bounds or left unchanged.
    """
    too_low = low is not None and ((exclude_low and a <= low) or (a < low))
    too_high = high is not None and ((exclude_high and a >= high) or (a > high))

    if (too_low and not clip_low) or (too_high and not clip_high):
        o1 = "<" if exclude_low else "<="
        o2 = "<" if exclude_high else "<="
        bounds = f"{'-inf' if low is None else low} {o1} {name} {o2} {'inf' if high is None else high}"
        raise ValueError(f"{name}={a} is out of bounds. Must be {bounds}")
    
    if too_low and clip_low:
        a = low

    if too_high and clip_high:
        a = high

    return a

def verify_int(
    a: int, 
    name: str, 
    low: int = 0, 
    high: int = None, 
    clip_low: bool = False, 
    clip_high: bool = False,
    exclude_low: bool = False, # use less than or equal
    exclude_high: bool = False, # use greater than or equal
) -> int:
    """
    Verifies that the value `a` is an integer and within the specified bounds.

    Args:
        a (int): The value to be checked.
        name (str): The name of the value to be used in error messages.
        low (int, optional): The lower bound of the value. Defaults to 0.
        high (int, optional): The upper bound of the value. Defaults to None.
        clip_low (bool, optional): If True, the value will be clipped to `low` if it is below `low`. Defaults to False.
        clip_high (bool, optional): If True, the value will be clipped to `high` if it exceeds `high`. Defaults to False.
        exclude_low (bool, optional): If True, `a` must be strictly greater than `low`. Defaults to False.
        exclude_high (bool, optional): If True, `a` must be strictly less than `high`. Defaults to False.

    Raises:
        ValueError: If `a` is not an integer or out of bounds.

    Returns:
        int: The verified integer value `a`.
    """

    if not isinstance(a, int):
        raise ValueError(f"{name} is not type int: {type(a)}")

    return verify_bounds(
        a = a,
        name = name,
        low = low,
        high = high,
        clip_low = clip_low,
        clip_high = clip_high,
        exclude_low = exclude_low,
        exclude_high = exclude_high
    )

    

def verify_float(
    f: float,
    name: str,
    low: float = 0.0,
    high: float = None,
    clip_low: bool = False,
    clip_high: bool = False,
    exclude_low: bool = False, # use less than or equal
    exclude_high: bool = False, # use greater than or equal
) -> float:
    """
    Verifies that the value `f` is a float and within the specified bounds.

    Args:
        f (float): The value to be checked.
        name (str): The name of the value to be used in error messages.
        low (float, optional): The lower bound of the value. Defaults to 0.0.
        high (float, optional): The upper bound of the value. Defaults to None.
        clip_low (bool, optional): If True, the value will be clipped to `low` if it is below `low`. Defaults to False.
        clip_high (bool, optional): If True, the value will be clipped to `high` if it exceeds `high`. Defaults to False.
        exclude_low (bool, optional): If True, `f` must be strictly greater than `low`. Defaults to False.
        exclude_high (bool, optional): If True, `f` must be strictly less than `high`. Defaults to False.

    Raises:
        ValueError: If `f` is not a float or out of bounds.

    Returns:
        float: The verified float value `f`.
    """
    if isinstance(f, int):
        f = float(f)
    elif not isinstance(f, float):
        raise ValueError(f"{name} is not type float: {type(f)}")

    return verify_bounds(
        a = f,
        name = name,
        low = low,
        high = high,
        clip_low = clip_low,
        clip_high = clip_high,
        exclude_low = exclude_low,
        exclude_high = exclude_high
    )
            
# lower, upper, title
def verify_str(
    s: str, 
    name: str, 
    valid: List[str] = [], 
    str_format: str = "lower"
) -> str:
    """
    Verifies that the value `s` is a string and optionally formats it according to the specified format.

    Args:
        s (str): The value to be checked.
        name (str): The name of the value to be used in error messages.
        valid (List[str], optional): A list of valid string values. Defaults to an empty list.
        str_format (str, optional): The format for the string. Can be "lower", "upper", or "title". Defaults to "lower".

    Raises:
        ValueError: If `s` is not a string or if it is not in the list of valid values.

    Returns:
        str: The verified string value `s` in the specified format.
    """
    if not isinstance(s, str):
        raise ValueError(f"{name} is not a str: {type(s)}")

    # remove trailing or leading whitespace
    s = s.strip()
    
    # convert string to correct format
    if str_format == "lower":
        s = s.lower()
    elif str_format == "upper":
        s = s.upper()
    elif str_format == "title":
        s = s.title()

    
    if len(valid) > 0 and s not in valid:
        raise ValueError(f"Invalid {name}: {s}. Must be in {valid}")

    return s

def verify_distribution_list(
    distro: List[float], 
    required_length: int, 
    distro_name: str, 
    list_name: str
) -> List[float]:
    """Verifies and normalizes a given distribution list.

    If the distribution list is `None`, it assumes a uniform distribution and returns it as is.
    If the distribution list is not of the required length or does not sum to 1.0, it raises an error or normalizes the list to sum to 1.0.

    Args:
        distro (List[float]): The distribution list to verify. Can be `None` for a uniform distribution.
        required_length (int): The expected length of the distribution list.
        distro_name (str): The name of the distribution list (used for error messages).
        list_name (str): The name of the list this distribution corresponds to (used for error messages).

    Returns:
        List[float]: The verified and possibly normalized distribution list.

    Raises:
        ValueError: If the distribution list is not of the required length or does not sum to 1.0 and cannot be normalized.
    """
    # None means uniform distribution, allowed
    if distro is None:
        return distro
    
    if len(distro) != required_length:
        raise ValueError(f"{distro_name} = {len(distro)} must be same length as {list_name} = {required_length}")
    
    if np.sum(distro) != 1.0:
        # automatically normalize distribution, warn users of this behavior
        # warnings.warn(f"{distro_name} does not sum to 1.0, automatically normalizing.", UserWarning, stacklevel=3)
        print(f"{distro_name} does not sum to 1.0, automatically normalizing.")
        distro = distro / np.sum(distro, dtype=float)

    return distro

def verify_list(
    l: list,
    name: str,
    no_duplicates: bool = False,
    data_type = None,
) -> list:
    """
    Verifies that the value `l` is a list and optionally checks for duplicates or verifies item types.

    Args:
        l (list): The value to be checked.
        name (str): The name of the value to be used in error messages.
        no_duplicates (bool, optional): If True, raises an error if the list contains duplicates. Defaults to False.
        data_type (type, optional): The type each item in the list should have. Defaults to None.

    Raises:
        ValueError: If `l` is not a list, if it contains duplicates (when `no_duplicates=True`), 
                    or if any item in the list is not of the required type.

    Returns:
        list: The verified list `l`.
    """
    if isinstance(l, np.ndarray):
        l = l.tolist()
    elif isinstance(l, tuple):
        l = list(l)
    elif not isinstance(l, list):
        raise ValueError(f"{name} is not a list: {type(l)}")

    if no_duplicates:
        if len(l) != len(set(l)):
            counts = Counter(l)
            duplicates = [item for item, count in counts.items() if count > 1]
            raise ValueError(f"{name} has duplicates {duplicates}")
    
    if data_type is not None:
        for i,item in enumerate(l):
            if not isinstance(item, data_type):
                raise ValueError(f"{name}[{i}] = {item} is not correct data type {data_type}: {type(item)}")

    return l

def verify_numpy_array(
    n: np.ndarray,
    name: str,
    min_length: int = None,
    max_length: int = None,
    exact_length: int = None,
    data_type = None,
) -> np.ndarray:
    """
    Verifies that the value `n` is a NumPy array and optionally checks its length or item types.

    Args:
        n (np.ndarray): The value to be checked.
        name (str): The name of the value to be used in error messages.
        min_length (int, optional): The minimum length of the array. Defaults to None.
        max_length (int, optional): The maximum length of the array. Defaults to None.
        exact_length (int, optional): The exact length of the array. Defaults to None.
        data_type (type, optional): The type each item in the array should have. Defaults to None.

    Raises:
        ValueError: If `n` is not a NumPy array or its length is not within the specified bounds, 
                    or if any item in the array is not of the required type.

    Returns:
        np.ndarray: The verified NumPy array `n`.
    """
    if isinstance(n, (list, tuple)):
        n = np.narray(n)
    elif not isinstance(n, np.ndarray):
        raise ValueError(f"{name} is not a numpy array: {type(n)}")

    if min_length is not None and len(n) < min_length:
        raise ValueError(f"{name} is not at least minimum length {min_length}: {len(n)}")

    if max_length is not None and len(n) > max_length:
        raise ValueError(f"{name} exceeds maximum length {max_length}: {len(n)}")

    if exact_length is not None and len(n) != exact_length:
        raise ValueError(f"{name} is not required length {exact_length}: {len(n)}")
    
    if data_type is not None:
        for i,item in enumerate(n):
            if not isinstance(item, data_type):
                raise ValueError(f"{name}[{i}] is not correct dtype {data_type}: {type(item)}")

    # check for np.nan's
    if (np.isnan(n).any()):
        raise ValueError('Data contains one or more NaN np.nan values.')

    # check for np.inf's
    if (np.isinf(n).any()):
        raise ValueError('Data contains one or more np.inf values.')

    return n


def verify_dict(
    d: dict,
    name: str,
    required_keys: list = [],
    required_types: list = [],
):
    """
    Verifies that the value `d` is a dictionary and optionally checks for required keys and their types.

    Args:
        d (dict): The value to be checked.
        name (str): The name of the value to be used in error messages.
        required_keys (list, optional): A list of required keys in the dictionary. Defaults to an empty list.
        required_types (list, optional): A list of types for each required key. Defaults to an empty list.

    Raises:
        ValueError: If `d` is not a dictionary, or if any required key is missing or has an incorrect type.

    Returns:
        dict: The verified dictionary `d`.
    """
    if not isinstance(d, dict):
        raise ValueError(f"{name} is not a dict: {type(d)}")

    for i,k in enumerate(required_keys):
        if not k in d.keys():
            raise ValueError(f"{name} is missing required key {k}: {d.keys()}")
        if len(required_keys) > 0:
            if not isinstance(d[k], required_types[i]):
                raise ValueError(f"{name}[{k}] is not required type {required_types[i]}: {type(k)}")
    
    return d


### TorchSig specific

def verify_transforms(
    t: Transform
) -> List[Transform | Callable]:
    """
    Verifies that the value `t` is a valid transform, which can be a single transform or a list of transforms.

    Args:
        t (Transform): The transform(s) to be checked.

    Raises:
        ValueError: If `t` is not a valid transform.

    Returns:
        List[Transform | Callable]: The verified list of transforms.
    """
    from torchsig.transforms.base_transforms import Transform
    if t is None:
        return []
    # convert all transforms to list of transforms
    if isinstance(t, Transform):
        t = [t]
    elif not isinstance(t, list):
        raise ValueError(f"transforms is not a list: {type(t)}")

    for transform in t:
        if not callable(transform):
            raise ValueError(f"non-callable or non-Transform object found in transforms; all transforms must be a callable: {transform}")
    
    return t

def verify_target_transforms(
    tt: TargetTransform
) -> List[TargetTransform | Callable]:
    """
    Verifies that the value `tt` is a valid target transform, which can be a single target transform or a list of transforms.

    Args:
        tt (TargetTransform): The target transform(s) to be checked.

    Raises:
        ValueError: If `tt` is not a valid target transform.

    Returns:
        List[TargetTransform | Callable]: The verified list of target transforms.
    """
    from torchsig.transforms.target_transforms import TargetTransform

    if tt is None:
        return []
    # convert target transforms to list
    if isinstance(tt, TargetTransform):
        tt = [tt]
    elif not isinstance(tt, list):
        raise ValueError(f"target transforms is not a list: {type(tt)}")

    for target_transform in tt:
        if not callable(target_transform):
            raise ValueError(f"non-callable or non-Transform object found in transforms; all transforms must be a callable: {target_transform}")
    
    return tt

