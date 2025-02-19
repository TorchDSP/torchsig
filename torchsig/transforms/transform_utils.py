"""Transform utilities
"""


__all__ = [
    "get_distribution",
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
]


from functools import partial
import numpy as np

# Built-In
from typing import Callable, Tuple, List

FloatParameter = float | Tuple[float, float] | List[float] | Callable[[int], float]
IntParameter = int | Tuple[int, int] | List[int] | Callable[[int], int]
StrParameter = str | List[str]
NumericParameter = FloatParameter | IntParameter




def get_distribution(
        params: NumericParameter | StrParameter, 
        rng: np.random.Generator = np.random.default_rng()
) -> Callable:
    """Generates an appropriate random distribution function based on provided parameters.

    Args:
        params (NumericParameter | StrParameter): The parameters defining the distribution. 
            It can be a numeric value, a tuple of two values defining a range, a list of values,
            or a callable that generates values.
        rng (np.random.Generator, optional): A random number generator instance. 
            Defaults to np.random.default_rng().

    Returns:
        Callable: A function that generates random values according to the distribution defined by `params`.

    """
    distribution = params

    if isinstance(params, Callable):
        # custom distribution function
        distribution = params

    if isinstance(params, list):
        # draw samples from uniform distribution from list values
        distribution = partial(rng.choice, params)

    if isinstance(params, tuple):
        # draw samples from uniform distribution from [params[0], params[1])]
        distribution = partial(rng.uniform, low=params[0], high=params[1])

    if isinstance(params, (int, float)):
        # draw samples from evenly spaced values within [0, params)
        # return partial(rng.choice, params)
        distribution = partial(rng.uniform, high = params)

    return distribution

    
