from typing import Callable, List, Protocol, Sequence, Tuple, Union
from functools import partial
import numpy as np

FloatParameter = Union[Callable[[int], float], float, Tuple[float, float], List]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


class RandomStatePartial(Protocol):
    """Type definition for the partially applied random distribution function
    returned by the functions in this module.

    These partials can be either called with zero arguments, in which case a
    single value is returned, or by passing in a size parameter, in which case
    a np.ndarray of the specified shape is returned.

    See: https://peps.python.org/pep-0544/
    See: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols
    """
    def __call__(self, size: Union[int, Sequence[int]] = ...) -> np.typing.ArrayLike:
        ...


def uniform_discrete_distribution(
        choices: List,
        random_generator: np.random.RandomState = np.random.RandomState()
) -> RandomStatePartial:
    return partial(random_generator.choice, choices)


def uniform_continuous_distribution(
        lower: Union[int, float],
        upper: Union[int, float],
        random_generator: np.random.RandomState = np.random.RandomState()
) -> RandomStatePartial:
    return partial(random_generator.uniform, lower, upper)


def to_distribution(
        param: NumericParameter,
        random_generator: np.random.RandomState = np.random.RandomState()
) -> RandomStatePartial:
    if isinstance(param, Callable):
        return param

    if isinstance(param, list):
        if isinstance(param[0], tuple):
            tuple_from_list = param[random_generator.randint(len(param))]
            return uniform_continuous_distribution(
                tuple_from_list[0], 
                tuple_from_list[1], 
                random_generator,
            )
        return uniform_discrete_distribution(param, random_generator)

    if isinstance(param, tuple):
        return uniform_continuous_distribution(param[0], param[1], random_generator)

    if isinstance(param, int) or isinstance(param, float):
        return uniform_discrete_distribution([param], random_generator)

    return param
