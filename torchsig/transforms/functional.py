from typing import Callable, Union, Tuple, List
from functools import partial
import numpy as np

FloatParameter = Union[Callable[[int], float], float, Tuple[float, float], List]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


def uniform_discrete_distribution(choices: List, random_generator: np.random.RandomState = np.random.RandomState()):
    return partial(random_generator.choice, choices)


def uniform_continuous_distribution(
        lower: Union[int, float],
        upper: Union[int, float],
        random_generator: np.random.RandomState = np.random.RandomState()
):
    return partial(random_generator.uniform, lower, upper)


def to_distribution(param, random_generator: np.random.RandomState = np.random.RandomState()):
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
