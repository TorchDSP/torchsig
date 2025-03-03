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




########## TODO this file should probably be removed