# type: ignore

from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor


__all__ = [
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
    "uniform_discrete_distribution",
    "uniform_continuous_distribution",
    "to_distribution",
    "normalize",
]


FloatParameter = Union[Callable[[int], float], float, Tuple[float, float], List]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


def uniform_discrete_distribution(
    choices: List,
    random_generator: Optional[torch.Generator] = None,
):
    random_generator = random_generator if random_generator else torch.Generator()
    return partial(random_generator.choice, choices)


def uniform_continuous_distribution(
    lower: Union[int, float],
    upper: Union[int, float],
    random_generator: Optional[torch.Generator] = None,
):
    random_generator = random_generator if random_generator else torch.Generator()
    return partial(random_generator.uniform, lower, upper)


def to_distribution(
    param,
    random_generator: Optional[torch.Generator] = None,
):
    random_generator = random_generator if random_generator else torch.Generator()
    if isinstance(param, Callable):
        return param

    # if isinstance(param, list):
    #     if isinstance(param[0], tuple):
    #         tuple_from_list = param[random_generator.randint(len(param))]
    #         return uniform_continuous_distribution(
    #             tuple_from_list[0],
    #             tuple_from_list[1],
    #             random_generator,
    #         )
    #     return uniform_discrete_distribution(param, random_generator)

    if isinstance(param, tuple):
        return uniform_continuous_distribution(param[0], param[1], random_generator)

    if isinstance(param, int) or isinstance(param, float):
        return uniform_discrete_distribution([param], random_generator)

    return param


def normalize(
    tensor: Tensor,
    norm_order: Optional[int] = None,
    flatten: bool = False,
) -> Tensor:
    """Scale a tensor so that a specfied norm computes to 1. For detailed information, see :func:`torch.linalg.norm.`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=np.inf, norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Args:
        tensor (:class:`torch.Tensor`)):
            (batch_size, vector_length, ...)-sized tensor to be normalized.

        norm_order (:class:`int`)):
            norm order to be passed to torch.linalg.norm

        flatten (:class:`bool`)):
            boolean specifying if the input array's norm should be calculated on the flattened representation of the input tensor

    Returns:
        Tensor:
            Normalized complex array.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"input should be a torch.Tensor. Got {type(tensor)}")

    if flatten:
        flat_tensor: torch.Tensor = tensor.view(tensor.size())
        norm = torch.norm(flat_tensor, norm_order)
    else:
        norm = torch.norm(tensor, norm_order)

    return torch.multiply(tensor, 1.0 / norm)
