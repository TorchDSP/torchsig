from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
    "StringParameter",
    "uniform_discrete_distribution",
    "uniform_continuous_distribution",
    "to_distribution",
    "normalize1d",
    "normalize2d",
    "phase_shift",
]


FloatParameter = Union[Callable[[float], float], float, Tuple[float, float], List[float]]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List[int]]
NumericParameter = Union[FloatParameter, IntParameter]
StringParameter = Union[Callable[[str], str], str, List[str]]


def uniform_discrete_distribution(
    choices: Union[List[int], List[float], List[str]],
    random_generator: Optional[torch.Generator] = None,
) -> Callable:
    """Randomization helper function for discrete distributions.
    
    Parameters
    ----------
    choices : list of int, float, str
        List of options from which to sample
    random_generator: torch.Generator, optional
        Optional random generator for tailored control of randomizations
        
    Returns
    -------
    Callable
        Function to sample from input choices.
        
    Examples
    --------
    >>> choices = [1, 2, 3]
    >>> rng = torch.Generator()
    >>> dist = uniform_discrete_distribution(choices, rng)
    >>> sample = dist()
    
    """
    random_generator = random_generator if random_generator else torch.Generator()
    weights = torch.Tensor([1 / len(choices)] * len(choices))
    return partial(
        lambda choices, random_generator: choices[
            torch.multinomial(torch.Tensor(weights), num_samples=1, generator=random_generator)
        ],
        choices,
        random_generator,
    )


def uniform_continuous_distribution(
    lower: Union[int, float],
    upper: Union[int, float],
    random_generator: Optional[torch.Generator] = None,
) -> Callable:
    """Randomization helper function for uniform continuous distributions.
    
    Parameters
    ----------
    lower : int, float
        Lower bound for random distribution
    upper : inf, float
        Upper bound for random distribution
    random_generator: torch.Generator, optional
        Optional random generator for tailored control of randomizations
        
    Returns
    -------
    Callable
        Function to sample from (lower, upper)
        
    Examples
    --------
    >>> lower, upper = 0.0, 1.0
    >>> rng = torch.Generator()
    >>> dist = uniform_continuous_distribution(lower, upper, rng)
    >>> sample = dist()
    
    """
    random_generator = random_generator if random_generator else torch.Generator()
    if lower == upper:
        return lambda: lower
    return partial(
        lambda lower, upper, random_generator: (upper - lower)
        * torch.rand(1, generator=random_generator)
        + lower,
        lower,
        upper,
        random_generator,
    )


def to_distribution(
    param: Union[Callable, NumericParameter, StringParameter],
    random_generator: Optional[torch.Generator] = None,
) -> Callable:
    """Randomization helper function for multiple types of distributions.
    
    If input is a tuple, the output will be a function that randomly samples
    from (lower, upper) bounds defined by the tuple. If input is a list, the
    output will be a function that randomly samples from the list. If the input
    is an int, float, or string, the output will be a function that returns the
    input value. If the input is a function, then it will simply output that 
    function (useful for custom distributions).
    
    Parameters
    ----------
    param : Callable, NumericParameter, StringParameter
        Specify the type of distribution to return
    random_generator: torch.Generator, optional
        Optional random generator for tailored control of randomizations
        
    Returns
    -------
    Callable
        Function to sample from input param
        
    Examples
    --------
    >>> rng = torch.Generator()
    >>> # Uniform continuous distribution in range [0.0, 1.0)
    >>> param = (0.0, 1.0)
    >>> dist = to_distribution(param, rng)
    >>> sample = dist()
    >>> # Discrete distribution from ['a', 'b', 'c']
    >>> param = ['a', 'b', 'c']
    >>> dist = to_distribution(param, rng)
    >>> sample = dist()
    >>> # Static output of 3.14
    >>> param = 3.14
    >>> dist = to_distribution(param, rng)
    >>> sample = dist()
    >>> # Custom distribution
    >>> param = lambda : 0.0 if np.random.rand() <= 0.25 else 1.0
    >>> dist = to_distribution(param, rng)
    >>> sample = dist()
    
    """
    random_generator = random_generator if random_generator else torch.Generator()
    if isinstance(param, Callable):  # type: ignore
        return param  # type: ignore

    if isinstance(param, tuple):
        return uniform_continuous_distribution(param[0], param[1], random_generator)

    if isinstance(param, int) or isinstance(param, float) or isinstance(param, str):
        return lambda: param

    if isinstance(param, list):
        return uniform_discrete_distribution(param, random_generator)

    return param


def normalize1d(input: Tensor, norm_order: Optional[int] = None) -> Tensor:
    """Scale a tensor so that a specfied norm computes to 1. 
    
    For detailed information, see :func:`torch.norm.linalg`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=float(inf), norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Parameters
    ----------
    input : torch.Tensor
        (N, T)-sized or (T,)-sized tensor to be normalized.
    norm_order: int, float, inf, -inf, 'fro', 'nuc', optional
        Type of norm with which to normalize using `torch.linalg.norm`

    Returns
    -------
    torch.Tensor
        Normalized 1D tensor.
        
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input should be a torch.Tensor. Got {type(input)}")

    input_size = len(input.size())
    if input_size != 1 and input_size != 2:
        raise ValueError(f"input.size() should be either 2 = (N, T) or 1 =(T, ). Got {input_size}")

    norm = torch.linalg.norm(input, ord=norm_order, dim=-1)
    norm = norm.unsqueeze(-1) if input_size == 2 else norm
    return torch.div(input, norm)


def normalize2d(input: Tensor, norm_order: Optional[int] = None, flatten: bool = False,) -> Tensor:
    """Scale a tensor so that a specfied norm computes to 1. 
    
    For detailed information, see :func:`torch.norm.linalg`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=float(inf), norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Parameters
    ----------
    input : torch.Tensor
        (N, C, T)-sized or (C, T)-sized tensor to be normalized.
    norm_order: int, float, inf, -inf, 'fro', 'nuc', optional
        Type of norm with which to normalize using `torch.linalg.norm`
    flatten: bool
        Boolean specifying if the input array's norm should be calculated on 
        the flattened representation of the input tensor

    Returns
    -------
    torch.Tensor
        Normalized 2D tensor
        
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input should be a torch.Tensor. Got {type(input)}")

    input_size = len(input.size())
    if input_size == 3:
        # batched input
        start_dim = 1
        unsqueeze = True
    elif input_size == 2:
        # single input
        start_dim = 0
        unsqueeze = False
    else:
        raise ValueError(
            f"len(input.size()) should be either 3 = (N, C, T) or 2 = (C, T). Got {input_size}"
        )

    if flatten:
        flat_tensor: torch.Tensor = input.flatten(start_dim=start_dim)
        norm = torch.linalg.norm(flat_tensor, ord=norm_order, dim=-1)
    else:
        norm = torch.linalg.norm(input, ord=norm_order, dim=(-2, -1))

    norm = norm.unsqueeze(-1).unsqueeze(-1) if unsqueeze else norm
    return torch.div(input, norm)


def phase_shift(input: Tensor, phase: float) -> Tensor:
    """Applies a phase rotation to input

    Parameters
    ----------
    input : torch.Tensor
        (N, T) or (T, )-sized tensor.
    phase: float
        Phase to rotate sample in [-pi, pi]

    Returns
    -------
    torch.Tensor
        Tensor that has undergone a phase rotation

    """
    return input * torch.exp(torch.tensor([1j * phase]))
