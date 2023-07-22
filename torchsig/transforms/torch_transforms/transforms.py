import abc
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from torchsig.transforms.torch_transforms import functional as F
from torchsig.transforms.torch_transforms.functional import (
    FloatParameter,
    IntParameter,
    NumericParameter,
)

from .types import Signal


class Transform(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base nn.Module class representing a Transform that operates on
    a `Signal` object containing data and metadata.

    """

    def __init__(self, *args, **kwargs) -> None:
        super(Transform, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, input: Signal) -> Signal:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Augmentation(Transform, metaclass=abc.ABCMeta):
    """Abstract base Transform class representing an augmentation that operates 
    on a `Signal` object containing data and metadata using a random context.

    Parameters
    ----------
    random_generator : torch.Generator, optional
        Optional torch generator for more control over random contexts and 
        avoiding the global random state.

    """

    def __init__(
        self, random_generator: Optional[torch.Generator] = None, *args, **kwargs,
    ) -> None:
        super(Augmentation, self).__init__(*args, **kwargs)
        self.random_generator = random_generator if random_generator else torch.Generator()

    @abc.abstractmethod
    def forward(self, input: Signal) -> Signal:
        random_context = torch.rand(1, generator=self.random_generator)
        raise NotImplementedError


class Compose(nn.Sequential):
    """Composes several transforms together.

    Parameters
    ----------
    transforms : list of `Transform` objects
        List of transforms to compose.

    Examples
    --------
    >>> import torchsig.transforms as ST
    >>> transform = ST.Compose([ST.PhaseShift(), ST.Normalize1D()])

    """

    def __init__(self, transforms: List[Union[Transform, Augmentation]]) -> None:
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, input: Signal) -> Signal:
        for t in self.transforms:
            input = t(input)
        return input

    def __repr__(self) -> str:
        return "\n".join([str(t) for t in self.transforms])


class Identity(Transform):
    """Just passes the input -- surprisingly useful in pipelines

    Examples
    --------
    >>> import torchsig.transforms as ST
    >>> transform = ST.Identity()

    """

    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Signal) -> Signal:
        return input


class Normalize1D(Transform):
    """Normalize a 1D vector with mean and standard deviation

    Parameters
    ----------
    norm : int, float, inf, -inf, 'fro', 'nuc', optional
        Type of norm with which to normalize

    Examples
    --------
    >>> import torchsig.transforms as ST
    >>> transform = ST.Normalize1D(norm=2) # normalize by l2 norm
    >>> transform = ST.Normalize1D(norm=1) # normalize by l1 norm

    """

    def __init__(self, norm: Optional[int] = 2) -> None:
        super(Normalize1D, self).__init__()
        self.norm = norm
        self.string = self.__class__.__name__ + "(" + "norm={}".format(norm) + ")"

    def __repr__(self) -> str:
        return self.string

    def forward(self, input: Signal) -> Signal:
        output: Signal = Signal(
            data=F.normalize1d(input.data, self.norm), metadata=input.metadata,
        )
        return output


class Normalize2D(Transform):
    """Normalize a 2D vector with mean and standard deviation

    Parameters
    ----------
    norm : int, float, inf, -inf, 'fro', 'nuc', optional
        Type of norm with which to normalize
    flatten: bool
        Specifies if the norm should be calculated on the flattened
        representation of the input tensor

    Examples
    --------
    >>> import torchsig.transforms as ST
    >>> transform = ST.Normalize2D(norm=2) # normalize by l2 norm
    >>> transform = ST.Normalize2D(norm=1) # normalize by l1 norm
    >>> transform = ST.Normalize2D(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """

    def __init__(self, norm: Optional[int] = 2, flatten: bool = False) -> None:
        super(Normalize2D, self).__init__()
        self.norm = norm
        self.flatten = flatten
        self.string = (
            self.__class__.__name__
            + "("
            + "norm={}, ".format(norm)
            + "flatten={}".format(flatten)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def forward(self, input: Signal) -> Signal:
        output: Signal = Signal(
            data=F.normalize2d(input.data, self.norm, self.flatten), metadata=input.metadata,
        )
        return output


class PhaseShift(Augmentation):
    """Applies a random phase offset to tensor

    Parameters
    ----------
    phase_offset : FloatParameter
        Phase offset distribution normalized to pi

    Examples
    --------
    >>> import torchsig.transforms as ST
    >>> # Phase Offset in range [-pi, pi]
    >>> transform = ST.PhaseShift(phase_offset=(-1, 1))
    >>> # Phase Offset is fixed at -pi/2
    >>> transform = ST.PhaseShift(phase_offset=(-.5, -.5)
    
    """

    def __init__(
        self,
        phase_offset: FloatParameter = (-1.0, 1.0),
        random_generator: Optional[torch.Generator] = None,
    ) -> None:
        super(PhaseShift, self).__init__(random_generator=random_generator)
        self.phase_offset = F.to_distribution(phase_offset, self.random_generator)
        self.string = self.__class__.__name__ + "(" + "phase_offset{}".format(phase_offset) + ")"

    def __repr__(self) -> str:
        return self.string

    def forward(self, input: Signal) -> Signal:
        phase = self.phase_offset()
        input.data = F.phase_shift(input.data, phase * torch.pi)
        return input
