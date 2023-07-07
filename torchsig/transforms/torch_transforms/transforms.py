# type: ignore

from typing import Any, Callable, List, Optional, Tuple, Union

from torch import Tensor, nn

from torchsig.transforms.torch_transforms import functional as F
from torchsig.transforms import Transform


class Compose(nn.Sequential):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to compose.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Compose([ST.AddNoise(noise_power_db=10), ST.InterleaveComplex()])

    """

    def __init__(self, transforms: List[Transform], **kwargs):
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        return "\n".join([str(t) for t in self.transforms])


class Normalize(nn.Module):
    """Normalize a IQ vector with mean and standard deviation.

    Args:
        norm :obj:`string`:
            Type of norm with which to normalize

        flatten :obj:`bool`:
            Specifies if the norm should be calculated on the flattened
            representation of the input tensor

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """

    def __init__(
        self,
        norm: int = 2,
        flatten: bool = False,
    ):
        super(Normalize, self).__init__()
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

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor data to be normalized.

        Returns:
            Tensor: Normalized Tensor data.
        """
        return F.normalize(tensor, self.norm, self.flatten)
