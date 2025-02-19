"""Base Transforms
"""

from __future__ import annotations

__all__ = [
    "Transform",
    "Compose",
    "Lambda",    
    "Normalize",
    "RandomApply",
    "RandAugment",
]

# TorchSig
import torchsig.transforms.functional as F
from torchsig.signals.signal_types import Signal, DatasetSignal
from torchsig.utils.random import Seedable
from torchsig.utils.printing import generate_repr_str

# Third Party
from abc import ABC
from typing import Callable, List, Literal, Optional


class Transform(ABC, Seedable):
    """Transform abstract class.
    """
    def __init__(
        self, 
        **kwargs
    ):      
        """Transform initialization as Seedable.
        """    
        Seedable.__init__(self, **kwargs)

    def update(self, signal: Signal | DatasetSignal) -> None:   
        """Update bookeeping for signals

        Args:
            signal (Signal | DatasetSignal): signal to update metadata.

        Raises:
            NotImplementedError: Inherited classes must override this method.
        """         
        raise NotImplementedError

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        """Performs transforms

        Args:
            signal (Any): Signal to be transformed.

        Raises:
            NotImplementedError: Inherited classes must override this method.

        Returns:
            Any: Transformed Signal.
            
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:   
        """Transform string representation.
        Should be able to recreate class from this string.

        Returns:
            str: Transform representation.
        """
        return generate_repr_str(self)


class Compose(Transform):
    """Composes several transforms together sequentially, in order.

    Attributes:
        transforms (List[Transform]): list of Transform objects.

    """
    def __init__(self, transforms: List[Transform], **kwargs):
        self.transforms = transforms
        super().__init__(**kwargs)

        for t in self.transforms:
            if isinstance(t, Seedable):
                t.add_parent(self)

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        for t in self.transforms:
            signal = t(signal)
        return signal


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.
       Warning: does not automatically update metadata

    Attributes:
        func (Callable): Lambda/function to be used for transform.

    Example:
        >>> from torchsig.transforms.base_transforms import Lambda
        >>> transform = Lambda(lambda x: x**2)  # A transform that squares all inputs.

    """
    def __init__(
            self, 
            func: Callable, 
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.func = func

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        signal.data = self.func(signal.data)
        return signal


class Normalize(Transform):
    """Normalize an IQ data vector.

    Attributes:
        norm (str): Order of the norm (refer to numpy.linalg.norm).

        flatten (bool): Specifies if the norm should be calculated on the flattened
            representation of the input tensor.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """
    def __init__(
        self,
        norm: Optional[int | float | Literal["fro", "nuc"]] = 2,
        flatten: bool = False,
        seed:int = None,
        **kwargs
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.norm = norm
        self.flatten = flatten
    
    # def __repr__(self) -> str:
    #     r = super().__repr__()
    #     r = r.replace("inf", "np.inf")
    #     return r

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        if self.flatten:
            signal.data = signal.data.reshape(signal.data.size)
        
        signal.data = F.normalize(
            signal.data,
            norm_order = self.norm,
            flatten = self.flatten
        )
        return signal


class RandomApply(Transform):
    """Randomly applies transform with probability p.

    Attributes:
        transform (Transform): Transform to randomly apply.
        probability (float): Probability to apply transform in range [0., 1.].

    """
    def __init__(
        self,
        transform,
        probability: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform = transform
        self.probability = probability

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        if self.random_generator.random() < self.probability:
            return self.transform(signal)
        return signal


class RandAugment(Transform):
    """RandAugment transform loosely based on:
    `"RandAugment: Practical automated data augmentation with a reduced search space"
      <https://arxiv.org/pdf/1909.13719.pdf>`.
    
    Attributes:
        transforms (List[Transform]): list of Transforms to choose from.
        choose (int, optional): Number of Transforms to randomly choose. Defaults to 2.
        replace (bool, optional): Allow replacement in random choose. Defaults to False.

    """
    def __init__(
        self, 
        transforms: List[Transform], 
        choose: int = 2, 
        replace: bool = False,
        seed:int = None,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.transforms = transforms
        self.choose = choose
        self.replace = replace

    def __call__(self, signal: Signal | DatasetSignal) -> Signal | DatasetSignal:
        chosen_transforms_idx = self.random_generator.choice(
            len(self.transforms),
            size=self.choose,
            replace=self.replace
        )
        for t in [self.transforms[idx] for idx in chosen_transforms_idx]:
            signal = t(signal)
        
        return signal

