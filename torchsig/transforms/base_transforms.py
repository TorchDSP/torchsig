"""Base and Utility Transforms"""

from __future__ import annotations

__all__ = [
    "Compose",
    "Lambda",
    "Normalize",
    "RandAugment",
    "RandomApply",
    "Transform",
]

from abc import ABC
from typing import TYPE_CHECKING, Literal

import torchsig.transforms.functional as F
from torchsig.utils.printing import generate_repr_str
from torchsig.utils.random import Seedable

if TYPE_CHECKING:
    from torchsig.signals.signal_types import Signal


class Transform(ABC, Seedable):
    """Transform abstract class.

    This is the base class for all transforms in TorchSig.
    All transforms should inherit from this class and implement the required methods.
    """

    def __init__(self, required_metadata: list[str] = [], **kwargs):
        """Transform initialization as Seedable.

        Args:
            required_metadata: List of metadata fields required for the transform to be applied.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # what metadata fields are requried for target transform to be applied
        self.required_metadata = required_metadata

        Seedable.__init__(self, **kwargs)

    def __validate__(self, signal):
        """Validates signal or metadata before applying transform.

        Args:
            signal: Signal to be validated.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            Validated signal.
        """
        raise NotImplementedError

    def __update__(self, signal):
        """Updates bookeeping for signals.

        Args:
            signal: Signal to update metadata.

        Raises:
            ValueError: If signal is None.
        """
        if signal is None:
            raise ValueError(
                f"Invalid signal object to update in transform {self.__class__.__name__}. Signal is None: {signal}"
            )

    def __apply__(self, signal):
        """Performs transform.

        Args:
            signal: Signal to be transformed.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            Transformed signal.
        """
        raise NotImplementedError

    def __call__(self, signal):
        """Validate signal, performs transform, update bookeeping.

        Args:
            signal: Signal to be transformed.

        Raises:
            NotImplementedError: Inherited classes must override this method.

        Returns:
            Transformed Signal.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """String representation of the transform.

        Returns:
            String representation of the transform.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Transform string representation.

        Should be able to recreate class from this string.

        Returns:
            Transform representation.
        """
        return generate_repr_str(self)


class Compose(Transform):
    """Composes several transforms together sequentially, in order.

    This transform applies a sequence of transforms to the input signal.

    Attributes:
        transforms: List of Transform objects to be applied sequentially.
    """

    def __init__(self, transforms: list[Transform], **kwargs):
        """Initialize the Compose transform.

        Args:
            transforms: List of Transform objects to be applied sequentially.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.transforms = transforms
        super().__init__(**kwargs)
        for t in self.transforms:
            if isinstance(t, Seedable):
                t.add_parent(self)

    def __call__(self, signal: Signal) -> Signal:
        """Apply all transforms in sequence.

        Args:
            signal: Signal to be transformed.

        Returns:
            Transformed signal after applying all transforms.
        """
        for t in self.transforms:
            signal = t(signal)
        return signal


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.

    Warning: Does not automatically update metadata.

    Attributes:
        func: Lambda/function to be used for transform.

    Example:
        >>> from torchsig.transforms.base_transforms import Lambda
        >>> transform = Lambda(lambda x: x**2)  # A transform that squares all inputs.
    """

    def __init__(self, func: callable, **kwargs) -> None:
        """Initialize the Lambda transform.

        Args:
            func: Lambda/function to be used for transform.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.func = func

    def __call__(self, signal: Signal) -> Signal:
        """Apply the lambda function to the signal data.

        Args:
            signal: Signal to be transformed.

        Returns:
            Transformed signal with modified data.
        """
        signal.data = self.func(signal.data)
        return signal


class Normalize(Transform):
    """Normalize an IQ data vector.

    This transform normalizes the IQ data according to the specified norm.

    Attributes:
        norm: Order of the norm (refer to numpy.linalg.norm).
        flatten: Specifies if the norm should be calculated on the flattened representation of the input tensor.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation
    """

    def __init__(
        self,
        norm: float | Literal["fro", "nuc"] | None = 2,
        flatten: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Normalize transform.

        Args:
            norm: Order of the norm (refer to numpy.linalg.norm). Defaults to 2.
            flatten: Specifies if the norm should be calculated on the flattened representation of the input tensor. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.norm = norm
        self.flatten = flatten

    def __call__(self, signal: Signal) -> Signal:
        """Normalize the signal data.

        Args:
            signal: Signal to be transformed.

        Returns:
            Transformed signal with normalized data.
        """
        if self.flatten:
            signal.data = signal.data.reshape(signal.data.size)

        signal.data = F.normalize(
            signal.data, norm_order=self.norm, flatten=self.flatten
        )
        return signal


class RandomApply(Transform):
    """Randomly applies transform with probability p.

    This transform applies the specified transform with a given probability.

    Attributes:
        transform: Transform to randomly apply.
        probability: Probability to apply transform in range [0., 1.].
    """

    def __init__(self, transform, probability: float, **kwargs):
        """Initialize the RandomApply transform.

        Args:
            transform: Transform to randomly apply.
            probability: Probability to apply transform in range [0., 1.].
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.transform = transform
        self.probability = probability
        if isinstance(self.transform, Seedable):
            self.transform.add_parent(self)

    def __call__(self, signal: Signal) -> Signal:
        """Apply the transform with the specified probability.

        Args:
            signal: Signal to be transformed.

        Returns:
            Transformed signal if the random number is less than probability, otherwise the original signal.
        """
        if self.random_generator.random() < self.probability:
            return self.transform(signal)
        return signal


class RandAugment(Transform):
    """RandAugment transform loosely based on:
    `"RandAugment: Practical automated data augmentation with a reduced search space"
      <https://arxiv.org/pdf/1909.13719.pdf>`_.

    This transform randomly selects and applies a subset of transforms from a list.

    Attributes:
        transforms: List of Transforms to choose from.
        choose: Number of Transforms to randomly choose. Defaults to 2.
        replace: Allow replacement in random choose. Defaults to False.
    """

    def __init__(
        self,
        transforms: list[Transform],
        choose: int = 2,
        replace: bool = False,
        **kwargs,
    ):
        """Initialize the RandAugment transform.

        Args:
            transforms: List of Transforms to choose from.
            choose: Number of Transforms to randomly choose. Defaults to 2.
            replace: Allow replacement in random choose. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.transforms = transforms
        for transform in self.transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)
        self.choose = choose
        self.replace = replace

    def __call__(self, signal: Signal) -> Signal:
        """Apply a random subset of transforms to the signal.

        Args:
            signal: Signal to be transformed.

        Returns:
            Transformed signal after applying the randomly chosen transforms.
        """
        chosen_transforms_idx = self.random_generator.choice(
            len(self.transforms), size=self.choose, replace=self.replace
        )
        for t in [self.transforms[idx] for idx in chosen_transforms_idx]:
            signal = t(signal)

        return signal
