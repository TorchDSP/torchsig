import warnings
import numpy as np
from copy import deepcopy
from typing import Any, List, Callable, Optional, Union

from torchsig.utils.types import SignalData


class Transform:
    """An abstract class representing a Transform that can either work on
    targets or data
    
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            warnings.warn("Seeding transforms is deprecated and does nothing", DeprecationWarning)

        self.random_generator = np.random
        
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to compose.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Compose([ST.AddNoise(10), ST.InterleaveComplex()])

    """
    def __init__(self, transforms: List[Transform], **kwargs):
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        return "\n".join([str(t) for t in self.transforms])


class NoTransform(Transform):
    """Just passes the data -- surprisingly useful in pipelines

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.NoTransform()

    """
    def __init__(self, **kwargs):
        super(NoTransform, self).__init__(**kwargs)

    def __call__(self, data: Any) -> Any:
        return data


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Lambda(lambda x: x**2)  # A transform that squares all inputs.

    """
    def __init__(self, func: Callable, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.func = func

    def __call__(self, data: Any) -> Any:
        return self.func(data)


class FixedRandom(Transform):
    """ Restricts a randomized transform to apply only a fixed set of seeds. 
    For example, this could be used to add noise randomly from among 1000 
    possible sets of noise or add fading from 1000 possible channels.

    Args:
        transform (:obj:`Callable`):
            transform to be called

        num_seeds (:obj:`int`):
            number of possible random seeds to use

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.FixedRandom(ST.AddNoise(10), num_seeds=10)

    """
    def __init__(self, transform: Transform, num_seeds: int, **kwargs):
        super(FixedRandom, self).__init__(**kwargs)
        self.transform = transform
        self.num_seeds = num_seeds

    def __call__(self, data: Any) -> Any:
        seed = self.random_generator.choice(self.num_seeds)
        orig_state = np.random.get_state()  # we do not want to somehow fix other random number generation processes.
        np.random.seed(seed)
        data = self.transform(data)
        np.random.set_state(orig_state)  # return numpy back to its previous state
        return data


class RandomApply(Transform):
    """ Randomly applies a set of transforms with probability p

    Args:
        transform (``Transform`` objects):
            transform to randomly apply

        probability (:obj:`float`):
            In [0, 1.0], the probability with which to apply a transform

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.RandomApply(ST.AddNoise(10), probability=.5)  # Add 10dB noise with probability .5

    """
    def __init__(self, transform: Transform, probability: float, **kwargs):
        super(RandomApply, self).__init__(**kwargs)
        self.transform = transform
        self.probability = probability

    def __call__(self, data: Any) -> Any:
        return self.transform(data) if self.random_generator.rand() < self.probability else data


class SignalTransform(Transform):
    """ An abstract base class which explicitly only operates on Signal data
    
    Args:
        time_dim (:obj:`int`): 
            Dimension along which to index time for a signal
        
    """
    def __init__(self, time_dim: int = 0, **kwargs):
        super(SignalTransform, self).__init__(**kwargs)
        self.time_dim = time_dim

    def __call__(self, data: SignalData) -> SignalData:
        raise NotImplementedError


class Concatenate(SignalTransform):
    """Concatenates Transforms into a Tuple

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to concatenate.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = Concatenate([ST.AddNoise(10), ST.DiscreteFourierTransform()])

    """
    def __init__(self, transforms: List[SignalTransform], **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = np.concatenate(
                [transform(deepcopy(data.iq_data)) for transform in self.transforms],
                axis=self.time_dim
            )
        else:
            data = np.concatenate(
                [transform(deepcopy(data)) for transform in self.transforms],
                axis=self.time_dim
            )
        return data

    def __repr__(self):
        return "\t".join([str(t) for t in self.transforms])


class TargetConcatenate(SignalTransform):
    """Concatenates Target Transforms into a Tuple

    Args:
        transforms (list of ``Transform`` objects):
            List of transforms to concatenate

    """
    def __init__(self, transforms: List[Transform], **kwargs):
        super(TargetConcatenate, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, target: Any) -> Any:
        return tuple([transform(target) for transform in self.transforms])


class RandAugment(SignalTransform):
    """RandAugment transform loosely based on: 
    `"RandAugment: Practical automated data augmentation with a reduced search space" <https://arxiv.org/pdf/1909.13719.pdf>`_.

    Args:
        transforms (list of `Transform` objects):
            List of transforms to choose from
            
        num_transforms (:obj: `int`):
            Number of transforms to randomly select
        
    """
    def __init__(self, transforms: List[SignalTransform], num_transforms: int = 2, **kwargs):
        super(RandAugment, self).__init__(**kwargs)
        self.transforms = transforms
        self.num_transforms = num_transforms

    def __call__(self, data: Any) -> Any:
        transforms = self.random_generator.choice(self.transforms, size=self.num_transforms)
        for t in transforms:
            data = t(data)
        return data
