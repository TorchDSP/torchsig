"""Dataset Transform/Impairment base class
"""

# TorchSig
from torchsig.transforms.base_transforms import Compose, Transform
from torchsig.transforms.transforms import SignalTransform

# Built-In
from typing import List

class Impairments(Transform):
    """Applies signal and dataset transformations at specific impairment levels.

    This class applies a set of signal and dataset transforms based on a given impairment
    level. The impairment level must be between 0 and 2, where each level corresponds to
    different sets of transformations for signals and datasets.
    * Level 0: Perfect
    * Level 1: Cabled enviornment
    * Level 2: Wireless environment

    Args:
        all_levels_signal_transforms (List[SignalTransform]): A list of signal transformations for all impairment levels.
        all_levels_dataset_transforms (List[DatasetTransform]): A list of dataset transformations for all impairment levels.
        level (int): The impairment level (must be between 0 and 2).
        **kwargs: Additional keyword arguments passed to the parent class `Transform`.

    Raises:
        ValueError: If the provided impairment level is outside the valid range (0, 1, 2).

    Attributes:
        level (int): The specified impairment level.
        signal_transforms (Compose): The composed signal transformations corresponding to the given impairment level.
        dataset_transforms (Compose): The composed dataset transformations corresponding to the given impairment level.
    """
    def __init__(
        self, 
        all_levels_signal_transforms: List[SignalTransform], 
        all_levels_dataset_transforms: List[SignalTransform], 
        level: int, 
        **kwargs
    ): 
        super().__init__(**kwargs)

        self.level = level

        self.signal_transforms = Compose(transforms = all_levels_signal_transforms[self.level])
        self.signal_transforms.add_parent(self)
        self.dataset_transforms = Compose(transforms = all_levels_dataset_transforms[self.level])
        self.dataset_transforms.add_parent(self)
