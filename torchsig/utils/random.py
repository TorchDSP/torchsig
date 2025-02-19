"""Utility to handle random number generators.
"""

# Third Party
from torch import Generator
import numpy as np

# Built-In
import secrets

class Seedable():
    """A class/interface representing objects capable of accessing random numbers and being seeded.
    Stores an inernal random number generator object.
    Can be seeded with the Seedable.seed(seed_value : long) function.
    Two Seedable objects with the same seed will always generate/access the same random values in the same order.
    Containing or composing Seedable objects are generally responsible for seeding contained or composed Seedable objects.
    
    """
    def __init__(self, seed: int = None, parent = None):
        """Initializes seedable object with self.seed = seed;
        if a parent Seedable object is passed in, they will share random number generators, and the seed argument will not be used

        Args:
            seed (int, optional): Seed for use for number genrators. Defaults to None.
            parent (Seedable, optional): Parent Seedable responsible for seeding this object. Defaults to None.
        """
        
        self.children = []
        self.parent = None
        self.torch_rng = None
        self.random_generator = None
        self.np_rng = None
        
        if not seed:
            seed = secrets.randbits(64)
        if parent:
            self.add_parent(parent)
        else:
            self.seed(seed)
    
    def add_parent(self, parent) -> None:
        """Add parent Seedable object and set up RNGs accordingly
        """ 
        self.parent = parent       
        self.parent.children += [self]
        self.update_from_parent()
    
    def update_from_parent(self) -> None:
        """Update numpy and torch number generators with parent seed
        """        
        self.torch_rng = self.parent.torch_rng
        self.np_rng = self.parent.np_rng
        self.random_generator = self.np_rng
        for child in self.children:
            child.update_from_parent()
        
    def seed(self, seed: int) -> None:
        """Seed number generators with given seed.

        Args:
            seed (int): Seed to use.
        """        
        self.rng_seed = seed
        self.setup_rngs()
    
    def get_second_seed(self, seed: int) -> int:
        """Gets second seed, 
        usually used to seed both torch and numpy generators with slightly different seeds

        Args:
            seed (int): Seed to use.

        Returns:
            int: New seed.
        """        
        return seed + 13 # TODO do this right
    
    def setup_rngs(self) -> None:
        """Initialize torch and numpy number generators, and update its children.
        """        
        self.np_rng = np.random.default_rng(seed=self.rng_seed)
        self.random_generator = self.np_rng
        self.torch_rng = Generator()
        self.torch_rng.manual_seed(self.get_second_seed(self.rng_seed))
        for child in self.children:
            child.update_from_parent()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seed={self.rng_seed}, parent={self.parent})"
        )
