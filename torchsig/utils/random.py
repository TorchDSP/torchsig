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
        if not seed:
            seed = secrets.randbits(64)
        self.seed(seed)
        if parent:
            self.add_parent(parent)
    
    def add_parent(self, parent) -> None:
        """Add parent Seedable object and set up RNGs accordingly
        """ 
        self.parent = parent       
        self.parent.children += [self]
        self.update_from_parent()
    
    def update_from_parent(self) -> None:
        """Update numpy and torch number generators with parent seed
        """        
        self.rng_seed = self.parent.rng_seed
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

    def get_distribution(self, params, scaling:str='linear'):
        new_distribution = make_distribution(params,scaling)
        new_distribution.add_parent(self)
        return new_distribution


def make_distribution(params,scaling:str='linear'):
    if callable(params):
        # custom distribution function
        raise NotImplementedError
    elif isinstance(params, list):
        # draw samples from uniform distribution from list values
        return ChoiceDistribution(params)
    elif isinstance(params, tuple) and scaling == 'linear':
        # draw samples from uniform distribution from [params[0], params[1]]
        return UniformRangeDistribution(params)
    elif isinstance(params, tuple) and scaling == 'log10':
        # draw samples from log10-weighted uniform distribution from [params[0], params[1]]
        return Log10UniformRangeDistribution(params)
    elif (isinstance(params, int) or isinstance(params, float)) and scaling == 'linear':
        # draw samples from evenly spaced values within [0, params)
        return UniformDistribution(params)
    else:
        raise ValueError(f'Undefined conditions in make_distribution(). params = {params}, scaling = {scaling}')


class Distribution(Seedable):
    """A class for representing random distributions; created by calling get_distribution(params) on a Seedable object
    distributions are callable, such that some_seedable.get_distribution(params)() should return a random number from the distribution
    """
    def __init__(self, params, **kwargs):
        Seedable.__init__(self, **kwargs)
        self.params = params
            
    def __repr__(self) -> str:
         return (
             f"{self.__class__.__name__}(params={self.params}, seed={self.rng_seed}, parent={self.parent})"
         )
        
    def get_value(self):
        raise NotImplementedError("The Distribution class does not specify a distribution by itself. This must be specified by a subclass.")
        
    def __call__(self, size=None):
        if size == None:
            return self.get_value()
        return np.array([self.get_value() for i in range(size)])

class ChoiceDistribution(Distribution):
    """A class for handling random choices from lists"""
    def __init__(self, params, **kwargs):
        Distribution.__init__(self, params, **kwargs)
    def get_value(self):
        return self.random_generator.choice(self.params)

class UniformRangeDistribution(Distribution):
    """A class for handling random uniform ranges"""
    def __init__(self, params, **kwargs):
        Distribution.__init__(self, params, **kwargs)
    def get_value(self):
        return self.random_generator.uniform(low=self.params[0], high=self.params[1])

class Log10UniformRangeDistribution(Distribution):
    """A class for handling log10-weighted random uniform ranges"""
    def __init__(self, params, **kwargs):
        Distribution.__init__(self, params, **kwargs)
    def get_value(self):
        low_log10 = np.log10(self.params[0])
        high_log10 = np.log10(self.params[1])
        random_exponent = self.random_generator.uniform(low=low_log10, high=high_log10)
        linear_value = 10**random_exponent
        return linear_value

class UniformDistribution(Distribution):
    """A class for handling uniform random variables"""
    def __init__(self, params, **kwargs):
        Distribution.__init__(self, params, **kwargs)
    def get_value(self):
        return self.random_generator.uniform(high=self.params)


