"""Utility to handle random number generators."""

import secrets
from typing import Any, Optional

import numpy as np
from torch import Generator


class Seedable:
    """A class/interface representing objects capable of accessing random numbers and being seeded.

    Stores an internal random number generator object. Can be seeded with the
    Seedable.seed(seed_value: int) function. Two Seedable objects with the same
    seed will always generate/access the same random values in the same order.
    Containing or composing Seedable objects are generally responsible for seeding
    contained or composed Seedable objects.
    """

    def __init__(self, seed: int | None = None, parent: Optional["Seedable"] = None, **kwargs):
        """Initializes seedable object with self.seed = seed.

        If a parent Seedable object is passed in, they will share random number
        generators, and the seed argument will not be used.

        Args:
            seed: Seed for use for number generators. Defaults to None.
            parent: Parent Seedable responsible for seeding this object.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.children = []
        self.parent = None
        self.rng_seed = None
        self.np_rng = None
        self.torch_rng = None
        self.random_generator = None
        self.kwargs = kwargs

        if not seed:
            # choose random seed
            seed = secrets.randbits(64)

        # seed itself
        self.seed(seed)

        if parent:
            # has parent Seedable objects
            # add parents
            self.add_parent(parent)

    def add_parent(self, parent: "Seedable") -> None:
        """Add parent Seedable object and set up RNGs accordingly.

        Args:
            parent: Parent Seedable object to add.
        """
        self.parent = parent
        self.parent.children += [self]
        self.update_from_parent()

    def update_from_parent(self) -> None:
        """Update numpy and torch number generators with parent seed."""
        self.rng_seed = self.parent.rng_seed
        self.torch_rng = self.parent.torch_rng
        self.np_rng = self.parent.np_rng
        self.random_generator = self.np_rng
        for child in self.children:
            child.update_from_parent()

    def seed(self, seed: int) -> None:
        """Seed number generators with given seed.

        Args:
            seed: Seed to use.
        """
        self.rng_seed = seed
        self.setup_rngs()

    def get_second_seed(self, seed: int) -> int:
        """Gets second seed, usually used to seed both torch and numpy generators with slightly different seeds.

        Args:
            seed: Seed to use.

        Returns:
            New seed.
        """
        return seed + 13

    def setup_rngs(self) -> None:
        """Initialize torch and numpy number generators, and update its children."""
        self.np_rng = np.random.default_rng(seed=self.rng_seed)
        self.random_generator = self.np_rng
        self.torch_rng = Generator()
        self.torch_rng.manual_seed(self.get_second_seed(self.rng_seed))
        for child in self.children:
            child.update_from_parent()

    def __repr__(self) -> str:
        """Printable representation with seed and parent.

        Returns:
            String representation of the object.
        """
        return f"{self.__class__.__name__}(seed={self.rng_seed}, parent={self.parent})"

    def get_distribution(
        self, params: list | tuple | float, scaling: str = "linear"
    ) -> "Distribution":
        """Create distribution function with proper seeding.

        Args:
            params: Parameters for distribution.
            scaling: Scaling param for distribution. Defaults to 'linear'.

        Returns:
            Distribution: Distribution function, seeded.
        """
        new_distribution = make_distribution(params, scaling)
        new_distribution.add_parent(self)
        return new_distribution


def make_distribution(
    params: list | tuple | float, scaling: str = "linear"
) -> "Distribution":
    """Creates distribution given params.

    Args:
        params: Params for distribution.
        scaling: Scaling param for distribution. Defaults to 'linear'.

    Raises:
        NotImplementedError: params is unimplamented type.
        ValueError: undefined distribution.

    Returns:
        Distribution: Distribution function from params.
    """
    if callable(params):
        # custom distribution function
        raise NotImplementedError
    if isinstance(params, list):
        # draw samples from uniform distribution from list values
        return ChoiceDistribution(params)
    if isinstance(params, tuple) and scaling == "linear":
        # draw samples from uniform distribution from [params[0], params[1]]
        return UniformRangeDistribution(params)
    if isinstance(params, tuple) and scaling == "log10":
        # draw samples from log10-weighted uniform distribution from [params[0], params[1]]
        return Log10UniformRangeDistribution(params)
    if isinstance(params, (int, float)) and scaling == "linear":
        # draw samples from evenly spaced values within [0, params)
        return UniformDistribution(params)

    # undefined distribution
    raise ValueError(
        f"Undefined conditions in make_distribution(). params = {params}, scaling = {scaling}"
    )


class Distribution(Seedable):
    """A class for representing random distributions.

    Created by calling get_distribution(params) on a Seedable object.
    Distributions are callable, such that some_seedable.get_distribution(params)()
    should return a random number from the distribution.
    """

    def __init__(self, params: Any, **kwargs):
        """Initialize distribution with given parameters.

        Args:
            params: Parameters for the distribution.
            **kwargs: Additional keyword arguments.
        """
        Seedable.__init__(self, **kwargs)
        self.params = params

    def __repr__(self) -> str:
        """Printable representation with params, seed, and parent.

        Returns:
            String representation of the object.
        """
        return f"{self.__class__.__name__}(params={self.params}, seed={self.rng_seed}, parent={self.parent})"

    def get_value(self) -> Any:
        """Samples from distribution function, returns a value.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            Value(s) from distribution.
        """
        raise NotImplementedError(
            "The Distribution class does not specify a distribution by itself. This must be specified by a subclass."
        )

    def __call__(self, size: int | None = None) -> Any | np.ndarray:
        """Distribution can return single value or np array of values sampled.

        Args:
            size: Number of values to return. Defaults to None.

        Returns:
            Value(s) sampled from distribution.
        """
        if size is None:
            return self.get_value()
        return np.array([self.get_value() for i in range(size)])


class ChoiceDistribution(Distribution):
    """A class for handling random choices from lists."""

    def __init__(self, params: list | np.ndarray | int, **kwargs):
        """Initialize choice distribution with given parameters.

        Args:
            params: List of values to choose from.
            **kwargs: Additional keyword arguments.
        """
        Distribution.__init__(self, params, **kwargs)

    def get_value(self) -> Any:
        """Samples a random value from the list of choices.

        Returns:
            Randomly selected value from the list.
        """
        return self.random_generator.choice(self.params)


class UniformRangeDistribution(Distribution):
    """A class for handling random uniform ranges."""

    def __init__(self, params: tuple[float, float], **kwargs):
        """Initialize uniform range distribution with given parameters.

        Args:
            params: Tuple of (low, high) values defining the range.
            **kwargs: Additional keyword arguments.
        """
        Distribution.__init__(self, params, **kwargs)

    def get_value(self) -> Any:
        """Samples a random value from the uniform distribution.

        Returns:
            Random value between low and high.
        """
        return self.random_generator.uniform(low=self.params[0], high=self.params[1])


class Log10UniformRangeDistribution(Distribution):
    """A class for handling log10-weighted random uniform ranges."""

    def __init__(self, params: tuple[float, float], **kwargs):
        """Initialize log10 uniform range distribution with given parameters.

        Args:
            params: Tuple of (low, high) values defining the range.
            **kwargs: Additional keyword arguments.
        """
        Distribution.__init__(self, params, **kwargs)

    def get_value(self) -> Any:
        """Samples a random value from the log10-weighted uniform distribution.

        Returns:
            Random value from the log10-weighted uniform distribution.

        Raises:
            ValueError: If params contain 0 or negative numbers.
        """
        if np.equal(self.params[0], 0) or np.equal(self.params[1], 0):
            raise ValueError(f"Cannot compute log10(0). params = {self.params}")
        if self.params[0] < 0 or self.params[1] < 0:
            raise ValueError(
                f"Cannot compute log10 of negative number. params = {self.params}"
            )

        low_log10 = np.log10(self.params[0])
        high_log10 = np.log10(self.params[1])
        random_exponent = self.random_generator.uniform(low=low_log10, high=high_log10)
        return 10**random_exponent


class UniformDistribution(Distribution):
    """A class for handling uniform random variables."""

    def __init__(self, params: float, **kwargs):
        """Initialize uniform distribution with given parameters.

        Args:
            params: Upper bound for the uniform distribution.
            **kwargs: Additional keyword arguments.
        """
        Distribution.__init__(self, params, **kwargs)

    def get_value(self) -> Any:
        """Samples a random value from the uniform distribution.

        Returns:
            Random value between 0 and high.
        """
        return self.random_generator.uniform(high=self.params)
