from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig.utils.abstractions import HierarchicalMetadataObject

if TYPE_CHECKING:
    from torchsig.signals import Signal


class BaseSignalGenerator(HierarchicalMetadataObject):
    """Defines a callable object which takes no arguments and returns a Signal.

    Takes a metadata object in init to specify values for things like min and max bandwidth.

    Attributes:
        metadata: A metadata object to be used in signal generation.
        transforms: Transforms to be applied to generated signals before returning them in the __call__() method
    """

    def __init__(self, transforms: list[Any] = [], **kwargs: dict[str, Any]) -> None:
        """Initializes Signal Builder.

        Args:
            transforms: List of transforms to be applied to generated signals before returning them in the __call__() method
            **kwargs: Additional keyword arguments to pass to the parent class (HierarchicalMetadataObject)
        """
        self.transforms = transforms
        HierarchicalMetadataObject.__init__(self, **kwargs)

    def set_default_class_name(self, name: str) -> None:
        """Sets the class_name to name if there wasn't already a class name set.

        Args:
            name: The class name to set if no class name exists.
        """
        if not hasattr(self, "class_name"):
            self["class_name"] = name

    def copy(self) -> BaseSignalGenerator:
        """Creates a deep copy of the SignalGenerator with copied transforms.

        Returns:
            A new instance of the SignalGenerator with copied metadata and transforms.
        """
        cpy = HierarchicalMetadataObject.copy(self)
        cpy.transforms = [transform.copy() for transform in self.transforms]
        return cpy

    def validate_metadata_fields(self) -> None:
        """Validates that all required metadata fields are present.

        Throws an exception if required_metadata_fields are not filled.
        Does nothing if required_metadata_fields is not set.

        Raises:
            TypeError: If any required metadata field names are not strings.
            ValueError: If any required metadata fields are missing.
        """
        try:
            _ = self.required_metadata_fields
        except:
            return
        for key in self.required_metadata_fields:
            if not isinstance(key, str):
                raise TypeError(
                    "Could not validate metadata; all required metadata field names should be strings"
                )
            try:
                _ = self[key]
            except AttributeError as err:
                raise ValueError(
                    f"{self.__class__.__name__} missing required metadata key: '{key}'"
                ) from err

    def __call__(self) -> Signal:
        """Generates a new signal and applies all transforms.

        Returns:
            The generated signal after applying all transforms.
        """
        new_signal = self.generate()  # generate the signal
        new_signal.add_parent(self)
        if hasattr(self, "class_name"):
            new_signal["class_name"] = (
                self.class_name
            )  # if a class_name is given, it will override any class_name already in signal.metadata
        for transform in self.transforms:  # apply all transforms
            new_signal = transform(new_signal)
        return new_signal

    def __repr__(self) -> str:
        """Returns a string representation of the SignalGenerator.

        Returns:
            A string representation showing the class name, metadata, and transforms.
        """
        repr_str = f"{self.__class__.__name__}("
        if self._metadata is not None:
            repr_str += "metadata="
            repr_str += str(self._metadata)
            repr_str += ", "
        if self.transforms is not None:
            repr_str += "transforms="
            repr_str += str(self.transforms)
            repr_str += ", "
        repr_str += ")"
        return repr_str

    def generate(self) -> Signal:
        """Generates a new signal.

        This method must be implemented by subclasses.

        Returns:
            A new Signal object.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement 'build'")


class ConcatSignalGenerator(BaseSignalGenerator):
    """A Signal Generator that wraps other signal generators and returns one of their outputs at random when called.

    This generator randomly selects one of the provided signal generators and returns its output.
    Each wrapped signal generator must be a valid BaseSignalGenerator instance.

    Attributes:
        signal_generators: List of BaseSignalGenerator instances to choose from.
        random_generator: Random number generator used to select a signal generator.
    """

    def __init__(
        self, signal_generators: list[BaseSignalGenerator], **kwargs: dict[str, Any]
    ) -> None:
        """Initializes the ConcatSignalGenerator.

        Args:
            signal_generators: List of BaseSignalGenerator instances to wrap.
            **kwargs: Additional keyword arguments to pass to the parent class.

        Raises:
            TypeError: If any of the signal_generators are not BaseSignalGenerator instances.
        """
        BaseSignalGenerator.__init__(self, **kwargs)
        self.signal_generators = signal_generators
        for signal_generator in self.signal_generators:
            if True:  # isinstance(signal_generator, Seedable):
                signal_generator.add_parent(self)
        try:
            if self.validate_init:
                signal_generator.validate_metadata_fields()
        except AttributeError:
            pass  # there is no validate function; ignore and assume the best; a user who doesn't write a validate function does so at their own risk

    def copy(self) -> ConcatSignalGenerator:
        """Creates a deep copy of the ConcatSignalGenerator with copied signal generators.

        Returns:
            A new instance of ConcatSignalGenerator with copied metadata and signal generators.
        """
        cpy = BaseSignalGenerator.copy(self)
        cpy.signal_generators = [
            signal_generator.copy() for signal_generator in self.signal_generators
        ]
        return cpy

    def validate_metadata_fields(self) -> bool:
        """Validates metadata fields for all wrapped signal generators.

        Calls validate_metadata_fields() on each wrapped signal generator.

        Returns:
            bool: True if all validations pass.

        Raises:
            ValueError: If any of the wrapped signal generators are missing required metadata fields.
        """
        for generator in self.signal_generators:
            generator.validate_metadata_fields()
        return True

    def __repr__(self) -> str:
        """Returns a string representation of the ConcatSignalGenerator.

        Returns:
            A string representation showing the class name, metadata, and signal generators.
        """
        repr_str = f"{self.__class__.__name__}("
        if self._metadata is not None:
            repr_str += "metadata="
            repr_str += str(self._metadata)
            repr_str += ", "
        if self.signal_generators is not None:
            repr_str += "signal_generators="
            repr_str += str(self.signal_generators)
            repr_str += ", "
        repr_str += ")"
        return repr_str

    def generate(self) -> Signal:
        """Generates a signal by randomly selecting one of the wrapped generators.

        Returns:
            Signal: The output of a randomly selected signal generator.
        """
        return self.random_generator.choice(self.signal_generators)()
