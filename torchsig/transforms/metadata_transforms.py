"""Metadata Transforms"""

__all__ = [
    "MetadataTransform",
    "YOLOLabel"
]

from torchsig.signals.signal_types import Signal
from torchsig.transforms.base_transforms import Transform
from torchsig.utils.printing import generate_repr_str


## Base/Helper Classes
class MetadataTransform(Transform):
    """Base class for metadata transforms.

    This class defines the basic structure of a metadata transform, which includes:
    - The ability to validate metadata before applying the transform.
    - A method for applying the transform on signal metadata.
    - A callable interface to apply the transform to a list of signal metadata.

    Attributes:
        required_metadata: List of metadata fields required for applying the target transform.

    Methods:
        __validate(metadata): Validates the signal metadata before applying the transform.
        __apply(metadata): Applies the target transform to the metadata. Should be overridden by subclasses.
        __call__(signal): Applies the transform to a list of signal metadata dictionaries.
        __str__(): Returns the string representation of the transform.
        __repr__(): Returns a detailed string representation of the transform object.
    """

    def __init__(self, required_metadata: list[str] = [], **kwargs) -> None:
        """Initialize the MetadataTransform.

        Args:
            required_metadata: List of metadata fields required for applying the target transform.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(required_metadata=required_metadata, **kwargs)

    def __validate__(self, signal):
        """Validate signal metadata before applying target transforms.

        Makes sure a signal has all required metadata for a transform;
        returns the original signal if it is valid; raises an exception otherwise.

        Args:
            signal: The signal to validate.

        Raises:
            ValueError: If metadata is missing required metadata fields or if input is not a Signal object.
        """
        if not isinstance(signal, Signal):
            raise TypeError(f"input ({type(signal)}) is not a Signal object.")
        for required_metadatum in self.required_metadata:
            if not hasattr(signal, required_metadatum):
                raise ValueError(
                    f"key: {required_metadatum} is missing from signal metadata, but is required by {self.__class__.__name__}"
                )
        return signal

    def __call__(self, signal: Signal) -> Signal:
        """Applies the target transform to a list of signal metadata.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal.
        """
        for component_signal in signal.component_signals:
            self.__apply__(component_signal)
        return signal

    def __apply__(self, signal):
        """Applies the target transform to a single signal metadata.

        Args:
            signal: The signal to transform.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Returns a detailed string representation of the transform object.

        Returns:
            A string representation of the transform object.
        """
        return generate_repr_str(self, exclude_params=["required_metadata"])


class YOLOLabel(MetadataTransform):
    """Adds a YOLO_label to a signal.

    This transform adds a YOLO_label to a signal in the form of a list of tuples (cid, cx, cy, width, height).

    Attributes:
        required_metadata: List of metadata fields required for applying the transform.
        targets_metadata: List of metadata fields that will be added by the transform.
    """

    def __init__(self, **kwargs):
        """Initialize the YOLOLabel transform.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            required_metadata=[
                "class_index",
                "start",
                "bandwidth",
                "center_freq",
                "dataset_metadata",
            ],
            **kwargs,
        )
        self.targets_metadata = ["yolo_label"]

    def __apply__(self, signal: Signal) -> Signal:
        """Applies the YOLOLabel transform to a single signal.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal with YOLO_label added.
        """
        class_index = signal.class_index
        # normalized to width of sample
        width = signal.duration
        # normalize bandwidth with sample rate
        height = signal.bandwidth / signal.sample_rate
        x_center = signal.start + (width / 2.0)
        # normalize center frequency with sample rate
        # subtract from 1 since (0,0) for YOLO is upper left, but we define (0,0) lower left
        y_center = (
            1 - ((signal.sample_rate / 2.0) + signal.center_freq) / signal.sample_rate
        )
        yolo_label = (class_index, x_center, y_center, width, height)
        signal["yolo_label"] = yolo_label
        return signal
