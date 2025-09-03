"""Metadata Transforms
"""

__all__ = [
    "MetadataTransform",
    "FamilyName",
    "FamilyIndex",
    "CustomLabel",
    "YOLOLabel"
]

# TorchSig
from torchsig.signals.signal_types import SignalMetadata, Signal
from torchsig.transforms.base_transforms import Transform
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.printing import generate_repr_str

# Built-In
from typing import List, Optional, Dict


## Base/Helper Classes
class MetadataTransform(Transform):
    """Metadata Transform base class

    This class defines the basic structure of a metadata transform, which includes:
    - The ability to validate metadata before applying the transform.
    - A method for applying the transform on signal metadata.
    - A callable interface to apply the transform to a list of signal metadata.

    Attributes:
        required_metadata (List[str]): List of metadata fields required for applying the target transform.

    Methods:
        __validate(metadata: Dict[str, Any]) -> Dict[str, Any]:
            Validates the signal metadata before applying the transform.
        
        __apply(metadata: Dict[str, Any]) -> Dict[str, Any]:
            Applies the target transform to the metadata. Should be overridden by subclasses.
        
        __call__(signal: Signal, enable_verify: bool = True)
            Applies the transform to a list of signal metadata dictionaries.

        __str__() -> str:
            Returns the string representation of the transform.

        __repr__() -> str:
            Returns a detailed string representation of the transform object.
    """
    def __init__(
        self,
        required_metadata: List[str] = [],
        **kwargs
    ) -> None:
        super().__init__(
            required_metadata=required_metadata,
            **kwargs
        )

    def __validate__(self, signal: SignalMetadata) -> SignalMetadata:
        """Validate signal metadata before applying target transforms
        makes sure a signal has all required metadata for a transform;
        returns the original signal if it is valid; raises an exception otherwise 

        Raises:
            ValueError: If metadata is missing required metadata fields.
        """        
        if not isinstance(signal, SignalMetadata):
            raise ValueError(f"metadata ({type(signal)}) is not a SignalMetadata object.")


        for required_metadatum in self.required_metadata:
            if not hasattr(signal, required_metadatum):
                raise ValueError(f"key: {required_metadatum} is missing from signal metadata, but is required by {self.__class__.__name__}")
            
        return signal
    
    def __call__(
        self, 
        signal: Signal,
        enable_verify = True
    ):
        """Applies the target transform to a list of signal metadata.
        """
        # apply metadata transform
        metadatas = signal.get_full_metadata()
        for metadata in metadatas:
            # verify signal metadata is valid
            if enable_verify:
                metadata = self.__validate__(metadata)

            # update dict with new metadata fields
            metadata = self.__apply__(metadata)

        return signal

    def __apply__(self, signal: SignalMetadata):
        """Applies the target transform to a single signal metadata.
        
        Args:
            signal SignalMetadata: The metadata to transform.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:   
        return generate_repr_str(self, exclude_params = ['required_metadata'])


class CustomLabel(MetadataTransform):
    """Adds a 'label' field to the metadata, which contains a tuple of fields 
    specified in the `label_fields` attribute.

    Attributes:
        label_fields (List[str]): The list of metadata fields to extract and place in the 'label' tuple.
    """

    def __init__(
        self, 
        label_fields: List[str], 
        label_name: str = 'label', 
        **kwargs
    ):
        super().__init__(
            required_metadata=label_fields,
            **kwargs
        )
        self.required_metadata = label_fields
        self.label_name = label_name

    def __apply__(self, signal: SignalMetadata) -> SignalMetadata:
        setattr(signal, self.label_name, tuple(getattr(signal, field) for field in self.required_metadata))
        return signal

class FamilyName(MetadataTransform):
    """
    Adds a family_name to a signal's metadata based on it's class_name

    Attributes:
        class_family_dict (Optional[Dict[str, str]], optional): Class name to Family name dict (keys=class name, values= family name). Defaults to TorchSigSignalLists.family_dict.
    """

    def __init__(
        self, 
        class_family_dict: Optional[Dict[str, str]] = TorchSigSignalLists.family_dict, 
        **kwargs
    ):    
        super().__init__(
            required_metadata=["class_name"],
            **kwargs
        )
        self.targets_metadata = ["family_name"]
        self.class_family_dict = class_family_dict

    
    def __apply__(self, signal: SignalMetadata) -> SignalMetadata:    
        setattr(signal, "family_name", self.class_family_dict[getattr(signal, "class_name")])
        return signal
        

class FamilyIndex(MetadataTransform):
    """
    Adds a family_index to a signal's metadata based on it's class_name

    Attributes:
        class_family_dict (Optional[Dict[str, str]], optional): Class name to Family name dict (keys=class name, values= family name). Defaults to TorchSigSignalLists.family_dict.
        family_list (Optional[List[str]], optional): Family list to index by. Defaults to alphabetical list of `class_family_dict` family names.
    """

    def __init__(
        self, 
        class_family_dict: Optional[Dict[str, str]] = TorchSigSignalLists.family_dict, 
        family_list: Optional[List[str]] = None, 
        **kwargs
    ):    
        super().__init__(
            required_metadata=["class_name"],
            **kwargs
        )
        self.targets_metadata = ["family_id"]
        self.class_family_dict = class_family_dict
        self.family_list = sorted(list(set(self.class_family_dict.values()))) if family_list is None else family_list

    
    def __apply__(self, signal: SignalMetadata) -> SignalMetadata: 

        fam_name = self.class_family_dict[getattr(signal, "class_name")]
        setattr(signal, "family_id", self.family_list.index(fam_name))
        return signal

class YOLOLabel(MetadataTransform):
    """
    Adds a YOLO_label to a signal, in the form of a list of tuples (cid, cx, cy, width, height)
    """

    def __init__(self, **kwargs):
        super().__init__(
            required_metadata=[
                "class_index",
                "start",
                "bandwidth",
                "center_freq",
                "sample_rate"
            ],
            **kwargs
        )
        self.targets_metadata = ["yolo_label"]

    
    def __apply__(self, signal: SignalMetadata) -> SignalMetadata:
        class_index = signal.class_index
        # normalized to width of sample
        width = signal.duration
        # normalize bandwidth with sample rate
        height = signal.bandwidth/signal.sample_rate
        x_center = signal.start + (width / 2.0)
        # normalize center frequency with sample rate
        # subtract from 1 since (0,0) for YOLO is upper left, but we define (0,0) lower left
        y_center = 1 - ((signal.sample_rate/2.0) + signal.center_freq) / signal.sample_rate
        yolo_label = (class_index, x_center, y_center, width, height)
        setattr(signal, "yolo_label", yolo_label)

        return signal
