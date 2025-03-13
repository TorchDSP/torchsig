"""Target Transforms
"""

__all__ = [
    "TargetTransform",
    "FamilyName",
    "FamilyIndex",
    "CustomLabel",
    "YOLOLabel",
]

# TorchSig
from torchsig.transforms.base_transforms import Transform
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.printing import generate_repr_str

# Built-In
from typing import List, Any, Optional, Dict


## Base/Helper Classes
class TargetTransform(Transform):
    """Target Transform base class

    This class defines the basic structure of a target transform, which includes:
    - The ability to validate metadata before applying the transform.
    - A method for applying the transform on signal metadata.
    - A callable interface to apply the transform to a list of signal metadata.

    Attributes:
        required_metadata (List[str]): List of metadata fields required for applying the target transform.
        targets_metadata (List[str]): List of target metadata fields to be added to output of target transform.

    Methods:
        __validate(metadata: Dict[str, Any]) -> Dict[str, Any]:
            Validates the signal metadata before applying the transform.
        
        __apply(metadata: Dict[str, Any]) -> Dict[str, Any]:
            Applies the target transform to the metadata. Should be overridden by subclasses.
        
        __call__(metadatas: List[Dict[str, Any]], enable_verify: bool = True) -> List[Dict[str, Any]]:
            Applies the transform to a list of signal metadata dictionaries.

        __str__() -> str:
            Returns the string representation of the transform.

        __repr__() -> str:
            Returns a detailed string representation of the transform object.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # what metadata fields are requried for target transform to be applied
        self.required_metadata = []
        # when computing targets of target transform, what fields to use
        self.targets_metadata = []

    def __validate__(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal metadata before applying target transforms
        makes sure a signal has all required metadata for a transform;
        returns the original signal if it is valid; raises an exception otherwise 

        Raises:
            ValueError: If metadata is not a dict or is missing required metadata fields.

        Returns:
            Dict[str, Any]: Validated signal metadata.
        """        
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata ({type(metadata)}) is not a list.")

        for required_metadatum in self.required_metadata:
            if not required_metadatum in metadata.keys():
                raise ValueError(f"key: {required_metadatum} is missing from signal metadata, but is required by {self.__class__.__name__}")
            
        return metadata

    def __apply__(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the target transform to a single signal metadata.
        
        Args:
            metadata (Dict[str, Any]): The metadata to transform.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            Dict[str, Any]: The transformed metadata.
        """
        raise NotImplementedError
    
    def __call__(
        self, 
        metadatas: List[Dict[str, Any]],
        enable_verify = True
    ) -> List[Any] | Dict[str, Any]:
        """Applies the target transform to a list of signal metadata.

        Args:
            metadatas (List[Dict[str, Any]]): The list of metadata dictionaries to transform.
            enable_verify (bool, optional): Whether to verify metadata before transforming. Defaults to True.

        Returns:
            List[Dict[str, Any]]: The transformed list of metadata dictionaries.
        """
        # apply target transform
        for metadata in metadatas:
            # verify signal metadata is valid
            if enable_verify:
                metadata = self.__validate__(metadata)

            # update dict with new metadata fields
            metadata = self.__apply__(metadata)

        return metadatas

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
    
    def __repr__(self) -> str:   
        return generate_repr_str(self, exclude_params = ['required_metadata', 'targets_metadata'])


class CustomLabel(TargetTransform):
    """Adds a 'label' field to the metadata, which contains a tuple of fields 
    specified in the `label_fields` attribute.

    Attributes:
        label_fields (List[str]): The list of metadata fields to extract and place in the 'label' tuple.
    """

    def __init__(self, label_fields: List[str], label_name: str = 'label', **kwargs):
        super().__init__(**kwargs)
        self.required_metadata = label_fields
        self.targets_metadata = [label_name]

    
    def __apply__(self, metadata):    
        metadata[self.targets_metadata[0]] = tuple([metadata[field] for field in self.required_metadata])
        return metadata

 

class PassThrough(TargetTransform):
    """A helper class that does not alter the signal metadata but adds requested fields to the output.

    This class is often used in combination with other transforms.
    """  
    def __init__(self, field: List[str] = [], **kwargs):
        super().__init__(**kwargs)
        self.required_metadata = field
        self.targets_metadata = field
    
    def __apply__(self, metadata: dict):
        return metadata


### Built-In Target Transforms
# These target transforms already have labels within the Signal class, 
# which is turned into a dictionary inside the DatasetDict class. Thus,
# they do not any further processig than grabbing the label
###

class CenterFreq(PassThrough):
    """Adds `center_freq` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['center_freq'])

class Bandwidth(PassThrough):
    """Adds `bandwidth` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['bandwidth'])

class StartInSamples(PassThrough):
    """Adds `start_in_samples` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['start_in_samples'])

class DurationInSamples(PassThrough):
    """Adds `duration_in_samples` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['duration_in_samples'])

class SNR(PassThrough):
    """Adds `snr_db` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['snr_db'])

class ClassName(PassThrough):
    """Adds `class_name` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['class_name'])

class ClassIndex(PassThrough):
    """Adds `class_index` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['class_index'])

class SampleRate(PassThrough):
    """Adds `sample_rate` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['sample_rate'])

class NumSamples(PassThrough):
    """Adds `num_samples` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['num_samples'])

class Start(PassThrough):
    """Adds `start` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['start'])

class Stop(PassThrough):
    """Adds `stop` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['stop'])

class Duration(PassThrough):
    """Adds `duration` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['duration'])

class StopInSamples(PassThrough):
    """Adds `stop_in_samples` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['stop_in_samples'])

class UpperFreq(PassThrough):
    """Adds `upper_freq` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['upper_freq'])

class LowerFreq(PassThrough):
    """Adds `lower_freq` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['lower_freq'])

class OversamplingRate(PassThrough):
    """Adds `oversampling_rate` from signal metadata
    """ 
    def __init__(self, **kwargs):
        super().__init__(field = ['oversampling_rate'])


# Special Target Transforms
# Target Transforms that require calculation to generate.
# They also need their metadata label field added to the metadata.

class FamilyName(TargetTransform):
    """
    Adds a family_name to a signal's metadata based on it's class_name

    Attributes:
        class_family_dict (Optional[Dict[str, str]], optional): Class name to Family name dict (keys=class name, values= family name). Defaults to TorchSigSignalLists.family_dict.
    """

    def __init__(self, class_family_dict: Optional[Dict[str, str]] = TorchSigSignalLists.family_dict, **kwargs):    
        super().__init__(**kwargs)
        self.required_metadata = ["class_name"]
        self.targets_metadata = ["family_name"]
        self.class_family_dict = class_family_dict

    
    def __apply__(self, metadata):    
        metadata["family_name"] = self.class_family_dict[metadata["class_name"]]
        return metadata
        

class FamilyIndex(TargetTransform):
    """
    Adds a family_index to a signal's metadata based on it's class_name

    Attributes:
        class_family_dict (Optional[Dict[str, str]], optional): Class name to Family name dict (keys=class name, values= family name). Defaults to TorchSigSignalLists.family_dict.
        family_list (Optional[List[str]], optional): Family list to index by. Defaults to alphabetical list of `class_family_dict` family names.
    """

    def __init__(self, class_family_dict: Optional[Dict[str, str]] = TorchSigSignalLists.family_dict, family_list: Optional[List[str]] = None, **kwargs):    
        super().__init__(**kwargs)
        self.required_metadata = ["class_name"]
        self.targets_metadata = ["family_id"]
        self.class_family_dict = class_family_dict
        self.family_list = sorted(list(set(self.class_family_dict.values()))) if family_list is None else family_list

    
    def __apply__(self, metadata): 

        fam_name = self.class_family_dict[metadata["class_name"]]
        metadata["family_id"] = self.family_list.index(fam_name)
        return metadata

class YOLOLabel(TargetTransform):
    """
    Adds a YOLO_label to a signal, in the form of a list of tuples (cid, cx, cy, width, height)

    Attributes:
        output (str, optional): Structure to aggregate YOLO labels ("dict", "list"). Defaults to "list".
    """

    output_list = ["list", "dict"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_metadata = ["class_index", "start", "bandwidth", "center_freq", "sample_rate"]
        self.targets_metadata = ["yolo_label"]

    
    def __apply__(self, metadata):
        class_index = metadata["class_index"]
        # normalized to width of sample
        width = metadata["duration"]
        # normalize bandwidth with sample rate
        height = metadata["bandwidth"] / metadata["sample_rate"]
        x_center = metadata["start"] + (width / 2.0)
        # normalize center frequency with sample rate
        # subtract from 1 since (0,0) for YOLO is upper left, but we define (0,0) lower left
        y_center = 1 - ((metadata["sample_rate"] / 2.0) + metadata["center_freq"]) / metadata["sample_rate"]
        yolo_label = (class_index, x_center, y_center, width, height)
        metadata["yolo_label"] = yolo_label

        return metadata


