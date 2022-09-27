import numpy as np
from typing import Tuple, List, Any, Union, Optional

from torchsig.utils import SignalDescription
from torchsig.transforms.transforms import Transform


class DescToClassName(Transform):
    """Transform to transform SignalDescription into either the single class name
    or a list of the classes present if there are multiple classes

    """
    def __init__(self):
        super(DescToClassName, self).__init__()

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> Union[List[str], str]:
        classes = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        for signal_desc_idx, signal_desc in enumerate(signal_description):
            curr_class_name = signal_desc.class_name[0] if isinstance(signal_desc.class_name, list) else signal_desc.class_name
            classes.append(curr_class_name)
        if len(classes) > 1:
            return classes
        elif len(classes) == 1:
            return classes[0]
        else:
            return []
        

class DescToClassNameSNR(Transform):
    """Transform to transform SignalDescription into either the single class name
    or a list of the classes present if there are multiple classes along with
    the SNRs for each

    """
    def __init__(self):
        super(DescToClassNameSNR, self).__init__()

    def __call__(
        self, 
        signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[Tuple[List[str],List[float]], Tuple[str, float]]:
        classes = []
        snrs = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        for signal_desc_idx, signal_desc in enumerate(signal_description):
            classes.append(signal_desc.class_name)
            snrs.append(signal_desc.snr)
        if len(classes) > 1:
            return classes, snrs
        else:
            return classes[0], snrs[0]
        

class DescToClassIndex(Transform):
    """Transform to transform SignalDescription into either the single class index
    or a list of the class indices present if there are multiple classes. Note:
    if the SignalDescription contains classes not present in the provided 
    `class_list`, the SignalDescription is interpretted as having no classes 
    present
    
    Args:
        class_list (:obj:`List[str]`):
            A full list of classes to map the class names to indices

    """
    def __init__(self, class_list: List[str] = None):
        super(DescToClassIndex, self).__init__()
        self.class_list = class_list

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> Union[List[int], int]:
        classes = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        for signal_desc_idx, signal_desc in enumerate(signal_description):
            if signal_desc.class_name in self.class_list:
                classes.append(self.class_list.index(signal_desc.class_name))
        if len(classes) > 1:
            return classes
        else:
            return classes[0]

        
class DescToClassIndexSNR(Transform):
    """Transform to transform SignalDescription into either the single class index
    or a list of the class indices present if there are multiple classes along
    with the SNRs of each. Note: if the SignalDescription contains classes not 
    present in the provided `class_list`, the SignalDescription is interpretted as
    having no classes present
    
    Args:
        class_list (:obj:`List[str]`):
            A full list of classes to map the class names to indices

    """
    def __init__(self, class_list: List[str] = None):
        super(DescToClassIndexSNR, self).__init__()
        self.class_list = class_list

    def __call__(
        self, 
        signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[Tuple[List[int], List[float]], Tuple[int, float]]:
        classes = []
        snrs = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        for signal_desc_idx, signal_desc in enumerate(signal_description):
            if signal_desc.class_name in self.class_list:
                classes.append(self.class_list.index(signal_desc.class_name))
                snrs.append(signal_desc.snr)
        if len(classes) > 1:
            return classes, snrs
        else:
            return classes[0], snrs[0]

        
class DescPassThrough(Transform):
    """Transform to simply pass the SignalDescription through. Same as applying no
    transform in most cases.
            
    """
    def __init__(self):
        super(DescPassThrough, self).__init__()

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> Union[List[SignalDescription], SignalDescription]:
        return signal_description
    
    
class DescToBinary(Transform):
    """Transform to transform SignalDescription into binary 0/1 label

    Args:
        label (:obj:`int`):
            Binary label to assign
            
    """
    def __init__(self, label: int):
        super(DescToBinary, self).__init__()
        self.label = label

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> int:
        return self.label

    
class DescToCustom(Transform):
    """Transform to transform SignalDescription into any static value

    Args:
        label (:obj:`Any`):
            Custom static label to assign
            
    """
    def __init__(self, label: Any):
        super(DescToCustom, self).__init__()
        self.label = label

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> Any:
        return self.label
    
    
class DescToClassEncoding(Transform):
    """Transform to transform SignalDescription into one- or multi-hot class
    encodings. Note that either the number of classes or the full class list
    must be provided as input. If neither are provided, the transform will 
    raise an error, and if both are provided, the transform will default to 
    using the full class list. If only the number of classes are provided,
    the SignalDescription objects must contain the class index field

    Args:
        class_list (:obj:`Optional[List[str]]`):
            Class list

        num_classes (:obj:`Optional[int]`):
            Number of classes in the encoding
        
    """
    def __init__(
        self, 
        class_list: Optional[List[str]] = None,
        num_classes: Optional[int] = None, 
    ) -> np.ndarray:
        super(DescToClassEncoding, self).__init__()
        self.class_list = class_list
        self.num_classes = num_classes if num_classes else len(class_list)

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        encoding = np.zeros((self.num_classes,))
        for signal_desc in signal_description:
            if self.class_list:
                encoding[self.class_list.index(signal_desc.class_name)] = 1.0
            else:
                encoding[signal_desc.class_index] = 1.0  
        return encoding
    

class DescToWeightedMixUp(Transform):
    """Transform to transform SignalDescription into weighted multi-hot class
    encodings.

    Args:
        class_list (:obj:`Optional[List[str]]`):
            Class list
        
    """
    def __init__(
        self, 
        class_list: List[str] = None,
    ) -> np.ndarray:
        super(DescToWeightedMixUp, self).__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        encoding = np.zeros((self.num_classes,))
        # Instead of a binary value for the encoding, set it to the SNR
        for signal_desc in signal_description:
            encoding[self.class_list.index(signal_desc.class_name)] += signal_desc.snr
        # Next, normalize to the total of all SNR values
        encoding = encoding / np.sum(encoding)
        return encoding
    
    
class DescToWeightedCutMix(Transform):
    """Transform to transform SignalDescription into weighted multi-hot class
    encodings.

    Args:
        class_list (:obj:`Optional[List[str]]`):
            Class list
        
    """
    def __init__(
        self, 
        class_list: List[str] = None,
    ) -> np.ndarray:
        super(DescToWeightedCutMix, self).__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)

    def __call__(self, signal_description: Union[List[SignalDescription], SignalDescription]) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description = [signal_description] if isinstance(signal_description, SignalDescription) else signal_description
        encoding = np.zeros((self.num_classes,))
        # Instead of a binary value for the encoding, set it to the cumulative duration
        for signal_desc in signal_description:
            encoding[self.class_list.index(signal_desc.class_name)] += signal_desc.duration
        # Normalize on total signals durations
        encoding = encoding / np.sum(encoding)
        return encoding
    
    
class LabelSmoothing(Transform):
    """Transform to transform a numpy array encoding to a smoothed version to 
    assist with overconfidence. The input hyperparameter `alpha` determines the
    degree of smoothing with the following equation:
    
        output = (1 - alpha) / num_hot * input + alpha / num_classes,
        
    Where,
        output ~ Smoothed output encoding
        alpha ~ Degree of smoothing to apply
        num_hot ~ Number of positively-labeled classes
        input ~ Input one/multi-hot encoding
        num_classes ~ Number of classes
        
    Note that the `LabelSmoothing` transform accepts a numpy encoding input, 
    and as such, should be used in conjunction with a preceeding 
    DescTo... transform that maps the SignalDescription to the expected
    numpy encoding format.
        
    Args:
        alpha (:obj:`float`):
            Degree of smoothing to apply
    
    """
    def __init__(self, alpha: float = 0.1) -> np.ndarray:
        super(LabelSmoothing, self).__init__()
        self.alpha = alpha

    def __call__(self, encoding: np.ndarray) -> np.ndarray:
        return (1 - self.alpha) / np.sum(encoding) * encoding + (self.alpha / encoding.shape[0])
    
