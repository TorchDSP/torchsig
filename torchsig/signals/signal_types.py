"""Signal and Signal Metadata classes.

This module defines the `Signal` and `SignalMetadata` classes and their associated functionality, 
which are used to represent and manipulate signal data and metadata.

Examples:
    Signal:
        >>> from torchsig.signals import Signal, SignalMetadata
        >>> d = [1.0, 2.0]
        >>> m = SignalMetadata(...)
        >>> new_sig = Signal(data = d, metadata = m)
"""

from __future__ import annotations

# TorchSig
from torchsig.utils.dsp import (
    lower_freq_from_center_freq_bandwidth,
    upper_freq_from_center_freq_bandwidth,
    center_freq_from_lower_upper_freq,
    bandwidth_from_lower_upper_freq,
    TorchSigComplexDataType
)
from torchsig.utils.verify import verify_numpy_array

# Third Party
import numpy as np

# Built-In
from typing import List, TYPE_CHECKING
import copy

# Imports for type checking
if TYPE_CHECKING:
    from torchsig.datasets.dataset_metadata import DatasetMetadata, ExternalDatasetMetadata


### Signal Metadata Types
signal_metadata_dict_types = {
    'center_freq':float,
    'bandwidth':float,
    'start_in_samples':int,
    'duration_in_samples':int,
    'snr_db':float,
    'class_name':str,
    'class_index':int,
    'sample_rate':float,
    'start':float,
    'stop':float,
    'duration':float,
    'stop_in_samples':int,
    'upper_freq':float,
    'lower_freq':float,
    'oversampling_rate':float
}
keys_types_list = [list(item)for item in list(zip(*signal_metadata_dict_types.items()))]

class SignalMetadata():
    """Represents metadata associated with a signal.

    Attributes:
        dataset_metadata (DatasetMetadata): The dataset metadata for the signal. Defaults to None.
        center_freq (float): The center frequency of the signal in Hz. Defaults to 0.0.
        bandwidth (float): The bandwidth of the signal in Hz. Defaults to 0.0.
        start_in_samples (int): The start time of the signal in terms of samples. Defaults to 0.
        duration_in_samples (int): The duration of the signal in terms of samples. Defaults to 0.
        snr_db (float): The Signal-to-Noise Ratio in dB. Defaults to 0.0.
        class_name (str): The class name of the signal (e.g., modulation type). Defaults to "None".
        class_index (int): The class index of the signal in the dataset. Defaults to -1.
    """
    def __init__(
        self,
        dataset_metadata: DatasetMetadata = None,
        center_freq: float = None,
        bandwidth: float = None,
        start_in_samples: int = None,
        duration_in_samples: int = None,
        snr_db: float = None,
        class_name: str = None,
        class_index: int = None,
        **kwargs
    ): 
        """Initializes the SignalMetadata object.

        Args:
            dataset_metadata (DatasetMetadata, optional): Metadata related to the dataset. Defaults to None.
            center_freq (float, optional): The center frequency of the signal in Hz. Defaults to 0.0.
            bandwidth (float, optional): The bandwidth of the signal in Hz. Defaults to 0.0.
            start_in_samples (int, optional): The start time in samples. Defaults to 0.
            duration_in_samples (int, optional): The duration in samples. Defaults to 0.
            snr_db (float, optional): The signal-to-noise ratio in decibels. Defaults to 0.0.
            class_name (str, optional): The class name of the signal. Defaults to "None".
            class_index (int, optional): The class index of the signal. Defaults to -1.
        """
        self.dataset_metadata = dataset_metadata
        # Core SignalMetadata fields
        self.center_freq = center_freq # center freq (-sample_rate/2, sample_rate/2)
        self.bandwidth = bandwidth # bandwidth in Hz
        self.start_in_samples = start_in_samples # index of signal start in IQ data
        self.duration_in_samples = duration_in_samples # signal length in IQ data array (num indicies)
        self.snr_db = snr_db # snr
        self.class_name = class_name # class modulation name
        self.class_index = class_index # class index wrt class list
        self._lower_frequency = None # starts as null; if we can, we will update the lower and upper frequency from center frequency and bandwidth
        self._upper_frequency = None
        self._lower_frequency = self.lower_freq
        self._upper_frequency = self.upper_freq

        # needed to enable/disable bounds checking for signal's center frequency.
        # since the center frequency will be set in TorchSigIterableDataset() after
        # transforms are applied
        self._center_freq_set = False

        self.applied_transforms = []

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def start(self) -> float:
        """Signal start normalized to duration of signal

        Returns signal start as a percentage of total time, ex: start=0.5 means
        the signal starts 50% of the way into the dataset IQ samples.

        Returns:
            float: signal start
        
        """
        return self.start_in_samples/self.dataset_metadata.num_iq_samples_dataset

    @start.setter
    def start(self, new_start: float):
        """Sets signal start

        Sets signal start as a percentage of total time, ex: start=0.5 means
        the signal starts 50% of the way into the dataset IQ samples.

        Args:
            new_start (float): The starting location as a percentage from 0.0 to 1.0.
        """
        self.start_in_samples = int(new_start * self.dataset_metadata.num_iq_samples_dataset)

    @property
    def stop(self) -> float:
        """Signal stop normalized to duration of signal

        Returns signal stop as a percentage of total time, ex: stop=0.5 means
        the signal stops 50% of the way into the dataset IQ samples.

        Returns:
            float: signal stop
        
        """
        return self.stop_in_samples/self.dataset_metadata.num_iq_samples_dataset

    @stop.setter
    def stop(self, new_stop: float):
        """Sets signal stop

        Sets signal stop as a percentage of total time, ex: stop=0.5 means
        the signal stops 50% of the way into the dataset IQ samples.

        Args:
            new_stop (float): The stopping location as a percentage from 0.0 to 1.0.
        """
        self.duration_in_samples = (new_stop * self.dataset_metadata.num_iq_samples_dataset) - self.start_in_samples

    @property
    def duration(self) -> float:
        """Signal duration (normalized)

        Returns signal duration normalized from 0.0 to 1.0

        Returns:
            float: signal duration
        """    
        return self.duration_in_samples/self.dataset_metadata.num_iq_samples_dataset

    @duration.setter
    def duration(self, new_duration: float):
        """Sets the duration of the signal based on a percentage of total time.

        Args:
            new_duration (float): The new duration as a percentage of total time.
        """
        self.duration_in_samples = new_duration * self.dataset_metadata.num_iq_samples_dataset

    @property
    def stop_in_samples(self) -> int:
        """Signal stop in samples

        Returns the index where the signal stops in the dataset IQ.

        Returns:
            int: signal stop
        
        """
        return self.start_in_samples + self.duration_in_samples

    @stop_in_samples.setter
    def stop_in_samples(self, new_stop_in_samples: int):
        """Sets the stop time of the signal in samples.

        Args:
            new_stop_in_samples (int): The new stop time in samples.
        """
        self.duration_in_samples = new_stop_in_samples - self.start_in_samples

    @property
    def upper_freq(self) -> float:
        """Calculates the upper frequency of a signal

        Calculates the upper frequency edge, or highest frequency, associated with
        the bandwidth of the signal.

        Returns:
            float: upper frequency
        
        """
        try:
            self._upper_frequency = upper_freq_from_center_freq_bandwidth(self.center_freq,self.bandwidth)
            return self._upper_frequency
        except Exception as e:
            return self._upper_frequency

    @upper_freq.setter
    def upper_freq(self, new_upper_freq: float):
        """Sets the upper frequency of the signal

        Sets the upper frequency and then updates the center frequency and bandwidth
        as they are directly related to the parameter.

        Args:
            new_upper_freq (float): The new upper frequency value
        """
        self._upper_frequency = new_upper_freq
        if self._lower_frequency is not None:
            self.bandwidth = bandwidth_from_lower_upper_freq(new_upper_freq,self.lower_freq)
            self.center_freq = center_freq_from_lower_upper_freq(new_upper_freq,self.lower_freq)

    @property
    def lower_freq(self) -> float:
        """Calculates the lower frequency of a signal

        Calculates the lower frequency edge, or lowest frequency, associated with
        the bandwidth of the signal.
        
        Returns:
            float: lower frequency
        
        """
        try:
            self._lower_frequency = lower_freq_from_center_freq_bandwidth(self.center_freq,self.bandwidth)
            return self._lower_frequency
        except Exception as e:
            return self._lower_frequency

    @lower_freq.setter
    def lower_freq(self, new_lower_freq: float):
        """Sets the lower frequency of the signal

        Sets the lower frequency and then updates the center frequency and bandwidth
        as they are directly related to the parameter.

        Args:
            new_lower_freq (float): The new lower frequency value
        """        
        self._lower_frequency = new_lower_freq
        if self._upper_frequency is not None:
            self.bandwidth = bandwidth_from_lower_upper_freq(self.upper_freq,new_lower_freq)
            self.center_freq = center_freq_from_lower_upper_freq(self.upper_freq,new_lower_freq)

    @property
    def oversampling_rate(self) -> float:
        """Calculates the oversampling rate for a signal

        Calculates the oversampling rate for a signal. If a signal's bandwidth
        is 1/2 the sampling rate, the oversampling rate is 2.

        Returns:
            float: oversampling rate
        
        """
        return self.sample_rate/self.bandwidth


    def to_dict(self) -> dict:
        """Returns SignalMetadataExternal as a full dictionary
        """
        attributes_original = self.__dict__.copy()  # Start with the instance variables

        attributes = attributes_original.copy()

        # exclude certain variables
        for var in attributes_original:
            if var in ["applied_transforms", "dataset_metadata", "_dataset_metadata", "_center_freq_set"]:
                del attributes[var]
        return attributes

    def deepcopy(self) -> SignalMetadata:
        """Returns a deep copy of itself

        Returns:
            SignalMetadata: Deep copy of SignalMetadata
        """        
        return copy.deepcopy(self)

    def __repr__(self):
        class_dict = self.to_dict()
        params = [f"{k}={v}" for k,v in class_dict.items()]
        params_str = ",".join(params)
        return f"{self.__class__.__name__}({params_str})"

def targets_as_metadata(targets, target_labels, dataset_metadata: DatasetMetadata = None) -> SignalMetadata:
    """Utility function for reading target labels as signal metadata objects; returns a new SignalMetadata

    Returns:
        SignalMetadata: new SignalMetadata object with target label
    """    
    signal_metadata = SignalMetadata(dataset_metadata=dataset_metadata)
    if not isinstance(targets,list):
        targets = [targets]
    for i in range(len(target_labels)):
        setattr(signal_metadata, target_labels[i], targets[i])
    return signal_metadata

def dict_to_signal_metadata(metadata_dict: dict, dataset_metadata: DatasetMetadata = None) -> SignalMetadata:
    """converts a dict to SignalMetadata

    Args:
        metadata_dict (dict): metadata
        dataset_metadata (DatasetMetadata, optional): dataset metadata related to metadata. Defaults to None.

    Returns:
        SignalMetadata: dict converted to SignalMetadata object
    """    
    return targets_as_metadata([metadata_dict[key] for key in metadata_dict.keys()], list(metadata_dict.keys()), dataset_metadata)

### Signal
class Signal():
    """Initializes the Signal with data and metadata.

        Args:
            data (np.ndarray, optional): Signal IQ data. Defaults to np.array([]).
            metadata (SignalMetadata | SignalMetadataExternal, optional): Signal metadata. Defaults to an empty instance of SignalMetadata().
        """
    def __init__(
        self, 
        data = np.array([]), 
        metadata = None, 
        component_signals: List[Signal] = [],
        dataset_metadata = None,
    ):
        """Initializes the Signal with data and metadata.

        Args:
            data: Signal IQ data. Defaults to np.array([])
            metadata: Signal metadata; Defaults to None
            component_signals (List[Signal], optional): individual components of the full signal, e.g. smaller individual signals collected together in a wideband signal. Defaults to []
            dataset_metadata: overrides dataset_metadata for metadata; sometimes useful in constructors or custom file readers; generally safe to ignore
        """
        self.data = data
        self.metadata = metadata
        if isinstance(self.metadata, dict):
            self.metadata = dict_to_signal_metadata(self.metadata)
        if dataset_metadata is not None:
            self.metadata.dataset_metadata = dataset_metadata
        self.component_signals = component_signals

    def verify(self):
        """Verifies data and metadata are valid.
        Should only be run after `SignalBuilder`. Signals post-transforms are not guaranteed to pass verify() checks.

        Raises:
            ValueError: Data or metadata is invalid.
        """
        # convert lists to array
        self.data = verify_numpy_array(
            self.data,
            name = "IQ data",
            exact_length=self.metadata.duration_in_samples,
            data_type=TorchSigComplexDataType
        )

        self.metadata.verify()


    def get_full_metadata(self):
        """
        Returns a list of all top level metadata objects in the Signal. 
        If no metadata is defined on a Signal, it's metadata is assumed to be the list of metadata of it's children.
        This process is applied recursively until no more children without metadata can be found.
        """
        if not self.metadata is None:
            return [self.metadata]
        metadatas = []
        for component_signal in self.component_signals:
            component_metadata = component_signal.get_full_metadata()
            metadatas += component_metadata
        return metadatas


    def __repr__(self):
        return f"{self.__class__.__name__}(metadata={self.metadata}, component_signals={self.component_signals})"

