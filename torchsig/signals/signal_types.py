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
    torchsig_complex_data_type
)
from torchsig.utils.verify import (
    verify_int,
    verify_float,
    verify_str,
    verify_numpy_array,
    verify_dict
)

# Third Party
import numpy as np

# Built-In
from typing import List, TYPE_CHECKING, Dict, Any
import copy

# Imports for type checking
if TYPE_CHECKING:
    from torchsig.datasets.dataset_metadata import DatasetMetadata


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
    'num_samples':int,
    'start':float,
    'stop':float,
    'duration':float,
    'stop_in_samples':int,
    'upper_freq':float,
    'lower_freq':float,
    'oversampling_rate':float,
    'samples_per_baud':float,
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
        self._dataset_metadata = dataset_metadata
        # Core SignalMetadata fields
        self.center_freq = center_freq # center freq (-sample_rate/2, sample_rate/2)
        self.bandwidth = bandwidth # bandwidth in Hz
        self.start_in_samples = start_in_samples # index of signal start in IQ data
        self.duration_in_samples = duration_in_samples # signal length in IQ data array (num indicies)
        self.snr_db = snr_db # snr
        self.class_name = class_name # class modulation name
        self.class_index = class_index # class index wrt class list

        self.applied_transforms = []

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        """Returns the dataset metadata for the signal.

        Returns:
            DatasetMetadata: The dataset metadata.
        """
        return self._dataset_metadata

    @property
    def sample_rate(self) -> float:
        """Signal sample rate

        Returns:
            float: sample rate
        """
        return self._dataset_metadata.sample_rate

    @property
    def num_samples(self) -> int:
        """Signal number of IQ samples

        Returns:
            int: number of IQ samples
        """        
        return self.duration_in_samples

    @property
    def start(self) -> float:
        """Signal start normalized to duration of signal

        Returns signal start as a percentage of total time, ex: start=0.5 means
        the signal starts 50% of the way into the dataset IQ samples.

        Returns:
            float: signal start
        
        """
        return self.start_in_samples/self._dataset_metadata.num_iq_samples_dataset

    @start.setter
    def start(self, new_start: float):
        """Sets signal start

        Sets signal start as a percentage of total time, ex: start=0.5 means
        the signal starts 50% of the way into the dataset IQ samples.

        Args:
            new_start (float): The starting location as a percentage from 0.0 to 1.0.
        """
        self.start_in_samples = int(new_start * self._dataset_metadata.num_iq_samples_dataset)

    @property
    def stop(self) -> float:
        """Signal stop normalized to duration of signal

        Returns signal stop as a percentage of total time, ex: stop=0.5 means
        the signal stops 50% of the way into the dataset IQ samples.

        Returns:
            float: signal stop
        
        """
        return self.stop_in_samples/self._dataset_metadata.num_iq_samples_dataset

    @stop.setter
    def stop(self, new_stop: float):
        """Sets signal stop

        Sets signal stop as a percentage of total time, ex: stop=0.5 means
        the signal stops 50% of the way into the dataset IQ samples.

        Args:
            new_stop (float): The stopping location as a percentage from 0.0 to 1.0.
        """
        self.duration_in_samples = (new_stop * self._dataset_metadata.num_iq_samples_dataset) - self.start_in_samples

    @property
    def duration(self) -> float:
        """Signal duration (normalized)

        Returns signal duration normalized from 0.0 to 1.0

        Returns:
            float: signal duration
        """    
        return self.duration_in_samples/self._dataset_metadata.num_iq_samples_dataset

    @duration.setter
    def duration(self, new_duration: float):
        """Sets the duration of the signal based on a percentage of total time.

        Args:
            new_duration (float): The new duration as a percentage of total time.
        """
        self.duration_in_samples = new_duration * self._dataset_metadata.num_iq_samples_dataset

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
        return upper_freq_from_center_freq_bandwidth(self.center_freq,self.bandwidth)

    @upper_freq.setter
    def upper_freq(self, new_upper_freq: float):
        """Sets the upper frequency of the signal

        Sets the upper frequency and then updates the center frequency and bandwidth
        as they are directly related to the parameter.

        Args:
            new_upper_freq (float): The new upper frequency value
        """
        self.center_freq = center_freq_from_lower_upper_freq(new_upper_freq,self.lower_freq)
        self.bandwidth = bandwidth_from_lower_upper_freq(new_upper_freq,self.lower_freq)

    @property
    def lower_freq(self) -> float:
        """Calculates the lower frequency of a signal

        Calculates the lower frequency edge, or lowest frequency, associated with
        the bandwidth of the signal.
        
        Returns:
            float: lower frequency
        
        """
        return lower_freq_from_center_freq_bandwidth(self.center_freq,self.bandwidth)

    @lower_freq.setter
    def lower_freq(self, new_lower_freq: float):
        """Sets the lower frequency of the signal

        Sets the lower frequency and then updates the center frequency and bandwidth
        as they are directly related to the parameter.

        Args:
            new_lower_freq (float): The new lower frequency value
        """        
        self.center_freq = center_freq_from_lower_upper_freq(self.upper_freq,new_lower_freq)
        self.bandwidth = bandwidth_from_lower_upper_freq(self.upper_freq,new_lower_freq)

    @property
    def oversampling_rate(self) -> float:
        """Calculates the oversampling rate for a signal

        Calculates the oversampling rate for a signal. If a signal's bandwidth
        is 1/2 the sampling rate, the oversampling rate is 2.

        Returns:
            float: oversampling rate
        
        """
        return self.sample_rate/self.bandwidth

    @property
    def samples_per_baud(self) -> float:
        """Calculates the samples per baud for a signal

        Calculates the samples per baud for some signals. Samples per baud
        is not universal but is accurate for QAM and PSK modulation familes.
        If the signal's bandwidth is 1/2 the sampling rate then the 
        samples per baud is 2.

        Returns:
            float: samples per baud
        
        """
        return self.oversampling_rate


    def to_dict(self) -> dict:
        """Returns SignalMetadata as a full dictionary
        """
        return {
            'center_freq':self.center_freq,
            'bandwidth':self.bandwidth,
            'start_in_samples':self.start_in_samples,
            'duration_in_samples':self.duration_in_samples,
            'snr_db':self.snr_db,
            'class_name':self.class_name,
            'class_index':self.class_index,
            'sample_rate':self.sample_rate,
            'num_samples':self.num_samples,
            'start':self.start,
            'stop':self.stop,
            'duration':self.duration,
            'stop_in_samples':self.stop_in_samples,
            'upper_freq':self.upper_freq,
            'lower_freq':self.lower_freq,
            'oversampling_rate':self.oversampling_rate,
            'samples_per_baud':self.samples_per_baud,
        }

    def deepcopy(self) -> SignalMetadata:
        """Returns a deep copy of itself

        Returns:
            SignalMetadata: Deep copy of SignalMetadata
        """        
        return copy.deepcopy(self)


    def verify(self) -> None:
        """Verifies Signal Metadata fields

        Raises:
            MissingSignalMetadata: Metadata missing.
            InvalidSignalMetadata: Metadata invalid.
        """

        if self._dataset_metadata is None:
            raise ValueError("dataset_metadata is None.")

        self.center_freq = verify_float(
            self.center_freq,
            name = "center_freq",
            low = self._dataset_metadata.signal_center_freq_min,
            high = self._dataset_metadata.signal_center_freq_max
        )

        self.bandwidth = verify_float(
            self.bandwidth,
            name = "bandwidth",
            low = 0.0,
            high = self._dataset_metadata.sample_rate,
            exclude_low = True
        )

        self.start_in_samples = verify_int(
            self.start_in_samples,
            name = "start_in_samples",
            low = 0,
            high = self._dataset_metadata.num_iq_samples_dataset,
            exclude_high = True
        )

        self.duration_in_samples = verify_int(
            self.duration_in_samples,
            name = "duration_in_samples",
            low = 0,
            high = self._dataset_metadata.num_iq_samples_dataset,
            exclude_low = True
        )

        self.snr_db = verify_float(
            self.snr_db,
            name = "snr_db",
            low = 0.0,
        )

        self.class_name = verify_str(
            self.class_name,
            name = "class_name",
        )

        self.class_index = verify_int(
            self.class_index,
            name = "class_index",
            low = 0,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(center_freq={self.center_freq}, bandwidth={self.bandwidth}, start_in_samples={self.start_in_samples}, duration_in_samples={self.duration_in_samples}, snr_db={self.snr_db}, class_name={self.class_name}, class_index={self.class_index})"



### Signal
class Signal():
    """Initializes the Signal with data and metadata.

        Args:
            data (np.ndarray, optional): Signal IQ data. Defaults to np.array([]).
            metadata (SignalMetadata, optional): Signal metadata. Defaults to an empty instance of SignalMetadata().
        """
    def __init__(self, data: np.ndarray = np.array([]), metadata: SignalMetadata = None):
        """Initializes the Signal with data and metadata.

        Args:
            data (np.ndarray, optional): Signal IQ data. Defaults to np.array([]).
            metadata (SignalMetadata, optional): Signal metadata. Defaults to an empty instance of SignalMetadata().
        """
        self.data = data
        self.metadata = metadata

    def verify(self):
        """Verifies data and metadata are valid.

        Raises:
            ValueError: Data or metadata is invalid.
        """
        # convert lists to array
        self.data = verify_numpy_array(
            self.data,
            name = "IQ data",
            exact_length=self.metadata.duration_in_samples,
            data_type=torchsig_complex_data_type
        )

        self.metadata.verify()

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, metadata={self.metadata})"

## Dataset Signal Types

class DatasetSignal():
    """DatasetSignal class. Represents a signal within a dataset with metadata.

    Attributes:
        data (np.ndarray): The IQ data of the signal.
        metadata (List[SignalMetadata]): The metadata associated with the signal.

    Args:
        data (np.ndarray, optional): The IQ data for the signal. Defaults to np.array([]).
        signals (List[Signal] | Signal | List[SignalMetadata] | SignalMetadata | List[Dict[str, Any]], optional): The list of signals or metadata objects associated with the dataset signal.
        dataset_metadata (DatasetMetadata, optional): The dataset metadata. Defaults to None.
    """
    def __init__(
        self, 
        data: np.ndarray = np.array([]), 
        signals: List[Signal] | Signal | List[SignalMetadata] | SignalMetadata | List[Dict[str, Any]] = None,
        dataset_metadata: DatasetMetadata = None
    ):
        self.data = data
        self.metadata = []
        
        if isinstance(signals, (Signal, SignalMetadata)):
            signals = [signals]

        for s in signals:
            if isinstance(s, Signal):
                self.metadata.append(s.metadata)
            elif isinstance(s, SignalMetadata):
                self.metadata.append(s)
            elif isinstance(s, dict):
                if dataset_metadata is None:
                    raise ValueError("dataset_metadata required if signals = list of dicts.")
                self.metadata.append(SignalMetadata(
                    dataset_metadata = dataset_metadata,
                    center_freq = s['center_freq'],
                    bandwidth = s['bandwidth'],
                    start_in_samples = s['start_in_samples'],
                    duration_in_samples = s['duration_in_samples'],
                    snr_db = s['snr_db'],
                    class_name = s['class_name'],
                    class_index = s['class_index']
                ))
            else:
                raise ValueError('Metadata type ' + str(type(s)) + ' not supported, metadata = ' + str(s))

    def verify(self):
        """Verifies data and metadata are valid.

        Raises:
            ValueError: Data or metadata is invalid.
        """
        for m in self.metadata:
            m.verify()

        self.data = verify_numpy_array(
            self.data,
            name = "data",
            exact_length = self.metadata[0].dataset_metadata.num_iq_samples_dataset,
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, metadata={self.metadata})"


class DatasetDict():
    """DatasetDict class. Represents a dictionary containing signal data and metadata.

    Attributes:
        data (np.ndarray): The IQ data of the signal.
        metadata (List[dict]): The list of metadata dictionaries associated with the signal.
        index (int, optional): The index of the signal in the dataset. Defaults to None.

    Args:
        signal (DatasetSignal): The DatasetSignal instance to extract data and metadata from.
    """
    def __init__(self, signal: DatasetSignal):
        self.data: np.ndarray = signal.data
        self.metadata: List[dict] = []

        for m in signal.metadata:
            self.metadata.append(m.to_dict())
    
    def verify(self):
        """Verifies data and metadata are valid.

        Raises:
            ValueError: Data or metadata is invalid.
        """
        self.data = verify_numpy_array(
            self.data,
            name = "data",
        )

        for i,m in enumerate(self.metadata):
            m = verify_dict(
                m,
                name = f"metadata[{i}]",
                required_keys = keys_types_list[0],
                required_types = keys_types_list[1]
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, metadata={self.metadata})"

