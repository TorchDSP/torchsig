"""Dataset Metadata class
"""

from __future__ import annotations

# TorchSig
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.verify import (
    verify_int,
    verify_float,
    verify_distribution_list,
    verify_list
)
from torchsig.utils.printing import (
    dataset_metadata_str,
)

# Third Party
import numpy as np
from typing import Dict, List, Any
import yaml

# Built-In
from copy import deepcopy, copy

def load_dataset_metadata(filepath):
    """loads and returns the metadata object stores at filepath"""
    with open(filepath, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    return DatasetMetadata(**loaded_yaml['required']).update_from(loaded_yaml['overrides'])

class DatasetMetadata():
    """Dataset Metadata base class. Contains useful information about the dataset.

    Maintains the metadata for the parameters of the datasets, such as
    sample rate. The class holds all of the high level information
    about the dataset that the signals, impairments and other
    processes will require. Parameters that are common to all signals
    will be stored in the dataset metadata. For example, all signal
    generation requires a common and consistent sampling rate reference.

    This class is needed needed at almost every level of the DSP, therefore
    rather than pass around multiple variables, or a dict, or use globals, this
    class is defined and passed as a parameter.

    This class stores metadata related to the dataset, including parameters
    related to signal generation, transforms, dataset path, and sample distribution.
    It also handles the verification of dataset settings and ensures that the configuration 
    is valid for the dataset creation process.
    """

    minimum_params: List[str] = [
        'num_iq_samples_dataset',
        'fft_size',
        'num_signals_max'
    ]

    def __init__(
        self, 
        num_iq_samples_dataset: int, 
        fft_size: int,
        num_signals_min: int = 1,
        num_signals_max: int = 1,
        sample_rate: float = 10e6,
        num_signals_distribution: np.ndarray | List[float]= None,
        snr_db_min: float = 0.0,
        snr_db_max: float = 50.0,
        signal_duration_min: float = None,
        signal_duration_max: float = None,
        signal_bandwidth_min: float = None,
        signal_bandwidth_max: float = None,
        signal_center_freq_min: float = None,
        signal_center_freq_max: float = None,
        cochannel_overlap_probability: float = 0.1,
        class_list: List[str] = None,
        class_distribution: np.ndarray | List[float]= None,
        **kwargs
    ):
        """Initializes Dataset Metadata

        Args:
            num_iq_samples_dataset (int): Length of I/Q array in dataset.
            fft_size (int): Size of FFT (number of bins) to be used in spectrogram.
            sample_rate (float, optional): Sample rate for dataset. Defaults to 10e6.
            num_signals_min (int, optional): Minimum number of signals per sample. Defaults to 0.
            num_signals_max (int): Maximum number of signals per sample in dataset.
            num_signals_distribution (np.ndarray | List[float], optional): Probability to generate sample with N signals 
                for each value in `[num_signals_min, num_signals_max]`. Defaults to None (uniform).
            snr_db_min (float, optional): Minimum SNR of signals to generate. Defaults to 0.0.
            snr_db_max (float, optional): Maximum SNR of signals to generate. Defaults to 50.0.
            signal_duration_min (float, optional): Minimum duration of signal. Defaults to None.
            signal_duration_max (float, optional): Maximum duration of signal. Defaults to None.
            signal_bandwidth_min (float, optional): Minimum bandwidth of the signal. Defaults to None.
            signal_bandwidth_max (float, optional): Maximum bandwidth of the signal. Defaults to None.
            signal_center_freq_min (float, optional): Minimum center frequency of the signal. Defaults to None.
            signal_center_freq_max (float, optional): Maximum center frequency of the signal. Defaults to None.
            cochannel_overlap_probability (float, optional): Probability to allow co-channel interference per signal. (default is 0.1).
            transforms (list): Transforms to apply. Defaults to [].
            target_transforms (list): List of Target Transforms to apply. Defaults to [].
            class_list (List[str], optional): Signal class name list. Defaults to TorchSigSignalLists.all_signals.
            class_distribution (np.ndarray | List[float], optional): Probabilities for each class in `class_list`. Defaults to None (uniform).

        Raises:
            ValueError: If any of the provided parameters are invalid or incompatible.
        """    
        #super().__init__(**kwargs)
        self._kwargs = kwargs
        
        self.num_iq_samples_dataset = num_iq_samples_dataset
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.fft_stride = copy(self.fft_size)
        self.num_signals_max = num_signals_max
        self.num_signals_min = self.num_signals_max if num_signals_min is None else num_signals_min
        self.num_signals_range = np.arange(start=self.num_signals_min, stop=num_signals_max + 1, dtype=int)
        self.num_signals_distribution = num_signals_distribution

        self.class_list = TorchSigSignalLists.all_signals if class_list is None else class_list
        self.class_distribution = class_distribution

        self.snr_db_max = snr_db_max
        self.snr_db_min = snr_db_min

        self.signal_duration_min = signal_duration_min if signal_duration_min is not None else 0.10*self.num_iq_samples_dataset/sample_rate
        self.signal_duration_max = signal_duration_max if signal_duration_max is not None else 0.20*self.num_iq_samples_dataset/sample_rate

        self.signal_bandwidth_min = signal_bandwidth_min if signal_bandwidth_min is not None else self.sample_rate/20
        self.signal_bandwidth_max = signal_bandwidth_max if signal_bandwidth_max is not None else self.sample_rate/10

        self.signal_center_freq_min = signal_center_freq_min if signal_center_freq_min is not None else self.frequency_min
        self.signal_center_freq_max = signal_center_freq_max if signal_center_freq_max is not None else self.frequency_max

        self.cochannel_overlap_probability = cochannel_overlap_probability

        # provide a noise power reference in dB
        self.noise_power_db = 0

        # run _verify() to ensure metadata is valid
        self.verify()

    def update_from(self, attr_dict):
        """updates the fields of this metadata object with the values in attr_dict; good for joining metadata together
            modifies existing object, and return without copying
        """
        for key in attr_dict.keys():
            setattr(self, key, attr_dict[key])
        return self

    def verify(self) -> None:
        """Verify that metadata is valid.

        This method checks the configuration of the metadata, ensuring all parameters
        are consistent, valid, and appropriate for dataset creation. It will raise 
        ValueError if any configuration is found to be incorrect.

        Raises:
            ValueError: If any dataset configuration is invalid.
        """

        self.class_distribution = verify_distribution_list(
            self.class_distribution, 
            len(self.class_list), 
            "class_distribution", 
            "class_list"
        )
        self.num_signals_distribution = verify_distribution_list(
            self.num_signals_distribution, 
            len(self.num_signals_range), 
            "num_signals_distribution", 
            "[num_signals_min, num_signals_max]"
        )

        self.class_list = verify_list(self.class_list, "class_list")

        # check all of the input parameters
        self.sample_rate = verify_float(
            self.sample_rate,
            name = "sample_rate",
            low = 0.0,
            exclude_low = True
        )

        self.fft_size = verify_int(
            self.fft_size,
            name = "fft_size",
            low = 0,
            exclude_low = True
        )

        self.fft_stride = verify_int(
            self.fft_stride,
            name = "fft_stride",
            low = 0,
            high = self.fft_size,
            exclude_low = True,
        )

        
        self.num_iq_samples_dataset = verify_int(
            self.num_iq_samples_dataset,
            name = "num_iq_samples_dataset",
            low = 0,
            exclude_low = True
        )

        self.num_signals_max = verify_int(
            self.num_signals_max,
            name = "num_signals_max",
            low = 0
        )

        self.num_signals_min = verify_int(
            self.num_signals_min,
            name = "num_signals_min",
            low = 0,
            high = self.num_signals_max
        )

        self.snr_db_max = verify_float(
            self.snr_db_max,
            name = "snr_db_max",
            low = None,
        )

        self.snr_db_min = verify_float(
            self.snr_db_min,
            name = "snr_db_min",
            low = None,
            high = self.snr_db_max
        )

        self.signal_duration_max = verify_float(
            self.signal_duration_max,
            name = "signal_duration_max",
            low = self.dataset_duration_min,
            high = self.dataset_duration_max
        )

        self.signal_duration_min = verify_float(
            self.signal_duration_min,
            name = "signal_duration_min",
            low = self.dataset_duration_min,
            high = self.dataset_duration_max
        )

        self.signal_bandwidth_min = verify_float(
            self.signal_bandwidth_min,
            name = "signal_bandwidth_min",
            low = self.dataset_bandwidth_min,
            high = self.dataset_bandwidth_max
        )

        self.signal_bandwidth_max = verify_float(
            self.signal_bandwidth_max,
            name = "signal_bandwidth_max",
            low = self.dataset_bandwidth_min,
            high = self.dataset_bandwidth_max
        )

        self.signal_center_freq_min = verify_float(
            self.signal_center_freq_min,
            name = "signal_center_freq_min",
            low = self.dataset_center_freq_min,
            high = self.dataset_center_freq_max
        )

        self.signal_center_freq_max = verify_float(
            self.signal_center_freq_max,
            name = "signal_center_freq_max",
            low = self.dataset_center_freq_min,
            high = self.dataset_center_freq_max
        )

        self.cochannel_overlap_probability = verify_float(
            self.cochannel_overlap_probability,
            name = "cochannel_overlap_probability",
            low = 0,
            high = 1
        )

        # check derived values
        
        verify_float(
            self.fft_frequency_resolution,
            name = "fft_frequency_resolution",
            low = 0.0,
            exclude_low = True
        )


        verify_float(
            self.fft_frequency_max,
            name = "fft_frequency_max",
            low = None,
            high = None,
        )

        verify_float(
            self.fft_frequency_min,
            name = "fft_frequency_max",
            low = None,
            high = self.fft_frequency_max,
            exclude_high = True
        )

        verify_int(
            self.signal_duration_in_samples_max,
            name = "signal_duration_in_samples_max",
            low = self.dataset_duration_in_samples_min,
            exclude_low = True
        )

        verify_int(
            self.signal_duration_in_samples_min,
            name = "signal_duration_in_samples_min",
            low = 0,
            high = self.dataset_duration_in_samples_max,
            exclude_low = True
        )

    def __str__(self) -> str:
        return dataset_metadata_str(self)

    def __repr__(self) -> str:
        """Returns a string representation of the DatasetMetadata instance.
        
        This provides a concise summary of the key parameters such as `num_iq_samples_dataset`, 
        `sample_rate`, `num_signals_max`, and `fft_size`.
        
        Returns:
            str: String representation of the DatasetMetadata instance.
        """
        return f"{self.__class__.__name__}(num_iq_samples_dataset={self.num_iq_samples_dataset}, sample_rate={self.sample_rate}, num_signals_max={self.num_signals_max}, fft_size={self.fft_size})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataset metadata into a dictionary format.

        This method organizes various metadata fields related to the dataset into categories such as 
        general dataset information, signal generation parameters, and dataset writing information.

        Returns:
            Dict[str, Any]: A dictionary representation of the dataset metadata.
        """
        # organize fields by area

        required = {
            'num_iq_samples_dataset': self.num_iq_samples_dataset,
            'fft_size': self.fft_size,
        }

        overrides = {
            'sample_rate': self.sample_rate,
            'num_signals_min': self.num_signals_min,
            'num_signals_max': self.num_signals_max,
            'num_signals_distribution': "uniform" if self.num_signals_distribution is None else self.num_signals_distribution.tolist(),
            'snr_db_min': self.snr_db_min,
            'snr_db_max': self.snr_db_max,
            'signal_duration_min': self.signal_duration_min,
            'signal_duration_max': self.signal_duration_max,
            'signal_bandwidth_min': self.signal_bandwidth_min,
            'signal_bandwidth_max': self.signal_bandwidth_max,
            'signal_center_freq_min': self.signal_center_freq_min,
            'signal_center_freq_max': self.signal_center_freq_max,
            'cochannel_overlap_probability': self.cochannel_overlap_probability,
            'class_list': deepcopy(self.class_list),
            'class_distribution': "uniform" if self.class_distribution is None else self.class_distribution.tolist(),
        }

        # dataset information
        dataset_info = {
            'num_iq_samples_dataset': self.num_iq_samples_dataset,
            'fft_size': self.fft_size,
            'sample_rate': self.sample_rate,
        }
        # signal generation
        signal_gen = {
            'num_signals_min': self.num_signals_min,
            'num_signals_max': self.num_signals_max,
            'num_signals_range': self.num_signals_range.tolist(),
            'num_signals_distribution': "uniform" if self.num_signals_distribution is None else self.num_signals_distribution.tolist(),
            'snr_db_min': self.snr_db_min,
            'snr_db_max': self.snr_db_max,
            'signal_duration_min': self.signal_duration_min,
            'signal_duration_max': self.signal_duration_max,
            'signal_bandwidth_min': self.signal_bandwidth_min,
            'signal_bandwidth_max': self.signal_bandwidth_max,
            'signal_center_freq_min': self.signal_center_freq_min,
            'signal_center_freq_max': self.signal_center_freq_max,
            'cochannel_overlap_probability': self.cochannel_overlap_probability,
            'fft_size': self.fft_size,
            'fft_frequency_resolution': self.fft_frequency_resolution,
            'fft_frequency_min': self.fft_frequency_min,
            'fft_frequency_max': self.fft_frequency_max,
            'class_list': deepcopy(self.class_list),
            'class_distribution': "uniform" if self.class_distribution is None else self.class_distribution.tolist(),
            'signal_duration_in_samples_min': self.signal_duration_in_samples_min,
            'signal_duration_in_samples_max': self.signal_duration_in_samples_max,
        }
               

        read_only = {
            'info': dataset_info,
            'signals': signal_gen,
        }


        return {
            'required': required,
            'overrides': overrides,
            'read_only': read_only,
        }

    @property
    def dataset_center_freq_max(self) -> float:
        """The maximum center frequency for a signal

        The maximum is a boundary condition such that the center frequency
        will not alias across the upper sampling rate boundary.

        The calculation includes a small epsilon such that the center_freq_max
        is never equal to sample_rate/2 to avoid the aliasing condition because
        -sample_rate/2 is equivalent to sample_rate/2.

        Returns:
            float: maximum center frequency boundary for signal
        """
        epsilon = 1e-10
        return (self.sample_rate/2)*(1-epsilon)

    @property
    def dataset_center_freq_min(self) -> float:
        """The minimum center frequency for a signal

        The minimum is a boundary condition such that the center frequency
        will not alias across the lower sampling rate boundary.

        Returns:
            float: minimum center frequency boundary for signal
        """
        return -self.sample_rate/2

    @property
    def dataset_duration_max(self) -> float:
        """The maximum duration possible within the dataset

        The maximum is a boundary condition such that the signal duration
        will not exceed the total time duration of the dataset.

        Returns:
            float: maximum duration for a signal
        """
        return self.num_iq_samples_dataset/self.sample_rate

    @property
    def dataset_duration_min(self) -> float:
        """The minimum duration possible within the dataset

        The minimum is a boundary condition such that the signal duration
        will not be less than a specified minimum.

        Returns:
            float: minimum duration for a signal
        """
        return (self.fft_size/16)*(1/self.sample_rate)

    @property
    def dataset_duration_in_samples_max(self) -> float:
        """The maximum duration in samples possible within the dataset

        The maximum is a boundary condition such that the signal duration
        in number of samples will not exceed the total number of samples
        within the dataset.

        Returns:
            float: maximum duration for a signal in number of samples
        """
        return int(self.dataset_duration_max*self.sample_rate)

    @property
    def dataset_duration_in_samples_min(self) -> float:
        """The minimum duration in samples possible within the dataset

        The minimum is a boundary condition such that the signal duration
        in number of samples will not exceed the total number of samples
        within the dataset.

        Returns:
            float: minimum duration for a signal in number of samples
        """
        return int(self.dataset_duration_min*self.sample_rate)

    @property
    def dataset_bandwidth_min(self) -> float:
        """The minimum possible bandwidth for a signal

        Provides a boundary for the minimum bandwidth of a signal, which
        is the bandwidth of a tone, which is sample rate / number of samples.

        Returns:
            float: the minimum bandwidth for a signal
        """
        return self.sample_rate / self.num_iq_samples_dataset

    @property
    def dataset_bandwidth_max(self) -> float:
        """The maximum possible bandwidth for a signal

        Provides a boundary for the maximum bandwidth of a signal, which
        is the sampling rate.

        Returns:
            float: the maximum bandwidth for a signal
        """
        return self.sample_rate

    @property
    def signal_duration_in_samples_max(self) -> int:
        """The maximum duration in samples for a signal

        Provides a maximum duration for a signal in number of samples.

        Returns:
            float: the maximum duration in samples for a signal
        """
        return int(self.signal_duration_max*self.sample_rate)

    @property
    def signal_duration_in_samples_min(self) -> int:
        """The minimum duration in samples for a signal

        Provides a minimum duration for a signal in number of samples.

        Returns:
            float: the minimum duration in samples for a signal
        """
        return int(self.signal_duration_min*self.sample_rate)


    ## Derived Read-Only Dataset Metadata


    @property
    def fft_frequency_resolution(self) -> float:
        """Frequency resolution of the spectrogram

        The frequency resolution, or resolution bandwidth, of the FFT.

        Returns:
            float: frequency resolution
        """
        return self.sample_rate/self.fft_size

    @property
    def fft_frequency_min(self) -> float:
        """The minimum frequency associated with the FFT

        Defines the smallest frequency within the FFT of the spectrogram.
        The FFT has discrete bins and therefore each bin has an associated
        frequency. This frequency is associated with the 0th bin or left-most
        frequency bin. 

        Returns:
            float: minimum FFT frequency
        """
        return -self.sample_rate/2

    @property
    def fft_frequency_max(self) -> float:
        """The maximum frequency associated with the FFT

        Defines the largest frequency within the FFT of the spectrogram.
        The FFT has discrete bins and therefore each bin has an associated
        frequency. This frequency is associated with the N-1'th bin or 
        right-most frequency bin. 

        Returns:
            float: maximum FFT frequency
        """
        return self.fft_frequency_min + ((self.fft_size-1) * self.fft_frequency_resolution)


    @property
    def frequency_min(self) -> float:
        """Minimum representable frequency

        Boundary edge for testing the lower Nyquist sampling boundary.

        Returns:
            float: minimum frequency
        """
        return -self.sample_rate/2


    @property
    def frequency_max(self) -> float:
        """Maximum representable frequency

        Boundary edge for testing the upper Nyquist sampling boundary.
        Due to the circular nature of the frequency domain, both -fs/2
        and fs/2 represent the boundary, therefore an epsilon value is
        used to back off the upper edge slightly.

        Returns:
            float: maximum frequency
        """
        epsilon = 1e-10
        return (self.sample_rate/2)*(1-epsilon)


class ExternalDatasetMetadata():
    """Dataset Metadata class for external data, with less required infrastructure
    and fields than the internal metadata class that generates TorchSig datasets.
    """

    minimum_params: List[str] = [
        'num_iq_samples_dataset',
        'class_list',
        'sample_rate'
    ]

    def __init__(
        self, 
        num_iq_samples_dataset: int,
        class_list: List[str] = [],
        sample_rate: float = 10e6,
        **kwargs
    ):
        """Initializes ExternalDatasetMetadata.

        Args:
            num_iq_samples_dataset (int): Length of I/Q array in dataset.
            class_list (List[str], optional): Signal class name list. Defaults to []].
            sample_rate (float, optional): Sample rate for dataset. Defaults to 10e6.
        Raises:
            ValueError: If any of the provided parameters are invalid or incompatible.
        """            
        self.num_iq_samples_dataset = num_iq_samples_dataset
        self.sample_rate = sample_rate
        self.class_list = class_list
        self.kwargs = kwargs

        # run _verify() to ensure metadata is valid
        self.verify()

    def verify(self) -> None:
        """Verify that metadata is valid.

        Raises:
            ValueError: If any dataset configuration is invalid.
        """

        # check all of the input parameters
        self.num_iq_samples_dataset = verify_int(
            self.num_iq_samples_dataset,
            name = "num_iq_samples_dataset",
            low = 0,
            exclude_low = True
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Returns a string representation of the DatasetMetadata instance.
        
        This provides a concise summary of the key parameters such as `num_iq_samples_dataset`, 
        `sample_rate`, `num_signals_max`, and `fft_size`.
        
        Returns:
            str: String representation of the DatasetMetadata instance.
        """
        return f"{self.__class__.__name__}(num_iq_samples_dataset={self.num_iq_samples_dataset}, sample_rate={self.sample_rate})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataset metadata into a dictionary format.

        This method organizes various metadata fields related to the dataset into categories such as 
        general dataset information, signal generation parameters, and dataset writing information.

        Returns:
            Dict[str, Any]: A dictionary representation of the dataset metadata.
        """
        # organize fields by area

        required = {
            'num_iq_samples_dataset': self.num_iq_samples_dataset,

        }

        overrides = {
            'sample_rate': self.sample_rate,
            'class_list': deepcopy(self.class_list),
        }

        # dataset information
        dataset_info = {
            'num_iq_samples_dataset': self.num_iq_samples_dataset,
            'sample_rate': self.sample_rate,
        }

        read_only = {
            'info': dataset_info
        }

        return {
            'required': required,
            'overrides': overrides,
            'read_only': read_only
        }
