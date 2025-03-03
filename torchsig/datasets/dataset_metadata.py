"""Dataset Metadata class for Narrowband and Wideband
"""

from __future__ import annotations

# TorchSig
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.random import Seedable
from torchsig.utils.verify import (
    verify_int,
    verify_float,
    verify_str,
    verify_distribution_list,
    verify_list,
    verify_transforms,
    verify_target_transforms
)
from torchsig.utils.printing import (
    dataset_metadata_str,
    dataset_metadata_repr,
)
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.impairments_wideband import WidebandImpairments
from torchsig.transforms.impairments_narrowband import NarrowbandImpairments

# Third Party
import numpy as np
from typing import Dict, List, Any

# Built-In
from copy import (
    deepcopy,
    copy
)


class DatasetMetadata(Seedable):
    """Dataset Metdata. Contains useful information about the dataset.

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
        impairment_level: int,
        num_signals_max: int, 
        sample_rate: float = 10e6, 
        num_signals_min: int = 0,
        num_signals_distribution: np.ndarray | List[float]= None,
        snr_db_min: float = 0.0,
        snr_db_max: float = 50.0,
        signal_duration_percent_min: float = 0.0,
        signal_duration_percent_max: float = 100.0,
        transforms: list = [],
        target_transforms: list = [],
        class_list: List[str] = None,
        class_distribution: np.ndarray | List[float]= None,
        num_samples: int = None,
        dataset_type: str = "None",
        **kwargs
    ):
        """Initializes Dataset Metadata

        Args:
            num_iq_samples_dataset (int): Length of I/Q array in dataset.
            sample_rate (float): Sample rate for dataset.
            fft_size (int): Size of FFT (number of bins) to be used in spectrogram.
            impairment_level (int): Signal impairment level.
            num_signals_max (int): Maximum number of signals per sample in dataset.
            num_signals_min (int, optional): Minimum number of signals per sample. Defaults to 0.
            num_signals_distribution (np.ndarray | List[float], optional): Probability to generate sample with N signals 
                for each value in `[num_signals_min, num_signals_max]`. Defaults to None (uniform).
            snr_db_min (float, optional): Minimum SNR of signals to generate. Defaults to 0.0.
            snr_db_max (float, optional): Maximum SNR of signals to generate. Defaults to 50.0.
            signal_duration_percent_min (float, optional): Minimum duration of signal in percentage. Defaults to 0.0.
            signal_duration_percent_max (float, optional): Maximum duration of signal in percentage. Defaults to 100.0.
            transforms (list): Transforms to apply. Defaults to [].
            target_transforms (list): List of Target Transforms to apply. Defaults to [].
            class_list (List[str], optional): Signal class name list. Defaults to TorchSigSignalLists.all_signals.
            class_distribution (np.ndarray | List[float], optional): Probabilities for each class in `class_list`. Defaults to None (uniform).
            num_samples (int, optional): Set dataset size. For infinite dataset, set to None, Defaults to None.
            dataset_type (str, optional): Dataset type name. Defaults to "None".

        Raises:
            ValueError: If any of the provided parameters are invalid or incompatible.
        """    
        super().__init__(**kwargs)
        self._num_iq_samples_dataset = num_iq_samples_dataset
        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._fft_stride = copy(self._fft_size)
        self._num_signals_max = num_signals_max
        self._num_signals_min = self._num_signals_max if num_signals_min is None else num_signals_min
        self._num_signals_range = np.arange(start=self._num_signals_min, stop=num_signals_max + 1, dtype=int)
        self._num_signals_distribution = num_signals_distribution
        self._transforms = transforms
        self._target_transforms = target_transforms
        self._impairment_level = impairment_level
        self._impairments = self._initialize_impairments() # will be updated in subclasses
        self._impairments.add_parent(self)

        self._class_list = TorchSigSignalLists.all_signals if class_list is None else class_list
        self._class_distribution = class_distribution

        self._dataset_type = dataset_type
        
        
        self._num_samples = num_samples

        # provide a noise power reference in dB
        self._noise_power_db = 0
        # minimum and maximum SNRs for signals
        self._snr_db_max = snr_db_max
        self._snr_db_min = snr_db_min

        self._signal_duration_percent_max = signal_duration_percent_max
        self._signal_duration_percent_min = signal_duration_percent_min

        # run _verify() to ensure metadata is valid
        self._verify()

    def _verify(self) -> None:
        """Verify that metadata is valid.

        This method checks the configuration of the metadata, ensuring all parameters
        are consistent, valid, and appropriate for dataset creation. It will raise 
        ValueError if any configuration is found to be incorrect.

        Raises:
            ValueError: If any dataset configuration is invalid.
        """

        # check dataset type
        self._dataset_type = verify_str(
            s = self._dataset_type,
            name = "dataset_type",
            valid = ["narrowband", "wideband"],
            str_format = "lower"
        )

        # Transforms
        self._transforms = verify_transforms(self._transforms)
        for transform in self._transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)

        # Target Transforms
        self._target_transforms = verify_target_transforms(self._target_transforms)
        for transform in self._target_transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)

        self._class_distribution = verify_distribution_list(
            self._class_distribution, 
            len(self.class_list), 
            "class_distribution", 
            "class_list"
        )
        self._num_signals_distribution = verify_distribution_list(
            self._num_signals_distribution, 
            len(self._num_signals_range), 
            "num_signals_distribution", 
            "[num_signals_min, num_signals_max]"
        )

        self._class_list = verify_list(self._class_list, "class_list")

        # check all of the input parameters
        self._sample_rate = verify_float(
            self._sample_rate,
            name = "sample_rate",
            low = 0.0,
            exclude_low = True
        )
        
        self._impairment_level = verify_int(
            self._impairment_level,
            name = "impairment_level",
            low = 0,
            high = 2
        )

        self._num_iq_samples_dataset = verify_int(
            self._num_iq_samples_dataset,
            name = "num_iq_samples_dataset",
            low = 0,
            exclude_low = True
        )

        self._fft_size = verify_int(
            self._fft_size,
            name = "fft_size",
            low = 0,
            exclude_low = True
        )

        self._fft_stride = verify_int(
            self._fft_stride,
            name = "fft_stride",
            low = 0,
            high = self._fft_size,
            exclude_low = True,
        )

        self._num_signals_max = verify_int(
            self._num_signals_max,
            name = "num_signals_max",
            low = 0
        )

        self._num_signals_min = verify_int(
            self._num_signals_min,
            name = "num_signals_min",
            low = 0,
            high = self._num_signals_max
        )

        self._snr_db_max = verify_float(
            self._snr_db_max,
            name = "snr_db_max",
            low = None,
        )

        self._snr_db_min = verify_float(
            self._snr_db_min,
            name = "snr_db_min",
            low = None,
            high = self._snr_db_max
        )

        self._signal_duration_percent_max = verify_float(
            self._signal_duration_percent_max,
            name = "signal_duration_percent_max",
            low = 0.0,
            high = 100.0
        )

        self._signal_duration_percent_min = verify_float(
            self._signal_duration_percent_min,
            name = "signal_duration_percent_min",
            low = 0.0,
            high = self._signal_duration_percent_max
        )

        if self._num_samples is not None:
            self._num_samples = verify_int(
                self._num_samples,
                name = "num_samples",
                low = 0,
                exclude_low = True
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
            self.duration_in_samples_max,
            name = "duration_in_samples_max",
            low = 0,
            exclude_low = True
        )

        verify_int(
            self.duration_in_samples_min,
            name = "duration_in_samples_min",
            low = 0,
            high = self.duration_in_samples_max,
            exclude_low = True
        )


    def _initialize_impairments(self) -> Impairments:
        """Initializes the dataset impairments (not implemented).

        This method is intended to be implemented by subclasses to initialize dataset-specific impairments.
        For now, it raises a NotImplementedError since the actual impairment logic will be defined in subclasses.

        Returns:
            Impairments: The initialized impairments object.
        
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return dataset_metadata_str(self)

    def __repr__(self) -> str:
        return dataset_metadata_repr(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataset metadata into a dictionary format.

        This method organizes various metadata fields related to the dataset into categories such as 
        general dataset information, signal generation parameters, and dataset writing information.

        Returns:
            Dict[str, Any]: A dictionary representation of the dataset metadata.
        """
        # organize fields by area

        required = {
            'dataset_type': self._dataset_type,
            'num_iq_samples_dataset': self._num_iq_samples_dataset,
            'impairment_level': self._impairment_level,
            'fft_size': self._fft_size
        }

        

        overrides = {
            'num_samples': self._num_samples,
            'sample_rate': self._sample_rate,
            'num_signals_min': self._num_signals_min,
            'num_signals_distribution': "uniform" if self._num_signals_distribution is None else self._num_signals_distribution.tolist(),
            'snr_db_min': self._snr_db_min,
            'snr_db_max': self.snr_db_max,
            'signal_duration_percent_min': self._signal_duration_percent_min,
            'signal_duration_percent_max': self._signal_duration_percent_max,
            'class_list': deepcopy(self._class_list),
            'class_distribution': "uniform" if self._class_distribution is None else self._class_distribution.tolist(),
            #####'seed': self.rng_seed
        }

        # dataset information
        dataset_info = {
            'dataset_type': self._dataset_type,
            'num_samples': "infinite" if self._num_samples is None else self._num_samples,
            'num_iq_samples_dataset': self._num_iq_samples_dataset,
            'fft_size': self._fft_size,
            'sample_rate': self._sample_rate,
            'impairment_level': self._impairment_level,
            #####'seed': self.rng_seed,
            'transforms': [str(tranform) for tranform in self._transforms],
            'target_transforms': [str(target_transform) for target_transform in self._target_transforms],
        }
        # signal genration
        signal_gen = {
            'num_signals_min': self._num_signals_min,
            'num_signals_max': self._num_signals_max,
            'num_signals_range': self._num_signals_range.tolist(),
            'num_signals_distribution': "uniform" if self._num_signals_distribution is None else self._num_signals_distribution.tolist(),
            'snr_db_min': self._snr_db_min,
            'snr_db_max': self._snr_db_max,
            'signal_duration_percent_min': self._signal_duration_percent_min,
            'signal_duration_percent_max': self._signal_duration_percent_max,
            'center_freq_min': self.center_freq_min,
            'center_freq_max': self.center_freq_max,
            'bandwidth_min': self.bandwidth_min,
            'bandwidth_max': self.bandwidth_max,
            'fft_size': self._fft_size,
            'fft_frequency_resolution': self.fft_frequency_resolution,
            'fft_frequency_min': self.fft_frequency_min,
            'fft_frequency_max': self.fft_frequency_max,
            'class_list': deepcopy(self._class_list),
            'class_distribution': "uniform" if self._class_distribution is None else self._class_distribution.tolist(),
            'duration_in_samples_min': self.duration_in_samples_min,
            'duration_in_samples_max': self.duration_in_samples_max,
        }
               

        read_only = {
            'info': dataset_info,
            'signals': signal_gen,
        }

        if self._dataset_type == "wideband":
            required["num_signals_max"] = self._num_signals_max

        return {
            'required': required,
            'overrides': overrides,
            'read_only': read_only,
        }

    # Functions that must be implemented by child classes. these
    # define the range for bandwidth and center frequency on a per
    # dataset basis due to differences between narrowband and
    # wideband datasets

    @property
    def center_freq_min(self) -> None:
        """Defines the minimum center frequency boundary for a signal

        Raises:
            NotImplementedError: Inherited classes must implement this method.
        """        
        raise NotImplementedError(".center_freq_min not implemented.")

    @property
    def center_freq_max(self) -> None:
        """Defines the minimum center frequency boundary for a signal

        Raises:
            NotImplementedError: Inherited classes must implement this method.
        """        
        raise NotImplementedError(".center_freq_max not implemented.")

    @property
    def bandwidth_min(self) -> None:
        """Defines the minimum bandwidth for a signal

        Raises:
            NotImplementedError: Inherited classes must implement this method.
        """        
        raise NotImplementedError(".bandwidth_min not implemented.")

    @property
    def bandwidth_max(self) -> None:
        """Defines the maximum bandwidth for a signal

        Raises:
            NotImplementedError: Inherited classes must implement this method.
        """        
        raise NotImplementedError(".bandwidth_max not implemented.")

    ## Read-Only Dataset Metadata fields
    @property
    def num_iq_samples_dataset(self) -> int:
        """Length of I/Q array per sample in dataset.

        Returns the number of IQ samples of the dataset, this is
        the length of the array that contains the IQ samples

        Returns:
            int: number of IQ samples
        """
        return self._num_iq_samples_dataset

    @property
    def sample_rate(self) -> float:
        """Sample rate for the dataset.

        Returns the sampling rate associated with the IQ samples of the dataset

        Returns:
            float: sample rate
        """
        return self._sample_rate
    
    @property
    def num_signals_max(self) -> int:
        """Max number of signals in each sample in the dataset

        Returns the number of distinct signals in the wideband dataset

        Returns:
            int: max number of signals
        """
        return self._num_signals_max

    @property
    def num_signals_min(self) -> int:
        """Minimum number of signals in each sample in the dataset.

        Returns:
            int: min number of signals
        """        
        return self._num_signals_min

    @property
    def num_signals_range(self) -> List[int]:
        """Range of num_signals can be generated by a sample.

        Returns:
            List[int]: List of num_signals possibilities.
        """        
        return self._num_signals_range

    @property
    def num_signals_distribution(self) -> List[float]:
        """Probabilities for each value in `num_signals_range`.

        Returns:
            List[float]: Probabilties sample generates N signals per sample.
        """        
        return self._num_signals_distribution

    @property
    def transforms(self) -> list:
        """Transforms to perform on signal data (after signal impairments).

        Returns:
            Transform: Transform to apply to data.
        """        
        return self._transforms

    @property
    def target_transforms(self) -> list:
        """Target Transform to apply.

        Returns:
            TargetTransform: _description_
        """        
        return self._target_transforms

    @property
    def impairment_level(self) -> int:
        """Level of signal impairments to apply to signals (0-2)

        Returns:
            int: Impairment level.
        """        
        return self._impairment_level

    @property
    def impairments(self) -> Impairments:
        """Impairment signal and dataset transforms

        Returns:
            Impairments: Transforms or impairments
        """        
        return self._impairments

    @property
    def class_list(self) -> List[str]:
        """Signal modulation class list for dataset.

        Returns:
            List[str]: List of signal modulation class names
        """        
        return self._class_list

    @property
    def class_distribution(self) -> np.ndarray | List[str]:
        """Signal modulation class distribution for dataset generation.

        Returns:
            np.ndarray | List[str]: List of class probabilites.
        """        
        return self._class_distribution
    

    @property
    def num_samples(self) -> int:
        """Getter for the number of samples in the dataset.

        This property returns the number of samples that the dataset is configured to have. If the value is set 
        to `None`, it indicates that the number of samples is considered infinite.

        Returns:
            int: The number of samples in the dataset, or a representation of infinite samples if set to `None`.
        """
        return self._num_samples

    @property
    def dataset_type(self) -> str:
        """Type of dataset.

        Returns:
            str: Dataset type name
        """        
        return self._dataset_type

    @property
    def noise_power_db(self) -> float:
        """Reference noise power (dB) for the dataset

        The noise power is a common reference to be used for all signal
        generation in order to establish accurate SNR calculations.
        The noise power dB is given in decibels. The PSD estimate of the
        AWGN is calculated such that the averaging across all frequency
        bins average to noise_power_db.

        Returns:
            float: noise power in dB
        """
        return self._noise_power_db

    @property
    def snr_db_min(self) -> float:
        """Minimum SNR in dB for signals in dataset

        Signals within the dataset will be assigned a signal to noise
        ratio (SNR), across a range defined by a minimum and maximum
        value. snr_db_min is the low end of the SNR range.

        Returns:
            float: minimum SNR in dB
        """
        return self._snr_db_min


    @property
    def snr_db_max(self) -> float:
        """Minimum SNR in dB for signals in dataset

        Signals within the dataset will be assigned a signal to noise
        ratio (SNR), across a range defined by a minimum and maximum
        value. snr_db_max is the high end of the SNR range.

        Returns:
            float: maximum SNR in dB
        """
        return self._snr_db_max

    @property
    def signal_duration_percent_max(self) -> float:
        """Getter for the maximum signal duration percentage.

        This property returns the maximum percentage of the total signal duration that is allowed or configured
        for the dataset. The duration percentage is typically used in signal generation or processing to specify 
        the maximum duration of individual signals relative to the total dataset duration.

        Returns:
            float: The maximum allowed percentage of the signal duration.
        """
        return self._signal_duration_percent_max
    
    @property
    def signal_duration_percent_min(self) -> float:
        """Getter for the minimum signal duration percentage.

        This property returns the minimum percentage of the total signal duration that is allowed or configured
        for the dataset. The duration percentage is used in signal generation or processing to specify 
        the minimum duration of individual signals relative to the total dataset duration.

        Returns:
            float: The minimum allowed percentage of the signal duration.
        """
        return self._signal_duration_percent_min

    @property
    def fft_size(self) -> int:
        """The size of FFT (number of bins) to be used in spectrogram.

        The FFT size used to compute the spectrogram for the wideband dataset.

        Returns:
            int: FFT size
        """
        return self._fft_size

    @property
    def fft_stride(self) -> int:
        """The stride of input samples in FFT (number of samples)

        The FFT stride controls the distance in samples between successive
        FFTs. A smaller FFT stride means more averaging between FFTs, a
        larger stride means less averaging between FFTs. fft_stride = fft_size
        means there is no overlap of samples between the current and next
        FFT. fft_stride = fft_size/2 means there is 50% overlap between the
        input samples of the the current and next fft.

        Returns:
            int: FFT stride
        """
        return self._fft_stride

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

    @property
    def duration_in_samples_max(self) -> int:
        """The maximum duration in samples for a signal

        Provides a maximum duration for a signal in number of samples.

        Returns:
            float: the maximum duration in samples for a signal
        """
        return int(self.num_iq_samples_dataset*self._signal_duration_percent_max/100)

    @property
    def duration_in_samples_min(self) -> int:
        """The minimum duration in samples for a signal

        Provides a minimum duration for a signal in number of samples.

        Returns:
            float: the minimum duration in samples for a signal
        """
        # translate the minimum percentage into number of samples
        duration_percentage_based = int(self.num_iq_samples_dataset*self._signal_duration_percent_min/100)

        # define secondary minimum duration
        duration_fft_based = int(np.round(self.fft_size/2))

        # select the larger of the two
        duration = np.max((duration_percentage_based,duration_fft_based))

        # convert data type
        duration = int(duration) 

        return duration



### Narrowband Metadata

class NarrowbandMetadata(DatasetMetadata):
    """Narrowband Dataset Metadata Class

    This class encapsulates the metadata for a narrowband dataset, extending the
    base `DatasetMetadata` class. It provides useful information about the dataset 
    such as the number of samples, the sample rate, the FFT size, the impairment level, 
    and signal-related parameters. Additionally, it handles specific properties for 
    narrowband signals, such as oversampling rates and center frequency offset 
    (CFO) error percentage.

    Attributes:
        minimum_params (List[str]): List of minimum required parameters for the narrowband dataset. 
        _oversampling_min (float): Minimum oversampling rate.
        _oversampling_max (float): Maximum oversampling rate.
        _cfo_percentage_error (float): Allowed center frequency offset error as a percentage.

    """

    minimum_params: List[str] = [
        'num_iq_samples_dataset',
        'fft_size',
        'impairment_level'
    ] 
    
    def __init__(
        self, 
        num_iq_samples_dataset: int, 
        fft_size: int,
        impairment_level: int,
        sample_rate: float = 10e6,
        num_signals_min: int = None,
        num_signals_distribution: np.ndarray | List[float]= None,
        snr_db_min: float = 0.0,
        snr_db_max: float = 50.0,
        signal_duration_percent_min: float = 80.0,
        signal_duration_percent_max: float = 100.0,
        transforms: list = [],
        target_transforms: list = [],
        class_list: List[str] = TorchSigSignalLists.all_signals,
        class_distribution = None,
        num_samples: int = None,
        **kwargs,
    ):
        """Initializes Narrowband Metadata. Sets `dataset_type="narrowband`.

        Args:
            num_iq_samples_dataset (int): The length of I/Q array per sample in the dataset.
            sample_rate (float, optional): The sample rate for the dataset (default is 10e6).
            fft_size (int): The size of FFT (number of bins) to be used in spectrogram calculation.
            num_signals_min (int, optional): Minimum number of signals per sample (default is 0).
            num_signals_distribution (np.ndarray | List[float], optional): The probability distribution 
                for generating samples with a specific number of signals. Defaults to uniform distribution if None.
            snr_db_min (float, optional): Minimum SNR (Signal-to-Noise Ratio) for the signals (default is 0.0).
            snr_db_max (float, optional): Maximum SNR for the signals (default is 50.0).
            signal_duration_percent_min (float, optional): Minimum duration of a signal as a percentage of the total sample duration (default is 80.0%).
            signal_duration_percent_max (float, optional): Maximum duration of a signal as a percentage (default is 100.0%).
            transforms (list, optional): Transforms to apply on the dataset (default in []).
            target_transforms (list, optional): Transforms for targets (default is an empty list).
            impairment_level (int, optional): Level of signal impairment (default is 2).
            class_list (List[str], optional): List of signal class names (default is all signals from TorchSigSignalLists).
            class_distribution (np.ndarray | List[float], optional): Probability distribution for classes (default is None).
            num_samples (int, optional): Length of the dataset. If None, an infinite dataset is assumed (default is None).

            
        """
        super().__init__(
            num_iq_samples_dataset=num_iq_samples_dataset, 
            sample_rate=sample_rate,
            fft_size=fft_size,
            impairment_level=impairment_level,
            num_signals_max=1,
            num_signals_min=num_signals_min,
            num_signals_distribution=num_signals_distribution,
            snr_db_min=snr_db_min,
            snr_db_max=snr_db_max,
            signal_duration_percent_min=signal_duration_percent_min,
            signal_duration_percent_max=signal_duration_percent_max,
            transforms=transforms, 
            target_transforms=target_transforms, 
            class_list=class_list,
            class_distribution=class_distribution,
            num_samples = num_samples,
            dataset_type="narrowband",
            **kwargs
        )


        # oversampling rate range will be used to derive bandwidth limits. the
        # oversampling rate is always greater than 1 such that there will be 
        # some proportion of the receive spectrum which is unoccupied, ex: the
        # noise floor will be visible
        self._oversampling_min = 1.25
        self._oversampling_max = 8

        # limits the range for center frequency as a proportion of the receiver
        # bandwidth. the center DatasetImpairmentsfrequency offset (CFO) represents the distance
        # of the signal's center frequency from complex baseband
        self._cfo_percentage_error = 10

    def _initialize_impairments(self) -> NarrowbandImpairments:
        """Initializes and returns an instance of the NarrowbandImpairments class.

        This method is responsible for creating an instance of the `NarrowbandImpairments` 
        class using the current `impairment_level` of the dataset. It allows for modeling 
        the impairments applied to narrowband signals within the dataset. The `parent` 
        argument of the `NarrowbandImpairments` class is set to the current instance 
        of the class, establishing a reference for interaction with other components.

        Returns:
            NarrowbandImpairments: A new instance of the `NarrowbandImpairments` class 
            initialized with the current `impairment_level` and parent dataset instance.
        """
        return NarrowbandImpairments(self.impairment_level)

    ## Read-Only Properties
    @property
    def center_freq_min(self) -> float:
        """The minimum center frequency for a signal

        The minimum is a boundary condition such that the center frequency
        will not alias across the lower sampling rate boundary.

        Returns:
            float: minimum center frequency boundary for signal
        """
        return -self.center_freq_max

    @property
    def center_freq_max(self) -> float:
        """The maximum center frequency for a signal

        The maximum is a boundary condition such that the center frequency
        will not alias across the upper sampling rate boundary.

        Returns:
            float: maximum center frequency boundary for signal
        """
        return self.sample_rate * (self._cfo_percentage_error / 100)

    @property
    def bandwidth_max(self) -> float:
        """The maximum bandwith for a signal

        Provides a boundary for the maximum bandwidth of a signal.
        The minimum signal bandwidth comes from the maximum oversampling
        rate.

        Returns:
            float: the maximum bandwidth for a signal
        """
        return self.sample_rate/self._oversampling_min

    @property
    def bandwidth_min(self) -> float:
        """The minimum bandwith for a signal

        Provides a boundary for the minimum bandwidth of a signal.
        The maximum signal bandwidth comes from the minimum oversampling
        rate.

        Returns:
            float: the minimum bandwidth for a signal
        """
        return self.sample_rate/self._oversampling_max




### Wideband Metadata

class WidebandMetadata(DatasetMetadata):
    """Wideband Dataset Metadata Class
    
    This class encapsulates all useful metadata for a wideband dataset, extending 
    the `DatasetMetadata` class. It adds functionality to manage the FFT size used 
    to compute the spectrogram, along with additional parameters specific to wideband 
    signals like bandwidth, center frequency, and impairments.

    Attributes:
        minimum_params (List[str]): List of the minimum parameters required for the dataset.

    """

    minimum_params: List[str] = [
        'num_iq_samples_dataset',
        'fft_size',
        'num_signals_max',
        'impairment_level'
    ]
    
    def __init__(
        self, 
        num_iq_samples_dataset: int, 
        fft_size: int,
        impairment_level: int,
        num_signals_max: int, 
        sample_rate: float = 10e6, 
        num_signals_min: int = None,
        num_signals_distribution: np.ndarray | List[float]= None,
        snr_db_min: float = 0.0,
        snr_db_max: float = 50.0,
        signal_duration_percent_min: float = 0.0,
        signal_duration_percent_max: float = 100.0,
        transforms: list = [],
        target_transforms: list = [],
        class_list: List[str] = TorchSigSignalLists.all_signals,
        class_distribution = None,
        num_samples: int = None,
        **kwargs,
    ):
        """Initializes the Wideband Metadata class, setting `dataset_type="wideband"`.
        
        Args:
            num_iq_samples_dataset (int): Length of I/Q array per sample in the dataset.
            sample_rate (float): Sample rate for the dataset.
            fft_size (int): Size of FFT (number of bins) used in spectrogram.
            impairment_level (int, optional): Impairment level for the signals.
            num_signals_max (int): Maximum number of signals per sample in the dataset.
            num_signals_min (int, optional): Minimum number of signals per sample (default is 0).
            num_signals_distribution (np.ndarray | List[float], optional): Distribution of signals for each value 
                in `[num_signals_min, num_signals_max]`. Defaults to None (uniform).
            snr_db_min (float, optional): Minimum SNR of signals (default is 0.0).
            snr_db_max (float, optional): Maximum SNR of signals (default is 50.0).
            signal_duration_percent_min (float, optional): Minimum signal duration percentage (default is 0.0).
            signal_duration_percent_max (float, optional): Maximum signal duration percentage (default is 100.0).
            transforms (list): Transforms applied to the dataset (default in []).
            target_transforms (list, optional): Target transforms applied (default is []).
            class_list (List[str], optional): List of signal class names (default is `TorchSigSignalLists.all_signals`).
            class_distribution (np.ndarray | List[float], optional): Probabilities for each class.
            # root (str, optional): Directory to write the dataset to disk (default is None).
            # overwrite (bool, optional): Whether to overwrite the existing dataset (default is False).
            num_samples (int, optional): Number of samples in the dataset (default is None, infinite dataset).
            **kwargs: Additional parameters to pass to the parent class.
        """  
        super().__init__(
            num_iq_samples_dataset=num_iq_samples_dataset, 
            sample_rate=sample_rate, 
            fft_size=fft_size,
            impairment_level=impairment_level,
            num_signals_max=num_signals_max, 
            num_signals_min=num_signals_min,
            num_signals_distribution=num_signals_distribution,
            snr_db_max=snr_db_max,
            snr_db_min=snr_db_min,
            signal_duration_percent_max=signal_duration_percent_max,
            signal_duration_percent_min=signal_duration_percent_min,
            transforms=transforms, 
            target_transforms=target_transforms, 
            class_list=class_list,
            class_distribution=class_distribution,
            # root = root,
            # save_type = save_type,
            # overwrite = overwrite,
            num_samples= num_samples,
            dataset_type="wideband",
            **kwargs,
        ) 

    
    def __repr__(self) -> str:
        """Returns a string representation of the WidebandMetadata instance.
        
        This provides a concise summary of the key parameters such as `num_iq_samples_dataset`, 
        `sample_rate`, `num_signals_max`, and `fft_size`.
        
        Returns:
            str: String representation of the WidebandMetadata instance.
        """
        return f"{self.__class__.__name__}(num_iq_samples_dataset={self.num_iq_samples_dataset}, sample_rate={self.sample_rate}, num_signals_max={self.num_signals_max}, fft_size={self.fft_size})"

    def _initialize_impairments(self) -> WidebandImpairments:
        """Initializes and returns an instance of the WidebandImpairments class.
        
        This method creates and returns an instance of the `WidebandImpairments` class 
        initialized with the current `impairment_level` of the dataset. It models 
        the impairments applied to the wideband signals.
        
        Returns:
            WidebandImpairments: A new instance of the `WidebandImpairments` class.
        """
        return WidebandImpairments(self.impairment_level)

    @property
    def center_freq_min(self) -> float:
        """The minimum center frequency for a signal

        The minimum is a boundary condition such that the center frequency
        will not alias across the lower sampling rate boundary.

        Returns:
            float: minimum center frequency boundary for signal
        """
        return -self.sample_rate/2

    @property
    def center_freq_max(self) -> float:
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
    def bandwidth_max(self) -> float:
        """The maximum bandwith for a signal

        Provides a boundary for the maximum bandwidth of a signal, which
        is the sampling rate.

        Returns:
            float: the maximum bandwidth for a signal
        """
        return self.sample_rate

    @property
    def bandwidth_min(self) -> float:
        """The minimum bandwith for a signal

        Provides a boundary for the minimum bandwidth of a signal, which
        is the bandwidth of a tone, which is sample rate / number of samples.

        Returns:
            float: the minimum bandwidth for a signal
        """
        return self.sample_rate / self.num_iq_samples_dataset
