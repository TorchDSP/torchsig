import numpy as np
from copy import deepcopy
from typing import Optional, Any

from torchsig.utils.types import SignalData, SignalDescription
from torchsig.transforms.transforms import SignalTransform
from torchsig.transforms.signal_processing import sp_functional as F
from torchsig.transforms.functional import NumericParameter, to_distribution


class Normalize(SignalTransform):
    """Normalize a IQ vector with mean and standard deviation.

    Args:
        norm :obj:`string`:
            Type of norm with which to normalize
        
        flatten :obj:`flatten`:
            Specifies if the norm should be calculated on the flattened
            representation of the input tensor

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """
    def __init__(
        self, 
        norm: Optional[int] = 2, 
        flatten: Optional[bool] = False,
    ):
        super(Normalize, self).__init__()
        self.norm = norm
        self.flatten = flatten

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.normalize(data.iq_data, self.norm, self.flatten)
        else:
            data = F.normalize(data, self.norm, self.flatten)
        return data


class RandomResample(SignalTransform):
    """Resample using poly-phase rational resampling technique.

    Args:
        rate_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            new_rate = rate_ratio*old_rate

            * If Callable, resamples to new_rate by calling rate_ratio()
            * If int or float, rate_ratio is fixed by value provided
            * If list, rate_ratio is any element in the list
            * If tuple, rate_ratio is in range of (tuple[0], tuple[1])

        num_iq_samples (:obj:`int`):
            Since resampling changes the number of points in a tensor, it is necessary to designate how
            many samples should be returned. In the case more samples are produced, the last num_iq_samples of
            the resampled tensor are returned.  In the case les samples are produced, the returned tensor is zero-padded
            to have num_iq_samples.

        keep_samples (:obj:`int`):
            Despite returning a different number of samples being an issue, return however many samples 
            are returned from resample_poly

    Note:
        When rate_ratio is > 1.0, the resampling algorithm produces more samples than the original tensor.
        When rate_ratio < 1.0, the resampling algorithm produces less samples than the original tensor. Hence,
        it is necessary to specify a number of samples to return from the newly resampled tensor so that there are
        always enough samples to return

    Example:
        >>> import torchsig.transforms as ST
        >>> # Randomly resample to a new_rate that is between .75 and 1.5 times the original rate
        >>> transform = ST.RandomResample(lambda: np.random.uniform(.75, 1.5, size=1), num_iq_samples=128)
        >>> # Randomly resample to a new_rate that is either 1.5 or 3.0
        >>> transform = ST.RandomResample([1.5, 3.0], num_iq_samples=128)
        >>> # Resample to a new_rate that is always 1.5
        >>> transform = ST.RandomResample(1.5, num_iq_samples=128)
        
    """
    def __init__(
        self, 
        rate_ratio: NumericParameter = (1.5, 3.0), 
        num_iq_samples: Optional[int] = 256, 
        keep_samples: Optional[bool] = False,
    ):
        super(RandomResample, self).__init__()
        self.rate_ratio = to_distribution(rate_ratio, self.random_generator)
        self.num_iq_samples = num_iq_samples
        self.keep_samples = keep_samples

    def __call__(self, data: Any) -> Any:
        new_rate = self.rate_ratio()
        if new_rate == 1.0:
            return data
        if isinstance(data, SignalData):
            # Update the SignalDescriptions with the new rate
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            anti_alias_lpf = False
            for signal_desc_idx, signal_desc in enumerate(signal_description):
                new_signal_desc = deepcopy(signal_desc)
                # Update time descriptions
                new_num_iq_samples = new_signal_desc.num_iq_samples * new_rate
                start_iq_sample = new_signal_desc.start * new_num_iq_samples
                stop_iq_sample = new_signal_desc.stop * new_num_iq_samples
                if new_rate > 1.0:
                    # If the new rate is greater than 1.0, the resampled tensor
                    # is larger than the original tensor and is truncated to be
                    # the last <self.num_iq_samples> only
                    trunc_samples = new_num_iq_samples - self.num_iq_samples
                    new_start_iq_sample = start_iq_sample - trunc_samples
                    new_stop_iq_sample = stop_iq_sample - trunc_samples
                    new_signal_desc.start = new_start_iq_sample / self.num_iq_samples if new_start_iq_sample > 0.0 else 0.0
                    new_signal_desc.stop = new_stop_iq_sample / self.num_iq_samples if new_stop_iq_sample < self.num_iq_samples else 1.0
                else:
                    # If the new rate is less than 1.0, the resampled tensor
                    # is smaller than the original tensor and is zero-padded
                    # at the end to length <self.num_iq_samples>
                    new_signal_desc.start *= new_rate
                    new_signal_desc.stop *= new_rate
                    
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                
                # Check for signals lost in truncation process
                if new_signal_desc.start > 1.0 or new_signal_desc.stop < 0.0:
                    continue
                
                # Update frequency descriptions
                new_signal_desc.samples_per_symbol *= new_rate
                # Check freq bounds for cases of partial signals
                # Upsampling these signals will distort them, but at least the label will follow
                if new_signal_desc.lower_frequency < -0.5 and new_signal_desc.upper_frequency / new_rate > -0.5 and new_rate > 1.0:
                    new_signal_desc.lower_frequency = -0.5
                    new_signal_desc.bandwidth = new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                    new_signal_desc.center_frequency = new_signal_desc.lower_frequency + new_signal_desc.bandwidth / 2
                if new_signal_desc.upper_frequency > 0.5 and new_signal_desc.lower_frequency / new_rate < 0.5 and new_rate > 1.0:
                    new_signal_desc.upper_frequency = 0.5
                    new_signal_desc.bandwidth = new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                    new_signal_desc.center_frequency = new_signal_desc.lower_frequency + new_signal_desc.bandwidth / 2
                new_signal_desc.lower_frequency /= new_rate
                new_signal_desc.upper_frequency /= new_rate
                new_signal_desc.center_frequency /= new_rate
                new_signal_desc.bandwidth /= new_rate
                
                if (new_signal_desc.lower_frequency < -0.45 or new_signal_desc.lower_frequency > 0.45 or \
                   new_signal_desc.upper_frequency < -0.45 or new_signal_desc.upper_frequency > 0.45) and \
                   new_rate < 1.0:
                    # If downsampling and new signals are near band edge, apply a LPF to handle aliasing
                    anti_alias_lpf = True
                
                # Check new freqs for inclusion
                if new_signal_desc.lower_frequency > 0.5 or new_signal_desc.upper_frequency < -0.5:
                    continue
                
                # Append updates to the new description
                new_signal_description.append(new_signal_desc)
                
            # Apply transform to data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = F.resample(
                data.iq_data,
                np.floor(new_rate*100).astype(np.int32),
                100,
                self.num_iq_samples,
                self.keep_samples,
                anti_alias_lpf,
            )
            
            # Update the new data's SignalDescription
            new_data.signal_description = new_signal_description
            
        else:
            new_data = F.resample(
                data,
                np.floor(new_rate*100).astype(np.int32),
                100,
                self.num_iq_samples,
                self.keep_samples
            )
        return new_data
