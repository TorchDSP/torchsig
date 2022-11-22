import numpy as np
from scipy import signal


def normalize(tensor: np.ndarray, norm_order: int = 2, flatten: bool = False) -> np.ndarray:
    """Scale a tensor so that a specfied norm computes to 1. For detailed information, see :func:`numpy.linalg.norm.`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=np.inf, norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Args:
        tensor (:class:`numpy.ndarray`)):
            (batch_size, vector_length, ...)-sized tensor to be normalized.
            
        norm_order (:class:`int`)):
            norm order to be passed to np.linalg.norm
            
        flatten (:class:`bool`)):
            boolean specifying if the input array's norm should be calculated on the flattened representation of the input tensor

    Returns:
        Tensor:
            Normalized complex array.
    """
    if flatten:
        flat_tensor = tensor.reshape(tensor.size)
        norm = np.linalg.norm(flat_tensor, norm_order, keepdims=True)
    else:
        norm = np.linalg.norm(tensor, norm_order, keepdims=True)
    return np.multiply(tensor, 1.0/norm)


def resample(
    tensor: np.ndarray, 
    up_rate: int,
    down_rate: int, 
    num_iq_samples: int, 
    keep_samples: bool, 
    anti_alias_lpf: bool = False,
) -> np.ndarray:
    """Resample a tensor by rational value

    Args:
        tensor (:class:`numpy.ndarray`):
            tensor to be resampled.

        up_rate (:class:`int`):
            rate at which to up-sample the tensor

        down_rate (:class:`int`):
            rate at which to down-sample the tensor

        num_iq_samples (:class:`int`):
            number of IQ samples to have after resampling

        keep_samples (:class:`bool`):
            boolean to specify if the resampled data should be returned as is
            
        anti_alias_lpf (:class:`bool`)):
            boolean to specify if an additional anti aliasing filter should be
            applied

    Returns:
        Tensor:
            Resampled tensor
    """
    if anti_alias_lpf:
        new_rate = up_rate/down_rate
        # Filter around center to future bandwidth
        num_taps = int(2*np.ceil(50*2*np.pi/new_rate/.125/22)) # fred harris rule of thumb * 2
        taps = signal.firwin(
            num_taps,
            new_rate*0.98,
            width=new_rate * .02,
            window=signal.get_window("blackman", num_taps),
            scale=True
        )
        tensor = signal.fftconvolve(tensor, taps, mode="same")
    
    # Resample
    resampled = signal.resample_poly(tensor, up_rate, down_rate)

    # Handle extra or not enough IQ samples
    if keep_samples:
        new_tensor = resampled
    elif resampled.shape[0] > num_iq_samples:
        new_tensor = resampled[-num_iq_samples:]
    else:
        new_tensor = np.zeros((num_iq_samples,), dtype=np.complex128)
        new_tensor[:resampled.shape[0]] = resampled
    
    return new_tensor
