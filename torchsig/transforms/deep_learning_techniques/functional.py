import numpy as np


def cut_out(
    tensor: np.ndarray, 
    cut_start: float, 
    cut_dur: float, 
    cut_type: str,
) -> np.ndarray:
    """Performs the CutOut using the input parameters

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.
            
        cut_start: (:obj:`float`):
            Start of cut region in range [0.0,1.0)
            
        cut_dur: (:obj:`float`):
            Duration of cut region in range (0.0,1.0]
            
        cut_type: (:obj:`str`):
            String specifying type of data to fill in cut region with

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone cut out

    """
    num_iq_samples = tensor.shape[0]
    cut_start = int(cut_start * num_iq_samples)

    # Create cut mask
    cut_mask_length = int(num_iq_samples * cut_dur)
    if cut_mask_length + cut_start > num_iq_samples:
        cut_mask_length = num_iq_samples - cut_start
    
    if cut_type == "zeros":
        cut_mask = np.zeros(cut_mask_length, dtype=np.complex64)
    elif cut_type == "ones":
        cut_mask = np.ones(cut_mask_length) + 1j*np.ones(cut_mask_length)
    elif cut_type == "low_noise":
        real_noise = np.random.randn(cut_mask_length)
        imag_noise = np.random.randn(cut_mask_length)
        noise_power_db = -100
        cut_mask = (10.0**(noise_power_db/20.0))*(real_noise + 1j*imag_noise)/np.sqrt(2)
    elif cut_type == "avg_noise":
        real_noise = np.random.randn(cut_mask_length)
        imag_noise = np.random.randn(cut_mask_length)
        avg_power = np.mean(np.abs(tensor)**2)
        cut_mask = avg_power*(real_noise + 1j*imag_noise)/np.sqrt(2)
    elif cut_type == "high_noise":
        real_noise = np.random.randn(cut_mask_length)
        imag_noise = np.random.randn(cut_mask_length)
        noise_power_db = 40
        cut_mask = (10.0**(noise_power_db/20.0))*(real_noise + 1j*imag_noise)/np.sqrt(2)
    else:
        raise ValueError("cut_type must be: zeros, ones, low_noise, avg_noise, or high_noise. Found: {}".format(cut_type))
        
    # Insert cut mask into tensor
    tensor[cut_start:cut_start+cut_mask_length] = cut_mask
    
    return tensor


def patch_shuffle(
    tensor: np.ndarray, 
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.
            
        patch_size (:obj:`int`):
            Size of each patch to shuffle
            
        shuffle_ratio (:obj:`float`):
            Ratio of patches to shuffle
            
    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone patch shuffling

    """
    num_patches = int(tensor.shape[0] / patch_size)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches, 
        replace=False, 
        size=num_to_shuffle,
    )
    
    for patch_idx in patches_to_shuffle:
        patch_start = int(patch_idx * patch_size)
        patch = tensor[patch_start:patch_start+patch_size]
        np.random.shuffle(patch)
        tensor[patch_start:patch_start+patch_size] = patch
        
    return tensor
