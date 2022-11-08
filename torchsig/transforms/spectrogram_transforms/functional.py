import numpy as np


def drop_spec_samples(
    tensor: np.ndarray, 
    drop_starts: np.ndarray, 
    drop_sizes: np.ndarray, 
    fill: str,
) -> np.ndarray:
    """Drop samples at specified input locations/durations with fill technique

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.
            
        drop_starts (:class:`numpy.ndarray`):
            Indices of where drops start
            
        drop_sizes (:class:`numpy.ndarray`):
            Durations of each drop instance
            
        fill (:obj:`str`):
            String specifying how the dropped samples should be replaced

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone the dropped samples

    """
    flat_spec = tensor.reshape(tensor.shape[0], tensor.shape[1]*tensor.shape[2])
    for idx, drop_start in enumerate(drop_starts):
        if fill == "ffill":
            drop_region_real = np.ones(drop_sizes[idx])*flat_spec[0,drop_start-1]
            drop_region_complex = np.ones(drop_sizes[idx])*flat_spec[1,drop_start-1]
            flat_spec[0,drop_start:drop_start+drop_sizes[idx]] = drop_region_real
            flat_spec[1,drop_start:drop_start+drop_sizes[idx]] = drop_region_complex
        elif fill == "bfill":
            drop_region_real = np.ones(drop_sizes[idx])*flat_spec[0,drop_start+drop_sizes[idx]]
            drop_region_complex = np.ones(drop_sizes[idx])*flat_spec[1,drop_start+drop_sizes[idx]]
            flat_spec[0,drop_start:drop_start+drop_sizes[idx]] = drop_region_real
            flat_spec[1,drop_start:drop_start+drop_sizes[idx]] = drop_region_complex
        elif fill == "mean":
            drop_region_real = np.ones(drop_sizes[idx])*np.mean(flat_spec[0])
            drop_region_complex = np.ones(drop_sizes[idx])*np.mean(flat_spec[1])
            flat_spec[0,drop_start:drop_start+drop_sizes[idx]] = drop_region_real
            flat_spec[1,drop_start:drop_start+drop_sizes[idx]] = drop_region_complex
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx])
            flat_spec[:,drop_start:drop_start+drop_sizes[idx]] = drop_region
        elif fill == "min":
            drop_region_real = np.ones(drop_sizes[idx])*np.min(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx])*np.min(np.abs(flat_spec[1]))
            flat_spec[0,drop_start:drop_start+drop_sizes[idx]] = drop_region_real
            flat_spec[1,drop_start:drop_start+drop_sizes[idx]] = drop_region_complex
        elif fill == "max":
            drop_region_real = np.ones(drop_sizes[idx])*np.max(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx])*np.max(np.abs(flat_spec[1]))
            flat_spec[0,drop_start:drop_start+drop_sizes[idx]] = drop_region_real
            flat_spec[1,drop_start:drop_start+drop_sizes[idx]] = drop_region_complex
        elif fill == "low":
            drop_region = np.ones(drop_sizes[idx])*1e-3
            flat_spec[:,drop_start:drop_start+drop_sizes[idx]] = drop_region
        elif fill == "ones":
            drop_region = np.ones(drop_sizes[idx])
            flat_spec[:,drop_start:drop_start+drop_sizes[idx]] = drop_region
        else:
            raise ValueError("fill expects ffill, bfill, mean, zero, min, max, low, ones. Found {}".format(fill))
    new_tensor = flat_spec.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2])
    return new_tensor


def spec_patch_shuffle(
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
    channels, height, width = tensor.shape
    num_freq_patches = int(height/patch_size)
    num_time_patches = int(width/patch_size)
    num_patches = int(num_freq_patches * num_time_patches)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )
    
    for patch_idx in patches_to_shuffle:
        freq_idx = np.floor(patch_idx / num_freq_patches)
        time_idx = patch_idx % num_time_patches
        patch = tensor[
            :,
            int(freq_idx*patch_size):int(freq_idx*patch_size+patch_size),
            int(time_idx*patch_size):int(time_idx*patch_size+patch_size)
        ]
        patch = patch.reshape(int(2*patch_size*patch_size))
        np.random.shuffle(patch)
        patch = patch.reshape(2,int(patch_size),int(patch_size))
        tensor[
            :,
            int(freq_idx*patch_size):int(freq_idx*patch_size+patch_size),
            int(time_idx*patch_size):int(time_idx*patch_size+patch_size)
        ] = patch
    return tensor


def spec_translate(
    tensor: np.ndarray, 
    time_shift: int,
    freq_shift: int,
) -> np.ndarray:
    """Apply time/freq translation to input spectrogram

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.
            
        time_shift (:obj:`int`):
            Time shift
            
        freq_shift (:obj:`int`):
            Frequency shift
            
    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone time/freq translation

    """
    # Pre-fill the data with background noise
    new_tensor = np.random.rand(*tensor.shape)*np.percentile(np.abs(tensor),50)

    # Apply translation
    channels, height, width = tensor.shape
    if time_shift >= 0 and freq_shift >= 0:
        valid_dur = width - time_shift
        valid_bw = height - freq_shift
        new_tensor[:,freq_shift:,time_shift:] = tensor[:,:valid_bw,:valid_dur]
    elif time_shift < 0 and freq_shift >= 0:
        valid_dur = width + time_shift
        valid_bw = height - freq_shift
        new_tensor[:,freq_shift:,:valid_dur] = tensor[:,:valid_bw,-time_shift:]
    elif time_shift >= 0 and freq_shift < 0:
        valid_dur = width - time_shift
        valid_bw = height + freq_shift
        new_tensor[:,:valid_bw,time_shift:] = tensor[:,-freq_shift:,:valid_dur]
    elif time_shift < 0 and freq_shift < 0:
        valid_dur = width + time_shift
        valid_bw = height + freq_shift
        new_tensor[:,:valid_bw,:valid_dur] = tensor[:,-freq_shift:,-time_shift:]

    return new_tensor
