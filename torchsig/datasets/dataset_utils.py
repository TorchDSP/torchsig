"""Dataset Utilities
"""

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    frequency_shift,
    upconversion_anti_aliasing_filter,
)

import numpy as np

import yaml

# name of yaml file where dataset information will be written
dataset_yaml_name = "create_dataset_info.yaml"
# name of yaml file where dataset writing information will be written
writer_yaml_name = "writer_info.yaml"



def dataset_full_path(dataset_type: str, impairment_level: int, train: bool = None) -> str:
    """Generates the full path for a dataset based on its type, impairment level, and whether it is for training.

    Args:
        dataset_type (str): Type of dataset (e.g., 'narrowband', 'wideband').
        impairment_level (int): The impairment level for the dataset (0 = clean, 1 = level 1, 2 = impaired).
        train (bool, optional): Whether the dataset is for training (True) or validation (False). Defaults to None.

    Returns:
        str: The full path to the dataset, e.g., 'torchsig_narrowband_clean/train'.

    Example:
        full_path = dataset_full_path('narrowband', 0, True)
        print(full_path)  # Output: 'torchsig_narrowband_clean/train'
    """
    impaired_names = [
        "clean",
        "impaired_level_1",
        "impaired"
    ]
    impaired = impaired_names[impairment_level]
    
    # e.g., torchsig_narrowband_clean
    full_root = f"torchsig_{dataset_type}_{impaired}"


    if train is not None:
        # e.g., torchsig_narrowband_clean/train
        subpath = "train" if train else "val"
        full_root = f"{full_root}/{subpath}"

    return full_root

    


def collate_fn(batch):
    """Collates a batch by zipping its elements together.

    Args:
        batch (tuple): A batch from the dataloader.

    Returns:
        tuple: A tuple of zipped elements, where each element corresponds to a single batch item.
    """
    return tuple(zip(*batch))


def frequency_shift_signal(
    signal: Signal,
    center_freq_min: float,
    center_freq_max: float,
    sample_rate: float,
    frequency_max: float,
    frequency_min: float,
    random_generator: np.random.Generator = np.random.default_rng(seed=None),
) -> Signal:
    """Randomly shifts the frequency of a signal to a new center frequency and applies aliasing filters if necessary.

    Args:
        signal (Signal): The signal object to be frequency shifted.
        center_freq_min (float): Minimum center frequency for the random shift.
        center_freq_max (float): Maximum center frequency for the random shift.
        sample_rate (float): The sample rate of the signal.
        frequency_max (float): Maximum frequency limit for aliasing.
        frequency_min (float): Minimum frequency limit for aliasing.
        random_generator (np.random.Generator, optional): Random number generator for generating the random shift. Defaults to `np.random.default_rng()`.

    Returns:
        Signal: The frequency-shifted signal with updated metadata.
    """
    # randomize the center frequency
    center_freq = random_generator.uniform(low=center_freq_min, high=center_freq_max)

    # frequency shift to center_freq
    signal.data = frequency_shift(signal.data, center_freq, sample_rate)

    # update center_freq field in metadata
    signal.metadata.center_freq = center_freq

    # calculate upper and lower frequency edges of signal
    upper_freq = signal.metadata.upper_freq
    lower_freq = signal.metadata.lower_freq

    # has aliasing occured due to the upconversion to the signal?
    if (upper_freq > frequency_max or lower_freq < frequency_min):
        # apply an anti-aliasing filter to the signal to attenuate energy that
        # wrapped around -fs/2 or fs/2. additionally, due to the filtering the
        # bandwidth changed bandwidth, and therefore changed the center frequency,
        # so update the two metadata fields accordingly
        signal.data, signal.metadata.center_freq, signal.metadata.bandwidth = upconversion_anti_aliasing_filter (
            signal.data,
            signal.metadata.center_freq,
            signal.metadata.bandwidth,
            sample_rate,
            frequency_max,
            frequency_min
        )
    #else: # do nothing

    # because we have altered both the IQ samples and metdata, run verify()
    # to ensure nothing is broken
    signal.verify()

    return signal


def save_type(transforms: list, target_transforms: list):
    """Determines if the dataset will generate 'raw' IQ data, which means no transform and target transforms have been applied.

    Args:
        transforms (list): A list of transformations to be applied to the data.
        target_transforms (list): A list of target transformations.

    Returns:
        bool: `True` if no transformations are applied, indicating raw IQ data; otherwise `False`.
    """
    if len(transforms) > 0 or len(target_transforms) > 0:
        return False
    return True
        



def to_dataset_metadata(dataset_metadata: DatasetMetadata | str | dict):
    """Converts the input dataset metadata to an appropriate DatasetMetadata object.

    Args:
        dataset_metadata (DatasetMetadata | str | dict): The dataset metadata, which can be:
            - A `DatasetMetadata` object,
            - A string representing the path to a YAML file containing the metadata,
            - A dictionary representing the dataset metadata.

    Returns:
        DatasetMetadata: The corresponding `DatasetMetadata` object initialized with the provided parameters.

    Raises:
        ValueError: If the input `dataset_metadata` is not valid or if required fields are missing from the metadata.
    """

    if isinstance(dataset_metadata, DatasetMetadata):
        return dataset_metadata

    if isinstance(dataset_metadata, str):
        with open(dataset_metadata, 'r') as f:
            dataset_metadata = yaml.load(f, Loader=yaml.FullLoader)

    if isinstance(dataset_metadata, dict):
        # check that yaml file has minimum required params
        if "required" not in dataset_metadata.keys():
            raise ValueError("Invalid dataset_metadata. Does not have required field.")
        
        # validate dataset_type exists
        if "dataset_type" not in dataset_metadata['required'].keys():
            raise ValueError("Invalid dataset_metadata. Does not have dataset_type field under required.")
        # get dataset_type
        dataset_type = dataset_metadata['required']['dataset_type'].lower()

        # check if accidentally set dataset_type wrong
        if "num_signals_max" in dataset_metadata['required'].keys() and dataset_type == "narrowband":
            raise ValueError("num_signals_max defined in required params but dataset_type is narrowband. Should dataset_type be wideband?")
        
        # use appropriate dataset metadata type
        metadata = DatasetMetadata

        # Validate minimum parameters given in yaml to instantiate
        for min_param in metadata.minimum_params:
            if min_param not in dataset_metadata['required'].keys():
                raise ValueError(f"Missing required parameter {min_param} in dataset_metadata.")

        
        # Put parameters into a flattened dictionary
        init_params_dict = dataset_metadata['required']

        # Remove dataset_type from the parameters
        del dataset_metadata['required']['dataset_type']

        # Remove transforms if they exist
        if "transforms" in dataset_metadata['overrides'].keys():
            del dataset_metadata['overrides']['transforms']
        
        # Remove target transforms if they exist
        if "target_transforms" in dataset_metadata['overrides'].keys():
            del dataset_metadata['overrides']['target_transforms']

        # Remove read_only if they exist
        if "read_only" in dataset_metadata.keys():
            del dataset_metadata['read_only']
        
        # Handle if class_distribution is "uniform"
        if "class_distribution" in dataset_metadata['overrides'].keys():
            if dataset_metadata['overrides']['class_distribution'] == "uniform":
                dataset_metadata['overrides']['class_distribution'] = None
        
        # Handle if class_list is "all"
        if "class_list" in dataset_metadata['overrides'].keys():
            if dataset_metadata['overrides']['class_list'] == "all":
                dataset_metadata['overrides']['class_list'] = None

        # Handle if num_signals_distribution is "uniform"
        if "num_signals_distribution" in dataset_metadata['overrides'].keys():
            if dataset_metadata['overrides']['num_signals_distribution'] == "uniform":
                dataset_metadata['overrides']['num_signals_distribution'] = None

        # Merge overrides and write parameters if they exist
        if "overrides" in dataset_metadata.keys():
            init_params_dict = init_params_dict | dataset_metadata['overrides']

        if "write" in dataset_metadata.keys():
            init_params_dict = init_params_dict | dataset_metadata['write']
        
        # Unpack dataset metadata and return the appropriate metadata object
        return metadata(**init_params_dict)

    # else:
    # If the input is neither DatasetMetadata, str, nor dict     
    raise ValueError("Invalid dataset_metadata.")
