""" Loads default yaml configs
"""

# Built-In
import yaml
from pathlib import Path
import sys

sys.path.append(f"{Path(__file__).parent}")

def get_default_yaml_config(
    impairment_level: bool | int,
    train: bool,
    ret_config_path: bool = False
) -> dict:
    """Loads the default YAML configuration for a given dataset type, impairment level, and training/validation status.

    This function constructs the path to the appropriate YAML configuration file based on the dataset type, impairment level, and whether the dataset is for training or validation. It then loads the YAML file and returns its contents as a dictionary. 

    Args:
        impairment_level (bool | int): The impairment level for the dataset:
            - 0 or False for 'clean' data,
            - 2 or True for 'impaired' data.
        train (bool): Whether the dataset is for training (`True`) or validation (`False`).
        ret_config_path (bool, optional): If `True`, the function also returns the path to the configuration file. Defaults to `False`.

    Returns:
        dict: The parsed dataset metadata from the YAML configuration file.
        If `ret_config_path` is `True`, returns a tuple of the dataset metadata and the configuration file path.

    Raises:
        ValueError: If the impairment level is invalid or 1.
    
    Example:
        # Load the default configuration for a clean narrowband dataset for training
        config = get_default_yaml_config('narrowband', 0, True)

        # Load the default configuration for an impaired dataset for validation and get the config path
        config, path = get_default_yaml_config(2, False, ret_config_path=True)
    """

    if impairment_level == 1:
        raise ValueError("Default config does not exist for impairment level 1")
    
    impairment_level = "impaired" if impairment_level == 2 else "clean"

    train = "train" if train else "val"

    config_path = f"dataset_{impairment_level}_{train}.yaml"
    full_config_path = f"{Path(__file__).parent}/{config_path}"

    

    with open(full_config_path, 'r') as f:
        dataset_metadata = yaml.load(f, Loader=yaml.FullLoader)

    if ret_config_path:
        return dataset_metadata, config_path
    # else:
    return dataset_metadata
