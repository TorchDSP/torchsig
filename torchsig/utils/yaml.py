"""YAML utilities
"""

import yaml
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset


def custom_representer(dumper, value):
    """Custom representer for YAML to handle sequences (lists).

    This function customizes how lists are represented in the YAML output, using 
    flow style for sequences (inline lists).

    Args:
        dumper (yaml.Dumper): The YAML dumper responsible for serializing the data.
        value (list): The list to be represented in YAML.

    Returns:
        yaml.Dumper: The dumper with the custom representation for the list.
    """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)

def write_dict_to_yaml(filename: str, info_dict: dict) -> None:
    """Writes a dictionary to a YAML file with customized settings.

    This function writes the provided `info_dict` to a YAML file. It customizes 
    the representation of lists by using the `custom_representer`, and it uses 
    specific formatting options (e.g., no sorting of keys, custom line width).

    Args:
        filename (str): The name of the YAML file to which the dictionary will be written.
        info_dict (dict): The dictionary to be written to the YAML file.

    Returns:
        None: This function does not return any value.
    """
    yaml.add_representer(list, custom_representer)

    with open(filename, 'w+') as file:
        yaml.dump(info_dict, file, default_flow_style=False, sort_keys=False, width=200)

def dataset_metadata_from_yaml_dict(yaml_dict):
    """
    passes data from the yaml_dict as needed into the DatasetMetadata constructor, and returns a new DatasetMetadata
    """
    return DatasetMetadata(
        num_iq_samples_dataset = yaml_dict["num_iq_samples_dataset"], 
        fft_size = yaml_dict["fft_size"],
        num_signals_min = yaml_dict["num_signals_min"],
        num_signals_max = yaml_dict["num_signals_max"],
        sample_rate = yaml_dict["sample_rate"],
        num_signals_distribution = yaml_dict["num_signals_distribution"],
        snr_db_min = yaml_dict["snr_db_min"],
        snr_db_max = yaml_dict["snr_db_max"],
        signal_duration_min = yaml_dict["signal_duration_min"],
        signal_duration_max = yaml_dict["signal_duration_max"],
        signal_bandwidth_min = yaml_dict["signal_bandwidth_min"],
        signal_bandwidth_max = yaml_dict["signal_bandwidth_max"],
        signal_center_freq_min = yaml_dict["signal_center_freq_min"],
        signal_center_freq_max = yaml_dict["signal_center_freq_max"],
        cochannel_overlap_probability = yaml_dict["cochannel_overlap_probability"],
        class_list = yaml_dict["class_list"],
        class_distribution = yaml_dict["class_distribution"],
    )

def dataset_from_yaml_dict(yaml_dict):
    """
    passes data from the yaml_dict as needed into the TorchSigIterableDataset constructor, and returns a new TorchSigIterableDataset
    """
    dataset_metadata = dataset_metadata_from_yaml_dict(yaml_dict["dataset_metadata"])
    return TorchSigIterableDataset(
        dataset_metadata = dataset_metadata,
        transforms = [],
        component_transforms = [],
        target_labels = yaml_dict["target_labels"],
        seed = yaml_dict["seed"],
    )

def load_dataset_yaml(filepath):
    """
    loads YAML data from specified filepath and uses it to construct and return a new TorchSigIterableDataset
    """
    loaded_dict = {}
    with open(filepath, 'r') as yaml_file:
        loaded_dict = yaml.safe_load(yaml_file)
    return dataset_from_yaml_dict(loaded_dict)

def save_dataset_yaml(filepath, dataset):
    """
    saves YAML data to specified filepath to represent the input TorchSigIterableDataset
    """
    yaml_dict = {}
    yaml_dict["seed"] = dataset.rng_seed
    yaml_dict["target_labels"] = dataset.target_labels
    yaml_dict["dataset_metadata"] = dataset_metadata_to_yaml_dict(dataset.dataset_metadata)
    write_dict_to_yaml(filepath, yaml_dict)

def dataset_metadata_to_yaml_dict(dataset_metadata):
    """
    returns a dictionary representation of a DatasetMetadata object for storing as YAML
    """
    yaml_dict = {}
    yaml_dict["num_iq_samples_dataset"] = dataset_metadata.num_iq_samples_dataset
    yaml_dict["fft_size"] = dataset_metadata.fft_size
    yaml_dict["num_signals_min"] = dataset_metadata.num_signals_min
    yaml_dict["num_signals_max"] = dataset_metadata.num_signals_max
    yaml_dict["sample_rate"] = dataset_metadata.sample_rate
    yaml_dict["num_signals_distribution"] = dataset_metadata.num_signals_distribution
    yaml_dict["snr_db_min"] = dataset_metadata.snr_db_min
    yaml_dict["snr_db_max"] = dataset_metadata.snr_db_max
    yaml_dict["signal_duration_min"] = dataset_metadata.signal_duration_min
    yaml_dict["signal_duration_max"] = dataset_metadata.signal_duration_max
    yaml_dict["signal_bandwidth_min"] = dataset_metadata.signal_bandwidth_min
    yaml_dict["signal_bandwidth_max"] = dataset_metadata.signal_bandwidth_max
    yaml_dict["signal_center_freq_min"] = dataset_metadata.signal_center_freq_min
    yaml_dict["signal_center_freq_max"] = dataset_metadata.signal_center_freq_max
    yaml_dict["cochannel_overlap_probability"] = dataset_metadata.cochannel_overlap_probability
    yaml_dict["class_list"] = dataset_metadata.class_list
    yaml_dict["class_distribution"] = dataset_metadata.class_distribution
    return yaml_dict

        