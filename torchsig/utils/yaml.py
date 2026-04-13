"""YAML utilities"""

from pathlib import Path
from typing import Any
import yaml

from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.datasets.datasets import TorchSigIterableDataset


def _require(cond: bool, msg: str) -> None:
    # Error helper function
    if not cond:
        raise ValueError(msg)
    

def custom_representer(dumper, value: list) -> yaml.Dumper:
    """Custom representer for YAML to handle sequences (lists).

    This function customizes how lists are represented in the YAML output, using
    flow style for sequences (inline lists).

    Args:
        dumper: The YAML dumper responsible for serializing the data.
        value: The list to be represented in YAML.

    Returns:
        The dumper with the custom representation for the list.
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


def load_config_from_yaml(path: Path) -> TorchSigDatasetConfig:
    """Loads YAML dataset configuration from the specified filepath, extracts the dataset metadata
    for use in dataset construction, and configures sampling mode and output representation.

    Args:
        filepath: Path to the YAML file containing dataset metadata.

    Returns:
        A dictionary containing the dataset metadata extracted from the YAML file.
    """
    # load configuration from yaml file
    cfg = yaml.safe_load(path.read_text()) or {}
    _require(isinstance(cfg, dict), "YAML root must be a mapping/dict")
    _require("dataset_metadata" in cfg and isinstance(cfg["dataset_metadata"], dict), 
                                                    "dataset_metadata must be a dict")
    # data format
    output = cfg.get("output") or {}
    _require(isinstance(output, dict), "output must be a dict")
    rep = output.get("representation")
    _require(rep in ("iq", "spectrogram"), "output.representation must be 'iq' or 'spectrogram'")

    # signal sampling (dataset classes)
    ss = cfg.get("signal_sampling") or {}
    _require(isinstance(ss, dict), "signal_sampling must be a dict")
    mode = ss.get("mode")
    _require(mode in ("per_signal", "per_family"), "signal_sampling.mode must be 'per_signal' or 'per_family'")

    return TorchSigDatasetConfig(
        dataset_id=str(cfg.get("dataset_id", path.stem)),
        dataset_length=int(cfg["dataset_length"]),
        seed=int(cfg["seed"]),
        impairment_level=int(cfg["impairment_level"]),
        output_representation=rep,
        output_spectrogram_fft=cfg["dataset_metadata"].get("fft_size"),
        signal_sampling_mode=mode,
        dataset_metadata=dict(cfg["dataset_metadata"]),
    )


def dataset_from_yaml_dict(yaml_dict: dict[str, Any]) -> TorchSigIterableDataset:
    """Creates a TorchSigIterableDataset from a YAML dictionary.

    Passes data from the yaml_dict as needed into the TorchSigIterableDataset
    constructor and returns a new TorchSigIterableDataset.

    Args:
        yaml_dict: dictionary containing dataset configuration with keys:
            - "dataset_metadata": Dataset metadata
            - "target_labels": List of target labels
            - "seed": Random seed value

    Returns:
        Configured TorchSigIterableDataset instance.
    """
    dataset_metadata = yaml_dict["dataset_metadata"]
    return TorchSigIterableDataset(
        metadata=dataset_metadata,
        transforms=[],
        target_labels=yaml_dict["target_labels"],
        seed=yaml_dict["seed"],
    )


def load_dataset_yaml(filepath: str) -> TorchSigIterableDataset:
    """Loads YAML data from specified filepath and constructs a dataset.

    Loads YAML data from the specified filepath and uses it to construct and
    return a new TorchSigIterableDataset.

    Args:
        filepath: Path to the YAML file containing dataset configuration.

    Returns:
        Configured TorchSigIterableDataset instance.
    """
    loaded_dict = {}
    with open(filepath) as yaml_file:
        loaded_dict = yaml.safe_load(yaml_file)
    return dataset_from_yaml_dict(loaded_dict)


def save_dataset_yaml(filepath: str, dataset: TorchSigIterableDataset) -> None:
    """Saves dataset configuration to a YAML file.

    Saves YAML data to the specified filepath to represent the input
    TorchSigIterableDataset.

    Args:
        filepath: Path where the YAML file will be saved.
        dataset: TorchSigIterableDataset instance to save.
    """
    yaml_dict = {}
    yaml_dict["seed"] = dataset.rng_seed
    yaml_dict["target_labels"] = dataset.target_labels
    yaml_dict["dataset_metadata"] = dataset_metadata_to_yaml_dict(dataset)
    write_dict_to_yaml(filepath, yaml_dict)


def dataset_metadata_to_yaml_dict(dataset_metadata: Any) -> dict[str, Any]:
    """Converts DatasetMetadata to a dictionary for YAML storage.

    Returns a dictionary representation of a DatasetMetadata object for storing
    as YAML.

    Args:
        dataset_metadata: DatasetMetadata object to convert.

    Returns:
        dictionary containing the metadata for YAML storage.
    """
    return dataset_metadata.get_full_metadata()



def write_dict_to_yaml(filename: str, info_dict: dict[str, Any]) -> None:
    """Writes a dictionary to a YAML file with customized settings.

    This function writes the provided `info_dict` to a YAML file. It customizes
    the representation of lists by using the `custom_representer`, and it uses
    specific formatting options (e.g., no sorting of keys, custom line width).

    Args:
        filename: The name of the YAML file to which the dictionary will be written.
        info_dict: The dictionary to be written to the YAML file.

    Returns:
        None: This function does not return any value.
    """
    yaml.add_representer(list, custom_representer)

    with open(filename, "w+") as file:
        yaml.dump(info_dict, file, default_flow_style=False, sort_keys=False, width=200)
