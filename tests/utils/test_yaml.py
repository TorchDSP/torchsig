
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.yaml import (
    write_dict_to_yaml,
    dataset_metadata_from_yaml_dict,
    load_dataset_yaml,
    save_dataset_yaml,
    dataset_metadata_to_yaml_dict
)

import os
from contextlib import redirect_stdout
import pytest
import yaml

def dataset_from_yaml_dict(yaml_dict):
    # Create DatasetMetadata object from the yaml_dict content
    metadata_params = yaml_dict['dataset_metadata']
    dataset_metadata = DatasetMetadata(
        num_iq_samples_dataset=metadata_params['num_iq_samples_dataset'],
        fft_size=metadata_params['fft_size'],
        num_signals_min=metadata_params['num_signals_min'],
        num_signals_max=metadata_params['num_signals_max'],
        sample_rate=metadata_params['sample_rate'],
        num_signals_distribution=metadata_params['num_signals_distribution'],
        snr_db_min=metadata_params['snr_db_min'],
        snr_db_max=metadata_params['snr_db_max'],
        signal_duration_min=metadata_params['signal_duration_min'],
        signal_duration_max=metadata_params['signal_duration_max'],
        signal_bandwidth_min=metadata_params['signal_bandwidth_min'],
        signal_bandwidth_max=metadata_params['signal_bandwidth_max'],
        signal_center_freq_min=metadata_params['signal_center_freq_min'],
        signal_center_freq_max=metadata_params['signal_center_freq_max'],
        cochannel_overlap_probability=metadata_params['cochannel_overlap_probability'],
        class_list=metadata_params['class_list'],
        class_distribution=metadata_params['class_distribution']
    )

    # Instantiate the dataset
    dataset = TorchSigIterableDataset(
        seed=yaml_dict['seed'],
        target_labels=yaml_dict['target_labels'],
        dataset_metadata=dataset_metadata
    )

    return dataset

def test_load_dataset_yaml(tmp_path):
    # Create a temporary YAML file
    temp_file = tmp_path / "dataset.yaml"

    # Example content for YAML
    yaml_content = {
        "seed": 42,
        "target_labels": ["label1", "label2"],
        "dataset_metadata": {
            "num_iq_samples_dataset": 1000,
            "fft_size": 2048,
            "num_signals_min": 1,
            "num_signals_max": 10,
            "sample_rate": 12000,
            "num_signals_distribution": "uniform",
            "snr_db_min": 0,
            "snr_db_max": 30,
            "signal_duration_min": 0.001,
            "signal_duration_max": 0.01,
            "signal_bandwidth_min": 100,
            "signal_bandwidth_max": 200,
            "signal_center_freq_min": 2.4,
            "signal_center_freq_max": 2.5,
            "cochannel_overlap_probability": 0.1,
            "class_list": ["class1", "class2"],
            "class_distribution": [0.5, 0.5]
        }
    }

    # Write the YAML data to the temporary file
    with open(temp_file, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file)

    # Load the YAML and create the dataset
    dataset = load_dataset_yaml(str(temp_file))

    # Verify that the dataset is constructed correctly
    assert dataset.rng_seed == 42
    assert dataset.target_labels == ["label1", "label2"]
    assert dataset.dataset_metadata.num_iq_samples_dataset == 1000
    assert dataset.dataset_metadata.fft_size == 2048
    assert dataset.dataset_metadata.num_signals_min == 1
    assert dataset.dataset_metadata.num_signals_max == 10
    assert dataset.dataset_metadata.sample_rate == 12000
    assert dataset.dataset_metadata.num_signals_distribution == "uniform"
    assert dataset.dataset_metadata.snr_db_min == 0
    assert dataset.dataset_metadata.snr_db_max == 30
    assert dataset.dataset_metadata.signal_duration_min == 0.001
    assert dataset.dataset_metadata.signal_duration_max == 0.01
    assert dataset.dataset_metadata.signal_bandwidth_min == 100
    assert dataset.dataset_metadata.signal_bandwidth_max == 200
    assert dataset.dataset_metadata.signal_center_freq_min == 2.4
    assert dataset.dataset_metadata.signal_center_freq_max == 2.5
    assert dataset.dataset_metadata.cochannel_overlap_probability == 0.1
    assert dataset.dataset_metadata.class_list == ["class1", "class2"]
    assert dataset.dataset_metadata.class_distribution == [0.5, 0.5]

def test_save_dataset_yaml(tmp_path, capsys):
    # Create a temporary file path
    temp_file = tmp_path / "dataset.yaml"

    # DatasetMetadata instance
    dataset_metadata = DatasetMetadata(
        num_iq_samples_dataset=1000,
        fft_size=2048,
        num_signals_min=1,
        num_signals_max=10,
        sample_rate=12000,
        num_signals_distribution="uniform",
        snr_db_min=0,
        snr_db_max=30,
        signal_duration_min=0.001,
        signal_duration_max=0.01,
        signal_bandwidth_min=100,
        signal_bandwidth_max=200,
        signal_center_freq_min=2.4,
        signal_center_freq_max=2.5,
        cochannel_overlap_probability=0.1,
        class_list=["class1", "class2"],
        class_distribution=[0.5, 0.5]
    )

    # TorchSigIterableDataset instance
    dataset = TorchSigIterableDataset(
        seed=42,
        target_labels=["label1", "label2"],
        dataset_metadata=dataset_metadata
    )

    # Call to save_dataset_yaml
    save_dataset_yaml(str(temp_file), dataset)

    # Load the written YAML file and compare
    with open(temp_file, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    expected_yaml_dict = {
        "seed": 42,
        "target_labels": ["label1", "label2"],
        "dataset_metadata": dataset_metadata_to_yaml_dict(dataset_metadata)
    }

    assert yaml_dict == expected_yaml_dict, "The YAML dictionary does not match the expected dictionary."


def test_dataset_metadata_to_yaml_dict():
    # Create a test instance of DatasetMetadata
    test_dataset_metadata = DatasetMetadata(
        num_iq_samples_dataset=1000,
        fft_size=2048,
        num_signals_min=1,
        num_signals_max=10,
        sample_rate=12000,
        num_signals_distribution="uniform",
        snr_db_min=0,
        snr_db_max=30,
        signal_duration_min=0.001,
        signal_duration_max=0.01,
        signal_bandwidth_min=100,
        signal_bandwidth_max=200,
        signal_center_freq_min=2.4,
        signal_center_freq_max=2.5,
        cochannel_overlap_probability=0.1,
        class_list=["class1", "class2"],
        class_distribution=[0.5, 0.5]
    )

    expected_yaml_dict = {
        "num_iq_samples_dataset": 1000,
        "fft_size": 2048,
        "num_signals_min": 1,
        "num_signals_max": 10,
        "sample_rate": 12000,
        "num_signals_distribution": "uniform",
        "snr_db_min": 0,
        "snr_db_max": 30,
        "signal_duration_min": 0.001,
        "signal_duration_max": 0.01,
        "signal_bandwidth_min": 100,
        "signal_bandwidth_max": 200,
        "signal_center_freq_min": 2.4,
        "signal_center_freq_max": 2.5,
        "cochannel_overlap_probability": 0.1,
        "class_list": ["class1", "class2"],
        "class_distribution": [0.5, 0.5]
    }

    result = dataset_metadata_to_yaml_dict(test_dataset_metadata)

    assert result == expected_yaml_dict