"""Unit tests for TorchSig default dataset configurations.
Tests cover config loading, validation, and DataModule creation.
"""

from pathlib import Path

import pytest

from torchsig.datasets import TorchSigDataModule
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.file_handlers import (
    PackedHDF5Reader,
    PackedHDF5Writer,
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.yaml import load_config_from_yaml

# Path to default configs directory
CONFIGS_DIR = Path(__file__).parent.parent.parent / "torchsig" / "datasets" / "default_configs"

# List of all default config files
DEFAULT_CONFIGS = [
    "narrowband_clean_train_all.yaml",
    "narrowband_clean_val_all.yaml",
    "narrowband_impaired_train_all.yaml",
    "narrowband_impaired_val_all.yaml",
    "narrowband_toy_dataset.yaml",
    "wideband_clean_train_all.yaml",
    "wideband_clean_val_all.yaml",
    "wideband_impaired_train_all.yaml",
    "wideband_impaired_val_all.yaml",
]


@pytest.fixture(params=DEFAULT_CONFIGS)
def config_file(request):
    """Fixture that provides path to each default config file."""
    return CONFIGS_DIR / request.param


class TestConfigLoaders:
    """Test loading and validation of YAML config files."""

    def test_config_file_exists(self, config_file):
        """Verify all default config files exist."""
        assert config_file.exists(), f"Missing config file: {config_file}"
        assert config_file.is_file()

    def test_load_config(self, config_file):
        """Test that each config can be loaded into TorchSigDatasetConfig."""
        config = load_config_from_yaml(config_file)
        assert isinstance(config, TorchSigDatasetConfig), f"Config {config_file.name} did not load as TorchSigDatasetConfig"

    def test_config_required_fields(self, config_file):
        """Verify all configs have required fields with correct types."""
        config = load_config_from_yaml(config_file)

        # Required fields and their expected types
        required_fields = {
            "dataset_id": str,
            "dataset_length": int,
            "seed": int,
            "impairment_level": int,
            "output_representation": str,
            "signal_sampling_mode": str,
            "dataset_metadata": dict,
            "output_spectrogram_fft": (int, type(None)),
        }

        for field, expected_type in required_fields.items():
            assert hasattr(config, field), f"Missing field: {field} in {config_file.name}"
            value = getattr(config, field)
            assert isinstance(value, expected_type), f"Field {field} has wrong type. Expected {expected_type}, got {type(value)}"

        # Validate enum fields
        assert config.output_representation in ["iq", "spectrogram"]
        assert config.signal_sampling_mode in ["per_signal", "per_family"]

        # Validate metadata
        metadata = config.dataset_metadata
        assert "sample_rate" in metadata, "Missing sample_rate in dataset_metadata"
        assert "num_iq_samples_dataset" in metadata, "Missing num_iq_samples_dataset in dataset_metadata"


class TestConfigProperties:
    """Test specific properties of known configurations."""

    @pytest.mark.parametrize(
        ("config_file", "expected"),
        [
            ("narrowband_clean_train_all.yaml", ("iq", 1_140_000, 0)),
            ("narrowband_clean_val_all.yaml", ("iq", 11_400, 0)),
            ("wideband_clean_train_all.yaml", ("spectrogram", 57_000, 0)),
            ("wideband_clean_val_all.yaml", ("spectrogram", 11_400, 0)),
            ("narrowband_toy_dataset.yaml", ("iq", 100, 0)),
            ("narrowband_impaired_train_all.yaml", ("iq", 5_700_000, 2)),
            ("wideband_impaired_train_all.yaml", ("spectrogram", 57_000, 2)),
        ],
    )
    def test_known_config_properties(self, config_file, expected):
        """Test expected properties of specific configs."""
        config = load_config_from_yaml(CONFIGS_DIR / config_file)
        assert config.output_representation == expected[0]
        assert config.dataset_length == expected[1]
        assert config.impairment_level == expected[2]

    def test_clean_configs_have_zero_impairment(self):
        """Test all clean configs have impairment_level=0."""
        clean_configs = ["narrowband_clean_train_all.yaml", "narrowband_clean_val_all.yaml", "wideband_clean_train_all.yaml", "wideband_clean_val_all.yaml", "narrowband_toy_dataset.yaml"]
        for config_file in clean_configs:
            config = load_config_from_yaml(CONFIGS_DIR / config_file)
            assert config.impairment_level == 0, f"{config_file} should have impairment_level=0"

    def test_impaired_configs_have_positive_impairment(self):
        """Test all impaired configs have impairment_level>0."""
        impaired_configs = ["narrowband_impaired_train_all.yaml", "narrowband_impaired_val_all.yaml", "wideband_impaired_train_all.yaml", "wideband_impaired_val_all.yaml"]
        for config_file in impaired_configs:
            config = load_config_from_yaml(CONFIGS_DIR / config_file)
            assert config.impairment_level > 0, f"{config_file} should have impairment_level>0"


class TestDataModuleCreation:
    """Test creating TorchSigDataModule from configs."""

    def test_datamodule_creation(self, config_file, tmp_path):
        """Test creating DataModule from each config."""
        config = load_config_from_yaml(config_file)
        dataset_root = tmp_path / f"test_{config.dataset_id}"

        dm = TorchSigDataModule.from_config(
            config,
            root=dataset_root,
            dataset_size=100,  # Use small size for testing
            overwrite=True,
        )

        assert dm.dataset_size == 100
        assert dm.seed == config.seed
        assert dm.impairment_level == config.impairment_level
        assert isinstance(dm.root, Path)
        assert dm.root == dataset_root

    def test_from_config_forwards_packed_writer_options(self, tmp_path):
        """Packed writer configuration survives the config constructor."""
        config = load_config_from_yaml(CONFIGS_DIR / "narrowband_toy_dataset.yaml")
        options = {
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
        }

        dm = TorchSigDataModule.from_config(
            config,
            root=tmp_path,
            file_writer=PackedHDF5Writer,
            file_writer_kwargs=options,
        )
        options["compression"] = "lzf"

        assert dm.file_reader is PackedHDF5Reader
        assert dm.file_writer_kwargs == {
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
        }

    def test_from_config_forwards_homogeneous_writer_options(
        self,
        tmp_path,
    ):
        config = load_config_from_yaml(CONFIGS_DIR / "narrowband_toy_dataset.yaml")
        options = {
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
            "chunk_samples": 4,
        }

        dm = TorchSigDataModule.from_config(
            config,
            root=tmp_path,
            file_writer=HomogeneousHDF5Writer,
            file_writer_kwargs=options,
        )
        options["chunk_samples"] = 8

        assert dm.file_reader is HomogeneousHDF5Reader
        assert dm.file_writer_kwargs["chunk_samples"] == 4

    def test_narrowband_transforms(self, tmp_path):
        """Test narrowband configs produce expected transforms."""
        config = load_config_from_yaml(CONFIGS_DIR / "narrowband_clean_train_all.yaml")
        dm = TorchSigDataModule.from_config(config, root=tmp_path / "test_narrowband", dataset_size=100, overwrite=True)

        transform_classes = [t.__class__ for t in dm.transforms]
        assert ComplexTo2D in transform_classes, "Narrowband config should include ComplexTo2D"
        assert Spectrogram not in transform_classes, "Narrowband config should not include Spectrogram"

    def test_wideband_transforms(self, tmp_path):
        """Test wideband configs produce expected transforms."""
        config = load_config_from_yaml(CONFIGS_DIR / "wideband_clean_train_all.yaml")
        dm = TorchSigDataModule.from_config(config, root=tmp_path / "test_wideband", dataset_size=100, overwrite=True)

        transform_classes = [t.__class__ for t in dm.transforms]
        assert Spectrogram in transform_classes, "Wideband config should include Spectrogram"
        assert ComplexTo2D not in transform_classes, "Wideband config should not include ComplexTo2D"


class TestYOLOLabelBehavior:
    """Test YOLOLabel behavior for spectrogram outputs."""

    def test_default_yolo_label(self, tmp_path):
        """Test that wideband configs include YOLOLabel by default."""
        config = load_config_from_yaml(CONFIGS_DIR / "wideband_clean_train_all.yaml")
        dm = TorchSigDataModule.from_config(config, root=tmp_path / "test_default_yolo", dataset_size=100, overwrite=True)

        transform_classes = [t.__class__ for t in dm.transforms]
        assert YOLOLabel in transform_classes, "Wideband should include YOLOLabel by default"
        assert "yolo_label" in dm.target_labels, "yolo_label should be in target_labels by default"

    def test_disable_yolo_label(self, tmp_path):
        """Test that YOLOLabel can be disabled."""
        config = load_config_from_yaml(CONFIGS_DIR / "wideband_clean_train_all.yaml")
        dm = TorchSigDataModule.from_config(
            config,
            root=tmp_path / "test_no_yolo",
            dataset_size=100,
            overwrite=True,
            target_labels=[],  # Explicitly disable YOLO
        )

        transform_classes = [t.__class__ for t in dm.transforms]
        assert YOLOLabel not in transform_classes, "YOLOLabel should be disabled"
        print(dm.target_labels)
        assert dm.target_labels is None, "target_labels should be empty"

    def test_custom_target_labels_with_yolo(self, tmp_path):
        """Test custom target labels including yolo_label."""
        config = load_config_from_yaml(CONFIGS_DIR / "wideband_clean_train_all.yaml")
        dm = TorchSigDataModule.from_config(config, root=tmp_path / "test_custom_with_yolo", dataset_size=100, overwrite=True, target_labels=["yolo_label", "snr"])

        transform_classes = [t.__class__ for t in dm.transforms]
        assert YOLOLabel in transform_classes, "YOLOLabel should be present"
        assert "yolo_label" in dm.target_labels, "yolo_label should be in target_labels"
        assert "snr" in dm.target_labels, "snr should be in target_labels"

    def test_narrowband_ignores_yolo_setting(self, tmp_path):
        """Test that narrowband configs ignore yolo_label in target_labels."""
        config = load_config_from_yaml(CONFIGS_DIR / "narrowband_clean_train_all.yaml")
        dm = TorchSigDataModule.from_config(
            config,
            root=tmp_path / "test_narrowband_yolo",
            dataset_size=100,
            overwrite=True,
            target_labels=["yolo_label"],  # This should be ignored for IQ output
        )

        transform_classes = [t.__class__ for t in dm.transforms]
        assert YOLOLabel not in transform_classes, "Narrowband should not include YOLOLabel"
        # The target_labels might still contain it, but it's not used
