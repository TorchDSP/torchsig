"""Unit tests for configurable class-selection probabilities."""

from collections import Counter

import numpy as np
import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import GroupingLabel
from torchsig.utils.defaults import TorchSigDefaults


class DummySignalGenerator(BaseSignalGenerator):
    """Minimal generator used to test sampling probabilities only."""

    def __init__(self, class_name: str, **kwargs):
        super().__init__(**kwargs)
        self["class_name"] = class_name

    def generate(self) -> Signal:
        return Signal(
            data=np.ones(32, dtype=np.complex64),
            center_freq=0.0,
            bandwidth=1.0,
        )


def make_dataset() -> TorchSigIterableDataset:
    """Create a minimal dataset with no default generators."""
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata["num_iq_samples_dataset"] = 128
    metadata["fft_size"] = 32
    metadata["fft_stride"] = 32
    metadata["num_signals_min"] = 1
    metadata["num_signals_max"] = 1
    metadata["cochannel_overlap_probability"] = 0.0
    metadata["noise_power_db"] = 0.0
    metadata["snr_db_min"] = 0.0
    metadata["snr_db_max"] = 0.0
    return TorchSigIterableDataset(signal_generators=[], metadata=metadata)


def sample_generator_frequencies(
    dataset: TorchSigIterableDataset,
    num_draws: int = 1000,
) -> dict[str, float]:
    """Draw many class selections and return empirical frequencies."""
    dataset.seed(2026)
    counts = Counter(dataset._random_signal_generator().class_name for _ in range(num_draws))
    return {class_name: counts[class_name] / num_draws for class_name in dataset["class_names"]}


def test_default_add_signal_generator_probabilities_are_uniform():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "8psk"]:
        dataset.add_signal_generator(
            DummySignalGenerator(class_name),
            class_name=class_name,
        )

    expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    assert np.allclose(dataset.signal_probabilities, expected)

    empirical = sample_generator_frequencies(dataset)
    for class_name in ["bpsk", "qpsk", "8psk"]:
        assert abs(empirical[class_name] - (1 / 3)) < 0.05


def test_sampling_grouping_balances_groups_and_classes_within_groups():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "8psk", "2fsk"]:
        dataset.add_signal_generator(
            DummySignalGenerator(class_name),
            class_name=class_name,
        )
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "phase", "regex": "psk$"},
                {"name": "frequency", "values": ["2fsk"]},
            ]
        }
    )

    dataset.balance_signal_generators_by_group(grouping)

    assert np.allclose(
        dataset.signal_probabilities,
        [1 / 6, 1 / 6, 1 / 6, 1 / 2],
    )
    empirical = sample_generator_frequencies(dataset, num_draws=4_000)
    assert abs(empirical["bpsk"] + empirical["qpsk"] + empirical["8psk"] - 0.5) < 0.04
    assert abs(empirical["2fsk"] - 0.5) < 0.04


def test_sampling_grouping_uses_configured_group_probabilities():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "8psk", "2fsk"]:
        dataset.add_signal_generator(DummySignalGenerator(class_name))
    grouping = GroupingLabel(
        {
            "groups": [
                {
                    "name": "phase",
                    "regex": "psk$",
                    "probability": 0.25,
                },
                {
                    "name": "frequency",
                    "values": ["2fsk"],
                    "probability": 0.75,
                },
            ]
        }
    )

    dataset.balance_signal_generators_by_group(grouping)

    assert np.allclose(
        dataset.signal_probabilities,
        [1 / 12, 1 / 12, 1 / 12, 3 / 4],
    )


def test_sampling_grouping_defaults_missing_probability_to_zero():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "2fsk"]:
        dataset.add_signal_generator(DummySignalGenerator(class_name))

    with pytest.warns(
        UserWarning,
        match="probability is missing for group 'phase'; defaulting to 0.0",
    ):
        grouping = GroupingLabel(
            {
                "groups": [
                    {"name": "phase", "regex": "psk$"},
                    {
                        "name": "frequency",
                        "values": ["2fsk"],
                        "probability": 1.0,
                    },
                ]
            }
        )

    dataset.balance_signal_generators_by_group(grouping)

    assert grouping.groups[0]["probability"] == 0.0
    assert np.allclose(dataset.signal_probabilities, [0.0, 0.0, 1.0])


def test_sampling_grouping_normalizes_group_likelihoods():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "8psk", "2fsk"]:
        dataset.add_signal_generator(DummySignalGenerator(class_name))
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "phase", "regex": "psk$", "likelihood": 1.0},
                {
                    "name": "frequency",
                    "values": ["2fsk"],
                    "likelihood": 3.0,
                },
            ]
        }
    )

    dataset.balance_signal_generators_by_group(grouping)

    assert np.allclose(
        dataset.signal_probabilities,
        [1 / 12, 1 / 12, 1 / 12, 3 / 4],
    )


def test_sampling_grouping_defaults_missing_likelihood_to_one():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "2fsk"]:
        dataset.add_signal_generator(DummySignalGenerator(class_name))
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "phase", "regex": "psk$", "likelihood": 3.0},
                {"name": "frequency", "values": ["2fsk"]},
            ]
        }
    )

    dataset.balance_signal_generators_by_group(grouping)

    assert np.allclose(dataset.signal_probabilities, [3 / 8, 3 / 8, 1 / 4])


def test_sampling_grouping_rejects_probability_for_unrepresented_group():
    dataset = make_dataset()
    dataset.add_signal_generator(DummySignalGenerator("bpsk"))
    grouping = GroupingLabel(
        {
            "groups": [
                {
                    "name": "phase",
                    "regex": "psk$",
                    "probability": 0.75,
                },
                {
                    "name": "frequency",
                    "values": ["2fsk"],
                    "probability": 0.25,
                },
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="positive probability to groups without signal generators",
    ):
        dataset.balance_signal_generators_by_group(grouping)


def test_sampling_grouping_constructor_accepts_yaml(tmp_path):
    config_path = tmp_path / "groups.yaml"
    config_path.write_text(
        """
source: class_name
groups:
  - name: phase
    regex: 'psk$'
    probability: 0.25
  - name: frequency
    values: [2fsk]
    probability: 0.75
"""
    )
    metadata = make_dataset().metadata
    dataset = TorchSigIterableDataset(
        signal_generators=[
            DummySignalGenerator("bpsk"),
            DummySignalGenerator("qpsk"),
            DummySignalGenerator("2fsk"),
        ],
        metadata=metadata,
        sampling_grouping=config_path,
    )

    assert np.allclose(
        dataset.signal_probabilities,
        [1 / 8, 1 / 8, 3 / 4],
    )


def test_sampling_grouping_accepts_builtin_family_preset():
    metadata = make_dataset().metadata
    dataset = TorchSigIterableDataset(
        signal_generators=[
            DummySignalGenerator("bpsk"),
            DummySignalGenerator("qpsk"),
            DummySignalGenerator("2fsk"),
        ],
        metadata=metadata,
        sampling_grouping="family",
    )

    assert np.allclose(
        dataset.signal_probabilities,
        [1 / 4, 1 / 4, 1 / 2],
    )
    assert dataset.sampling_grouping.name_label == "family_name"
    assert dataset.sampling_grouping.index_label == "family_index"


def test_sampling_grouping_rejects_metadata_unavailable_before_generation():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
    )
    grouping = GroupingLabel(
        {
            "source": "bandwidth",
            "groups": [
                {"name": "narrow", "formula": "value < 1000"},
                {"name": "all", "default": True},
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="source 'bandwidth' is not available before generation",
    ):
        dataset.balance_signal_generators_by_group(grouping)


def test_sampling_grouping_rejects_dataset_without_generators():
    grouping = GroupingLabel({"groups": [{"name": "all", "default": True}]})

    with pytest.raises(
        ValueError,
        match="cannot balance groups without signal generators",
    ):
        make_dataset().balance_signal_generators_by_group(grouping)


def test_sampling_grouping_rejects_unmatched_generator():
    dataset = make_dataset()
    dataset.add_signal_generator(DummySignalGenerator("tone"))
    grouping = GroupingLabel({"groups": [{"name": "phase", "regex": "psk$"}]})

    with pytest.raises(
        ValueError,
        match="generator value 'tone'.*no group matched",
    ):
        dataset.balance_signal_generators_by_group(grouping)


def test_sampling_grouping_balances_accepted_components(monkeypatch):
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "8psk", "2fsk"]:
        dataset.add_signal_generator(DummySignalGenerator(class_name))
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "phase", "regex": "psk$"},
                {"name": "frequency", "values": ["2fsk"]},
            ]
        }
    )
    dataset.balance_signal_generators_by_group(grouping)
    monkeypatch.setattr(
        dataset,
        "_build_noise_floor",
        lambda: np.zeros(128, dtype=np.complex64),
    )
    monkeypatch.setattr(
        "torchsig.datasets.datasets.update_signal_snr_bandwidth",
        lambda _dataset, _signal: None,
    )
    monkeypatch.setattr(
        "torchsig.datasets.datasets.frequency_shift_signal",
        lambda signal, **_kwargs: signal,
    )

    counts = Counter()
    dataset.seed(2026)
    for _ in range(1_000):
        sample = next(dataset)
        group_name, _ = grouping.match(sample.component_signals[0].class_name)
        counts[group_name] += 1

    assert abs(counts["phase"] / 1_000 - 0.5) < 0.05
    assert abs(counts["frequency"] / 1_000 - 0.5) < 0.05


def test_sampling_group_is_preserved_across_overlap_retry(monkeypatch):
    dataset = make_dataset()
    dataset["num_signals_min"] = 2
    dataset["num_signals_max"] = 2
    for class_name in ["bpsk", "qpsk", "2fsk"]:
        dataset.add_signal_generator(DummySignalGenerator(class_name))
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "phase", "regex": "psk$"},
                {"name": "frequency", "values": ["2fsk"]},
            ]
        }
    )
    dataset.balance_signal_generators_by_group(grouping)
    monkeypatch.setattr(
        dataset,
        "_build_noise_floor",
        lambda: np.zeros(128, dtype=np.complex64),
    )
    monkeypatch.setattr(
        "torchsig.datasets.datasets.update_signal_snr_bandwidth",
        lambda _dataset, _signal: None,
    )
    monkeypatch.setattr(
        "torchsig.datasets.datasets.frequency_shift_signal",
        lambda signal, **_kwargs: signal,
    )
    overlap_results = iter([False, True, False])
    monkeypatch.setattr(
        dataset,
        "_check_if_overlap",
        lambda _rectangle, _rectangles: next(overlap_results),
    )
    selected_groups = []
    original_selector = dataset._random_signal_generator

    def record_selection(group_name=None):
        generator = original_selector(group_name=group_name)
        selected_group, _ = grouping.match(generator.class_name)
        selected_groups.append(selected_group)
        return generator

    monkeypatch.setattr(dataset, "_random_signal_generator", record_selection)
    dataset.seed(7)

    sample = next(dataset)

    assert len(sample.component_signals) == 2
    assert len(selected_groups) == 3
    assert selected_groups[2] == selected_groups[1]


def test_likelihoods_remain_backward_compatible():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
        likelihood=2,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("qpsk"),
        class_name="qpsk",
        likelihood=1,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("8psk"),
        class_name="8psk",
        likelihood=1,
    )

    expected = np.array([0.50, 0.25, 0.25], dtype=float)
    assert np.allclose(dataset.signal_probabilities, expected)

    empirical = sample_generator_frequencies(dataset)
    assert abs(empirical["bpsk"] - 0.50) < 0.05
    assert abs(empirical["qpsk"] - 0.25) < 0.05
    assert abs(empirical["8psk"] - 0.25) < 0.05


def test_explicit_probabilities_are_honored():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
        probability=0.60,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("qpsk"),
        class_name="qpsk",
        probability=0.25,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("8psk"),
        class_name="8psk",
        probability=0.15,
    )

    expected = np.array([0.60, 0.25, 0.15], dtype=float)
    assert np.allclose(dataset.signal_probabilities, expected)

    empirical = sample_generator_frequencies(dataset)
    assert abs(empirical["bpsk"] - 0.60) < 0.05
    assert abs(empirical["qpsk"] - 0.25) < 0.05
    assert abs(empirical["8psk"] - 0.15) < 0.05


def test_explicit_probabilities_must_sum_to_one_before_sampling():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
        probability=0.60,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("qpsk"),
        class_name="qpsk",
        probability=0.25,
    )

    with pytest.raises(ValueError, match="sum to 1.0 before sampling"):
        dataset._random_signal_generator()
