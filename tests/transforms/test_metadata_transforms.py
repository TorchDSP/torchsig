"""Tests for metadata transforms."""

import json

import numpy as np
import pytest
import yaml

from torchsig.datasets.datasets import apply_transforms_and_labels_to_signal
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, FAMILY_SHARED_LIST
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import (
    GroupingLabel,
    MetadataTransform,
    MultiHotLabel,
    YOLOLabel,
)


class DummyMetadataTransform(MetadataTransform):
    """Concrete metadata transform for testing base behavior."""

    def __init__(self, required_metadata=None):
        super().__init__(required_metadata=required_metadata or [])
        self.applied_signals = []

    def __apply__(self, signal):
        self.applied_signals.append(signal)
        signal["was_applied"] = True
        return signal


@pytest.fixture
def component_signal():
    return Signal(
        data=np.zeros(1000, dtype=np.complex64),
        class_index=3,
        start_in_samples=250,
        duration_in_samples=500,
        num_iq_samples_dataset=1000,
        bandwidth=200.0,
        estimated_occupied_bandwidth=100.0,
        center_freq=100.0,
        sample_rate=1000.0,
        dataset_metadata={"sample_rate": 1000.0},
    )


@pytest.fixture
def parent_signal(component_signal):
    """Create a parent Signal containing one component signal."""
    return Signal(
        data=None,
        component_signals=[component_signal],
        sample_rate=1000.0,
        dataset_metadata={"sample_rate": 1000.0},
    )


def test_metadata_transform_validate_accepts_signal_with_required_metadata(component_signal):
    transform = DummyMetadataTransform(required_metadata=["class_index", "start"])

    validated_signal = transform.__validate__(component_signal)

    assert validated_signal is component_signal


def test_metadata_transform_validate_rejects_non_signal():
    transform = DummyMetadataTransform()

    with pytest.raises(TypeError, match="is not a Signal object"):
        transform.__validate__({"class_index": 1})


def test_metadata_transform_validate_rejects_missing_required_metadata(component_signal):
    transform = DummyMetadataTransform(required_metadata=["missing_field"])

    with pytest.raises(
        ValueError,
        match="key: missing_field is missing from signal metadata",
    ):
        transform.__validate__(component_signal)


def test_metadata_transform_call_applies_to_each_component_signal(parent_signal):
    transform = DummyMetadataTransform()

    transformed_signal = transform(parent_signal)

    assert transformed_signal is parent_signal
    assert len(transform.applied_signals) == 1
    assert transform.applied_signals[0] is parent_signal.component_signals[0]
    assert parent_signal.component_signals[0].was_applied is True


def test_metadata_transform_call_applies_to_signal_without_components(
    component_signal,
):
    transform = DummyMetadataTransform()

    transformed_signal = transform(component_signal)

    assert transformed_signal is component_signal
    assert component_signal.was_applied is True


def test_metadata_transform_base_apply_raises_not_implemented(component_signal):
    transform = MetadataTransform()

    with pytest.raises(NotImplementedError):
        transform.__apply__(component_signal)


def test_metadata_transform_repr_excludes_required_metadata():
    transform = DummyMetadataTransform(required_metadata=["class_index"])

    repr_str = repr(transform)

    assert "DummyMetadataTransform" in repr_str
    assert "required_metadata" not in repr_str


def test_multi_hot_label_encodes_unique_component_classes():
    signal = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(class_index=1),
            Signal(class_index=3),
            Signal(class_index=1),
        ],
        class_names=["a", "b", "c", "d"],
    )

    transformed = MultiHotLabel()(signal)

    np.testing.assert_array_equal(
        transformed.multi_hot_label,
        np.array([0, 1, 0, 1], dtype=np.float32),
    )


def test_multi_hot_label_supports_leaf_and_explicit_class_count():
    signal = Signal(class_index=np.int64(2))

    transformed = MultiHotLabel(num_classes=4, output_key="classes")(signal)

    np.testing.assert_array_equal(
        transformed.classes,
        np.array([0, 0, 1, 0], dtype=np.float32),
    )


def test_multi_hot_label_encodes_empty_composite_as_all_zero():
    signal = Signal(class_names=["a", "b", "c"])

    transformed = MultiHotLabel()(signal)

    np.testing.assert_array_equal(
        transformed.multi_hot_label,
        np.zeros(3, dtype=np.float32),
    )


@pytest.mark.parametrize("num_classes", [0, -1, 2.5, True])
def test_multi_hot_label_rejects_invalid_class_count(num_classes):
    with pytest.raises(ValueError, match="positive integer"):
        MultiHotLabel(num_classes=num_classes)


def test_multi_hot_label_requires_class_names_when_count_is_not_given():
    with pytest.raises(ValueError, match="class_names is missing"):
        MultiHotLabel()(Signal(class_index=0))


@pytest.mark.parametrize("class_index", [-1, 3])
def test_multi_hot_label_rejects_out_of_range_class_index(class_index):
    signal = Signal(class_index=class_index)

    with pytest.raises(ValueError, match="outside"):
        MultiHotLabel(num_classes=3)(signal)


def test_multi_hot_label_rejects_non_integer_class_index():
    signal = Signal(class_index=1.5)

    with pytest.raises(TypeError, match="must be an integer"):
        MultiHotLabel(num_classes=3)(signal)


def test_yolo_label_initializes_expected_metadata_fields():
    transform = YOLOLabel()

    assert transform.required_metadata == [
        "class_index",
        "start",
        "estimated_occupied_bandwidth",
        "center_freq",
        "dataset_metadata",
    ]
    assert transform.bandwidth_key == "estimated_occupied_bandwidth"
    assert transform.class_index_field == "class_index"
    assert transform.targets_metadata == ["yolo_label"]


def test_yolo_label_adds_expected_label_to_component_signal(parent_signal):
    transform = YOLOLabel()

    transformed_signal = transform(parent_signal)

    component_signal = transformed_signal.component_signals[0]
    assert component_signal.yolo_label == pytest.approx(
        (
            3,  # class_index
            0.5,  # start + duration / 2
            0.4,  # 1 - ((sample_rate / 2 + center_freq) / sample_rate)
            0.5,  # duration
            0.1,  # estimated_occupied_bandwidth / sample_rate
        )
    )


def test_yolo_label_apply_returns_component_signal(component_signal):
    transform = YOLOLabel()

    transformed_component = transform.__apply__(component_signal)

    assert transformed_component is component_signal
    assert transformed_component.yolo_label == pytest.approx((3, 0.5, 0.4, 0.5, 0.1))


def test_yolo_label_can_use_canonical_bandwidth(component_signal):
    transform = YOLOLabel(bandwidth_key="bandwidth")

    transformed_component = transform(component_signal)

    assert transform.required_metadata[2] == "bandwidth"
    assert transformed_component.yolo_label == pytest.approx((3, 0.5, 0.4, 0.5, 0.2))


def test_yolo_label_rejects_invalid_bandwidth_key():
    with pytest.raises(ValueError, match="bandwidth_key must be"):
        YOLOLabel(bandwidth_key="occupied_bandwidth")


@pytest.mark.parametrize("class_index_field", ["", None, 3])
def test_yolo_label_rejects_invalid_class_index_field(class_index_field):
    with pytest.raises(ValueError, match="class_index_field must be"):
        YOLOLabel(class_index_field=class_index_field)


def test_yolo_label_requires_selected_bandwidth(component_signal):
    del component_signal["estimated_occupied_bandwidth"]

    with pytest.raises(
        ValueError,
        match="estimated_occupied_bandwidth is missing",
    ):
        YOLOLabel()(component_signal)


def test_yolo_label_can_use_family_index_from_grouping(parent_signal):
    component = parent_signal.component_signals[0]
    component["class_name"] = "bpsk"

    GroupingLabel("family")(parent_signal)
    canonical_label = YOLOLabel()(component).yolo_label
    family_label = YOLOLabel(class_index_field="family_index")(component).yolo_label

    assert family_label[0] == FAMILY_SHARED_LIST.index(CLASS_FAMILY_DICT["bpsk"])
    assert family_label[1:] == pytest.approx(canonical_label[1:])


def test_yolo_label_can_use_custom_group_index(parent_signal):
    component = parent_signal.component_signals[0]
    component["class_name"] = "bpsk"
    grouping = GroupingLabel(
        {
            "labels": {"name": "kind", "index": "kind_index"},
            "groups": [
                {"name": "frequency", "values": ["2fsk"]},
                {"name": "phase", "values": ["bpsk"]},
            ],
        }
    )

    grouping(parent_signal)
    transformed = YOLOLabel(class_index_field="kind_index")(parent_signal)

    assert transformed.component_signals[0].yolo_label == pytest.approx((1, 0.5, 0.4, 0.5, 0.1))


def test_yolo_label_requires_selected_class_index_field(component_signal):
    with pytest.raises(
        ValueError,
        match="key: family_index is missing from signal metadata",
    ):
        YOLOLabel(class_index_field="family_index")(component_signal)


def test_grouping_label_uses_exact_value_rules(parent_signal):
    parent_signal.component_signals[0]["class_name"] = "bpsk"
    transform = GroupingLabel(
        {
            "source": "class_name",
            "labels": {
                "name": "modulation_group",
                "index": "modulation_group_index",
            },
            "groups": [
                {"name": "frequency", "values": ["2fsk"]},
                {"name": "linear", "values": ["bpsk", "qpsk"]},
            ],
        }
    )

    transformed_signal = transform(parent_signal)
    component = transformed_signal.component_signals[0]

    assert component.modulation_group == "linear"
    assert component.modulation_group_index == 1
    assert transform.targets_metadata == [
        "modulation_group",
        "modulation_group_index",
    ]


def test_grouping_label_loads_yaml_and_supports_regex_and_formula(
    tmp_path,
):
    config_path = tmp_path / "groups.yaml"
    config_path.write_text(
        """
source: class_name
groups:
  - name: frequency_shift
    regex: '^[248]g?fsk$'
  - name: high_order_qam
    formula: 'value.endswith("qam") and value != "16qam"'
"""
    )
    transform = GroupingLabel(config_path)

    fsk_signal = transform(Signal(class_name="4gfsk"))
    qam_signal = transform(Signal(class_name="64qam"))

    assert fsk_signal.group_name == "frequency_shift"
    assert fsk_signal.group_index == 0
    assert qam_signal.group_name == "high_order_qam"
    assert qam_signal.group_index == 1


def test_grouping_label_supports_arithmetic_formula_and_default():
    transform = GroupingLabel(
        {
            "source": "bandwidth",
            "groups": [
                {
                    "name": "narrow",
                    "formula": "value / 1000 < 2",
                },
                {
                    "name": "even_khz",
                    "formula": "(value // 1000) % 2 == 0",
                },
                {"name": "all", "default": True},
            ],
        }
    )

    assert transform(Signal(bandwidth=1_000)).group_name == "narrow"
    assert transform(Signal(bandwidth=4_000)).group_name == "even_khz"
    fallback = transform(Signal(bandwidth=7_000))
    assert fallback.group_name == "all"
    assert fallback.group_index == 2


def test_grouping_label_match_does_not_require_a_signal():
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "linear", "values": ["bpsk", "qpsk"]},
                {"name": "all", "default": True},
            ]
        }
    )

    assert grouping.match("qpsk") == ("linear", 0)
    assert grouping.match("tone") == ("all", 1)


def test_grouping_label_family_preset_uses_canonical_mapping():
    grouping = GroupingLabel("family")

    assert grouping.source == "class_name"
    assert grouping.targets_metadata == ["family_name", "family_index"]
    assert GroupingLabel.available_presets() == ("family",)
    for class_name, expected_family in CLASS_FAMILY_DICT.items():
        family_name, family_index = grouping.match(class_name)
        assert family_name == expected_family
        assert family_index == FAMILY_SHARED_LIST.index(expected_family)


def test_grouping_label_queries_family_preset():
    grouping = GroupingLabel("family")

    assert grouping.group_names == tuple(FAMILY_SHARED_LIST)
    assert grouping.group_list == FAMILY_SHARED_LIST
    assert grouping.values_for("psk") == tuple(sorted(class_name for class_name, family in CLASS_FAMILY_DICT.items() if family == "psk"))
    assert grouping.all_values == tuple(value for group_name in grouping.group_names for value in grouping.values_for(group_name))


def test_grouping_label_family_preset_serialization_round_trip():
    grouping = GroupingLabel("family")

    config = grouping.to_dict()
    restored = GroupingLabel(config)

    assert json.loads(json.dumps(config)) == config
    assert yaml.safe_load(yaml.safe_dump(config, sort_keys=False)) == config
    assert restored.group_list == grouping.group_list
    for value in grouping.all_values:
        assert restored.match(value) == grouping.match(value)


def test_grouping_label_queries_mapping_without_exposing_mutable_values():
    grouping = GroupingLabel(
        {
            "groups": [
                {"name": "linear", "values": ["bpsk", "qpsk"]},
                {"name": "frequency", "regex": "fsk$"},
                {"name": "all", "default": True},
            ],
        }
    )

    assert grouping.group_names == ("linear", "frequency", "all")
    assert grouping.values_for("linear") == ("bpsk", "qpsk")
    assert grouping.values_for("frequency") == ()
    assert grouping.values_for("all") == ()
    assert grouping.all_values == ("bpsk", "qpsk")


def test_grouping_label_mapping_serialization_is_defensive():
    grouping = GroupingLabel(
        {
            "source": "class_name",
            "labels": {"name": "kind", "index": "kind_index"},
            "groups": [
                {"name": "linear", "values": ["bpsk", "qpsk"], "likelihood": 2.0},
                {"name": "frequency", "regex": "fsk$", "likelihood": 1.0},
                {"name": "all", "default": True, "likelihood": 1.0},
            ],
        }
    )

    config = grouping.to_dict()
    restored = GroupingLabel(config)

    assert config == {
        "source": "class_name",
        "labels": {"name": "kind", "index": "kind_index"},
        "groups": [
            {"name": "linear", "values": ["bpsk", "qpsk"], "likelihood": 2.0},
            {"name": "frequency", "regex": "fsk$", "likelihood": 1.0},
            {"name": "all", "default": True, "likelihood": 1.0},
        ],
    }
    for value in ("bpsk", "2fsk", "tone"):
        assert restored.match(value) == grouping.match(value)

    config["labels"]["name"] = "changed"
    config["groups"][0]["name"] = "changed"
    config["groups"][0]["values"].append("8psk")
    names = grouping.group_list
    names.append("changed")

    assert grouping.name_label == "kind"
    assert grouping.group_list == ["linear", "frequency", "all"]
    assert grouping.values_for("linear") == ("bpsk", "qpsk")


def test_grouping_label_queries_yaml_config(tmp_path):
    config_path = tmp_path / "groups.yaml"
    config_path.write_text(
        """
groups:
  - name: first
    values: [one, two]
  - name: second
    values: [three]
"""
    )

    grouping = GroupingLabel(config_path)

    assert grouping.group_names == ("first", "second")
    assert grouping.group_list == ["first", "second"]
    assert grouping.values_for("second") == ("three",)
    assert grouping.all_values == ("one", "two", "three")

    restored = GroupingLabel(grouping.to_dict())
    for value in ("one", "two", "three"):
        assert restored.match(value) == grouping.match(value)


def test_grouping_label_values_for_rejects_unknown_group():
    grouping = GroupingLabel({"groups": [{"name": "linear", "values": ["bpsk"]}]})

    with pytest.raises(KeyError, match=r"unknown grouping name 'missing'.*linear"):
        grouping.values_for("missing")


def test_grouping_label_family_preset_returns_independent_configs():
    first = GroupingLabel("family")
    second = GroupingLabel("family")

    first.groups[0]["values"].append("custom-class")

    assert "custom-class" not in second.groups[0]["values"]


def test_grouping_label_values_are_available_as_dataset_targets(parent_signal):
    parent_signal["num_signals_max"] = 2
    parent_signal.component_signals[0]["class_name"] = "bpsk"
    transform = GroupingLabel(
        {
            "groups": [
                {"name": "linear", "values": ["bpsk"]},
            ],
        }
    )

    _, targets = apply_transforms_and_labels_to_signal(
        parent_signal,
        transforms=[transform],
        target_labels=["group_name", "group_index"],
    )

    assert targets == [["linear"], [0]]


def test_grouping_label_rejects_unmatched_value():
    transform = GroupingLabel({"groups": [{"name": "linear", "values": ["bpsk"]}]})

    with pytest.raises(
        ValueError,
        match=r"'tone'.*did not match any configured group",
    ):
        transform(Signal(class_name="tone"))


def test_grouping_label_rejects_mixed_sampling_weights():
    config = {
        "groups": [
            {"name": "phase", "regex": "psk$", "probability": 0.5},
            {"name": "frequency", "regex": "fsk$", "likelihood": 1.0},
        ]
    }

    with pytest.raises(
        ValueError,
        match="cannot mix probability and likelihood",
    ):
        GroupingLabel(config)


@pytest.mark.parametrize(
    ("groups", "expected_message"),
    [
        (
            [
                {"name": "phase", "regex": "psk$", "probability": 0.4},
                {"name": "frequency", "regex": "fsk$", "probability": 0.4},
            ],
            "group probabilities must sum to 1.0",
        ),
        (
            [{"name": "phase", "regex": "psk$", "probability": -1.0}],
            "probability must be >= 0",
        ),
        (
            [{"name": "phase", "regex": "psk$", "likelihood": 0.0}],
            "likelihood must be > 0",
        ),
    ],
)
def test_grouping_label_rejects_invalid_sampling_weights(
    groups,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        GroupingLabel({"groups": groups})


@pytest.mark.parametrize(
    ("config", "expected_message"),
    [
        (
            {"groups": []},
            "groups must be a non-empty list",
        ),
        (
            {
                "groups": [
                    {
                        "name": "linear",
                        "values": ["bpsk"],
                        "regex": "psk$",
                    }
                ]
            },
            "must define exactly one",
        ),
        (
            {
                "groups": [
                    {"name": "all", "default": True},
                    {"name": "linear", "values": ["bpsk"]},
                ]
            },
            "default group must be last",
        ),
        (
            {
                "groups": [
                    {"name": "all", "default": False},
                ]
            },
            "default rule must be true",
        ),
        (
            {
                "groups": [
                    {"name": "linear", "formula": "__import__('os')"},
                ]
            },
            "may only call safe string methods",
        ),
        (
            {
                "groups": [
                    {"name": "linear", "formula": "value.__class__"},
                ]
            },
            "method '__class__' is not allowed",
        ),
    ],
)
def test_grouping_label_rejects_invalid_config(
    config,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        GroupingLabel(config)
