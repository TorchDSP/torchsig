from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from torchsig.signals.builder import BaseSignalGenerator, ConcatSignalGenerator
from torchsig.signals.signal_types import Signal


class DummyTransform:
    def __init__(self, key="transformed", value=True):
        self.key = key
        self.value = value

    def __call__(self, signal):
        signal[self.key] = self.value
        return signal

    def copy(self):
        return DummyTransform(self.key, self.value)


class DummySignalGenerator(BaseSignalGenerator):
    def generate(self):
        return Signal(data=np.ones(8, dtype=np.complex64))


class RequiredMetadataGenerator(BaseSignalGenerator):
    required_metadata_fields = ["sample_rate"]

    def generate(self):
        return Signal(data=np.ones(8, dtype=np.complex64))


class BadRequiredMetadataGenerator(BaseSignalGenerator):
    required_metadata_fields = ["sample_rate", 123]

    def generate(self):
        return Signal(data=np.ones(8, dtype=np.complex64))


# =============================================================================
# BaseSignalGenerator initialization
# =============================================================================


def test_base_signal_generator_defaults_to_empty_transform_list():
    generator = DummySignalGenerator()

    assert generator.transforms == []


def test_base_signal_generator_does_not_share_default_transform_list():
    generator_1 = DummySignalGenerator()
    generator_2 = DummySignalGenerator()

    generator_1.transforms.append(DummyTransform())

    assert generator_2.transforms == []


def test_base_signal_generator_preserves_provided_transforms():
    transform = DummyTransform()
    generator = DummySignalGenerator(transforms=[transform])

    assert generator.transforms == [transform]


# =============================================================================
# BaseSignalGenerator class names
# =============================================================================


def test_base_signal_generator_set_default_class_name_when_missing():
    generator = DummySignalGenerator()

    generator.set_default_class_name("bpsk")

    assert generator.class_name == "bpsk"


def test_base_signal_generator_set_default_class_name_does_not_overwrite_existing():
    generator = DummySignalGenerator(class_name="qpsk")

    generator.set_default_class_name("bpsk")

    assert generator.class_name == "qpsk"


# =============================================================================
# BaseSignalGenerator copy
# =============================================================================


def test_base_signal_generator_copy_copies_metadata_and_transforms():
    transform = DummyTransform()
    generator = DummySignalGenerator(
        transforms=[transform],
        class_name="bpsk",
        sample_rate=1000,
    )

    copied = generator.copy()

    assert copied is not generator
    assert copied.class_name == "bpsk"
    assert copied.sample_rate == 1000
    assert copied.transforms is not generator.transforms
    assert len(copied.transforms) == 1
    assert copied.transforms[0] is not transform


def test_base_signal_generator_copy_preserves_noncopyable_transforms():
    transform = lambda signal: signal
    generator = DummySignalGenerator(transforms=[transform])

    copied = generator.copy()

    assert copied.transforms == [transform]


def test_base_signal_generator_copy_preserves_parent_by_default():
    parent = DummySignalGenerator(sample_rate=1000)
    generator = DummySignalGenerator()
    generator.add_parent(parent)

    copied = generator.copy()

    assert copied.parent is parent


def test_base_signal_generator_copy_can_detach_parent():
    parent = DummySignalGenerator(sample_rate=1000)
    generator = DummySignalGenerator()
    generator.add_parent(parent)

    copied = generator.copy(preserve_parent=False)

    assert copied.parent is None


# =============================================================================
# BaseSignalGenerator validation
# =============================================================================


def test_base_signal_generator_validate_metadata_fields_no_required_fields():
    generator = DummySignalGenerator()

    assert generator.validate_metadata_fields() is None


def test_base_signal_generator_validate_metadata_fields_accepts_present_metadata():
    generator = RequiredMetadataGenerator(sample_rate=1000)

    assert generator.validate_metadata_fields() is None


def test_base_signal_generator_validate_metadata_fields_accepts_parent_metadata():
    parent = DummySignalGenerator(sample_rate=1000)
    generator = RequiredMetadataGenerator()
    generator.add_parent(parent)

    assert generator.validate_metadata_fields() is None


def test_base_signal_generator_validate_metadata_fields_rejects_missing_metadata():
    generator = RequiredMetadataGenerator()

    with pytest.raises(
        ValueError,
        match="RequiredMetadataGenerator missing required metadata key: 'sample_rate'",
    ):
        generator.validate_metadata_fields()


def test_base_signal_generator_validate_metadata_fields_rejects_non_string_field():
    generator = BadRequiredMetadataGenerator(sample_rate=1000)

    with pytest.raises(
        TypeError,
        match="all required metadata field names should be strings",
    ):
        generator.validate_metadata_fields()


# =============================================================================
# BaseSignalGenerator call
# =============================================================================


def test_base_signal_generator_call_generates_signal_and_adds_parent():
    generator = DummySignalGenerator()

    signal = generator()

    assert isinstance(signal, Signal)
    assert signal.parent is generator


def test_base_signal_generator_call_sets_class_name_on_signal():
    generator = DummySignalGenerator(class_name="bpsk")

    signal = generator()

    assert signal.class_name == "bpsk"


def test_base_signal_generator_call_applies_transforms_in_order():
    transforms = [
        DummyTransform("first", 1),
        DummyTransform("second", 2),
    ]
    generator = DummySignalGenerator(transforms=transforms)

    signal = generator()

    assert signal.first == 1
    assert signal.second == 2


def test_base_signal_generator_generate_must_be_implemented():
    generator = BaseSignalGenerator()

    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        generator.generate()


def test_base_signal_generator_repr_includes_metadata_and_transforms():
    transform = DummyTransform()
    generator = DummySignalGenerator(transforms=[transform], class_name="bpsk")

    result = repr(generator)

    assert result.startswith("DummySignalGenerator(")
    assert "metadata=" in result
    assert "class_name" in result
    assert "transforms=" in result


# =============================================================================
# ConcatSignalGenerator initialization
# =============================================================================


def test_concat_signal_generator_accepts_base_signal_generators():
    generator_1 = DummySignalGenerator(class_name="a")
    generator_2 = DummySignalGenerator(class_name="b")

    concat = ConcatSignalGenerator([generator_1, generator_2])

    assert concat.signal_generators == [generator_1, generator_2]
    assert generator_1.parent is concat
    assert generator_2.parent is concat


def test_concat_signal_generator_rejects_non_signal_generator():
    with pytest.raises(
        TypeError,
        match="signal_generator must be type BaseSignalGenerator",
    ):
        ConcatSignalGenerator([object()])


def test_concat_signal_generator_does_not_validate_children_during_init():
    generator = RequiredMetadataGenerator()

    concat = ConcatSignalGenerator([generator])

    assert generator.parent is concat


# =============================================================================
# ConcatSignalGenerator copy
# =============================================================================


def test_concat_signal_generator_copy_copies_children():
    generator_1 = DummySignalGenerator(class_name="a")
    generator_2 = DummySignalGenerator(class_name="b")
    concat = ConcatSignalGenerator([generator_1, generator_2], dataset_name="test")

    copied = concat.copy()

    assert copied is not concat
    assert copied.dataset_name == "test"
    assert copied.signal_generators is not concat.signal_generators
    assert len(copied.signal_generators) == 2
    assert copied.signal_generators[0] is not generator_1
    assert copied.signal_generators[1] is not generator_2
    assert copied.signal_generators[0].class_name == "a"
    assert copied.signal_generators[1].class_name == "b"


def test_concat_signal_generator_copy_can_detach_parent():
    parent = DummySignalGenerator(sample_rate=1000)
    child = DummySignalGenerator(class_name="a")
    concat = ConcatSignalGenerator([child])
    concat.add_parent(parent)

    copied = concat.copy(preserve_parent=False)

    assert copied.parent is None
    assert all(generator.parent is copied for generator in copied.signal_generators)


# =============================================================================
# ConcatSignalGenerator validation
# =============================================================================


def test_concat_signal_generator_validate_metadata_fields_validates_children():
    generator_1 = RequiredMetadataGenerator()
    generator_2 = RequiredMetadataGenerator()
    concat = ConcatSignalGenerator(
        [generator_1, generator_2],
        sample_rate=1000,
    )

    assert concat.validate_metadata_fields() is True


def test_concat_signal_generator_validate_metadata_fields_propagates_child_error():
    generator = RequiredMetadataGenerator()
    concat = ConcatSignalGenerator([generator])

    with pytest.raises(
        ValueError,
        match="RequiredMetadataGenerator missing required metadata key: 'sample_rate'",
    ):
        concat.validate_metadata_fields()


def test_concat_signal_generator_validate_metadata_fields_calls_each_child():
    generator_1 = DummySignalGenerator()
    generator_2 = DummySignalGenerator()

    generator_1.validate_metadata_fields = MagicMock(return_value=None)
    generator_2.validate_metadata_fields = MagicMock(return_value=None)

    concat = ConcatSignalGenerator([generator_1, generator_2])

    assert concat.validate_metadata_fields() is True
    generator_1.validate_metadata_fields.assert_called_once_with()
    generator_2.validate_metadata_fields.assert_called_once_with()


# =============================================================================
# ConcatSignalGenerator generate / repr
# =============================================================================


class DummyRNG:
    def choice(self, generators):
        return generators[1]


def test_concat_signal_generator_generate_returns_child_signal():
    generator_1 = DummySignalGenerator(class_name="a")
    generator_2 = DummySignalGenerator(class_name="b")
    concat = ConcatSignalGenerator([generator_1, generator_2])
    concat.random_generator = DummyRNG()

    signal = concat.generate()

    assert isinstance(signal, Signal)
    assert signal.class_name == "b"


def test_concat_signal_generator_repr_includes_metadata_and_children():
    generator = DummySignalGenerator(class_name="a")
    concat = ConcatSignalGenerator([generator], dataset_name="test")

    result = repr(concat)

    assert result.startswith("ConcatSignalGenerator(")
    assert "metadata=" in result
    assert "dataset_name" in result
    assert "signal_generators=" in result
