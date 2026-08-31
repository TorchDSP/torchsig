import numpy as np
import pytest

from torchsig.signals.signal_types import Signal, SignalMetadataObject


def test_signal_metadata_start_getter_and_setter():
    metadata = SignalMetadataObject(
        num_iq_samples_dataset=1000,
        start_in_samples=250,
    )

    assert metadata.start == pytest.approx(0.25)

    metadata.start = 0.5

    assert metadata.start_in_samples == 500
    assert metadata.start == pytest.approx(0.5)


def test_signal_metadata_stop_getter_and_setter():
    metadata = SignalMetadataObject(
        num_iq_samples_dataset=1000,
        start_in_samples=200,
        duration_in_samples=300,
    )

    assert metadata.stop_in_samples == 500
    assert metadata.stop == pytest.approx(0.5)

    metadata.stop = 0.8

    assert metadata.duration_in_samples == pytest.approx(600)
    assert metadata.stop_in_samples == pytest.approx(800)
    assert metadata.stop == pytest.approx(0.8)


def test_signal_metadata_duration_getter_and_setter():
    metadata = SignalMetadataObject(
        num_iq_samples_dataset=1000,
        duration_in_samples=250,
    )

    assert metadata.duration == pytest.approx(0.25)

    metadata.duration = 0.75

    assert metadata.duration_in_samples == pytest.approx(750)
    assert metadata.duration == pytest.approx(0.75)


def test_signal_metadata_stop_in_samples_setter():
    metadata = SignalMetadataObject(
        start_in_samples=100,
        duration_in_samples=50,
    )

    assert metadata.stop_in_samples == 150

    metadata.stop_in_samples = 400

    assert metadata.duration_in_samples == 300
    assert metadata.stop_in_samples == 400


def test_signal_metadata_frequency_properties_from_center_and_bandwidth():
    metadata = SignalMetadataObject(
        center_freq=1000.0,
        bandwidth=200.0,
    )

    assert metadata.lower_freq == pytest.approx(900.0)
    assert metadata.upper_freq == pytest.approx(1100.0)


def test_signal_metadata_center_freq_update_recalculates_cached_edges():
    metadata = SignalMetadataObject(
        center_freq=1000.0,
        bandwidth=200.0,
    )
    assert metadata.lower_freq == pytest.approx(900.0)
    assert metadata.upper_freq == pytest.approx(1100.0)

    metadata["center_freq"] = -500.0

    assert metadata.lower_freq == pytest.approx(-600.0)
    assert metadata.upper_freq == pytest.approx(-400.0)
    assert "_lower_frequency" not in metadata.metadata
    assert "_upper_frequency" not in metadata.metadata


def test_signal_metadata_bandwidth_update_recalculates_cached_edges():
    metadata = SignalMetadataObject(
        center_freq=1000.0,
        bandwidth=200.0,
    )
    assert metadata.lower_freq == pytest.approx(900.0)
    assert metadata.upper_freq == pytest.approx(1100.0)

    metadata["bandwidth"] = 600.0

    assert metadata.lower_freq == pytest.approx(700.0)
    assert metadata.upper_freq == pytest.approx(1300.0)
    assert "_lower_frequency" not in metadata.metadata
    assert "_upper_frequency" not in metadata.metadata


def test_signal_metadata_canonical_fields_override_legacy_cached_edges():
    metadata = SignalMetadataObject(
        center_freq=1000.0,
        bandwidth=200.0,
        _lower_frequency=-999.0,
        _upper_frequency=999.0,
    )

    assert metadata.lower_freq == pytest.approx(900.0)
    assert metadata.upper_freq == pytest.approx(1100.0)


def test_signal_metadata_edge_setter_updates_existing_canonical_interval():
    metadata = SignalMetadataObject(
        center_freq=1000.0,
        bandwidth=200.0,
    )

    metadata.lower_freq = 800.0

    assert metadata.lower_freq == pytest.approx(800.0)
    assert metadata.upper_freq == pytest.approx(1100.0)
    assert metadata.center_freq == pytest.approx(950.0)
    assert metadata.bandwidth == pytest.approx(300.0)


def test_signal_metadata_upper_freq_setter_updates_bandwidth_and_center_freq():
    metadata = SignalMetadataObject()

    metadata.lower_freq = 900.0
    metadata.upper_freq = 1300.0

    assert metadata.lower_freq == pytest.approx(900.0)
    assert metadata.upper_freq == pytest.approx(1300.0)
    assert metadata.bandwidth == pytest.approx(400.0)
    assert metadata.center_freq == pytest.approx(1100.0)
    assert "_lower_frequency" not in metadata.metadata
    assert "_upper_frequency" not in metadata.metadata


def test_signal_metadata_lower_freq_setter_updates_bandwidth_and_center_freq():
    metadata = SignalMetadataObject()
    metadata.upper_freq = 1300.0
    metadata.lower_freq = 700.0

    assert metadata.lower_freq == pytest.approx(700.0)
    assert metadata.upper_freq == pytest.approx(1300.0)
    assert metadata.bandwidth == pytest.approx(600.0)
    assert metadata.center_freq == pytest.approx(1000.0)
    assert "_lower_frequency" not in metadata.metadata
    assert "_upper_frequency" not in metadata.metadata


def test_signal_metadata_frequency_properties_raise_when_missing_inputs():
    metadata = SignalMetadataObject()

    with pytest.raises(ValueError, match="missing center_freq or bandwidth"):
        _ = metadata.lower_freq

    with pytest.raises(ValueError, match="missing center_freq or bandwidth"):
        _ = metadata.upper_freq


def test_signal_metadata_oversampling_rate():
    metadata = SignalMetadataObject(
        sample_rate=10_000.0,
        bandwidth=2_000.0,
    )

    assert metadata.oversampling_rate == pytest.approx(5.0)


def test_signal_metadata_to_dict_excludes_internal_fields():
    metadata = SignalMetadataObject(
        class_name="qpsk",
        class_index=34,
        applied_transforms=["transform"],
        dataset_metadata={"a": 1},
        _dataset_metadata={"b": 2},
        _center_freq_set=True,
    )

    result = metadata.to_dict()

    assert result["class_name"] == "qpsk"
    assert result["class_index"] == 34
    assert "applied_transforms" not in result
    assert "dataset_metadata" not in result
    assert "_dataset_metadata" not in result
    assert "_center_freq_set" not in result


def test_signal_initializes_empty_data_when_data_is_none():
    signal = Signal()

    assert isinstance(signal.data, np.ndarray)
    assert signal.data.size == 0
    assert signal.duration_in_samples == 0


def test_signal_converts_input_data_to_numpy_array():
    signal = Signal(data=[1.0, 2.0, 3.0])

    assert isinstance(signal.data, np.ndarray)
    assert np.array_equal(signal.data, np.array([1.0, 2.0, 3.0]))
    assert signal.duration_in_samples == 3


def test_signal_stores_metadata_fields():
    signal = Signal(
        data=np.array([1.0, 2.0]),
        class_name="qpsk",
        class_index=34,
        snr_db=12.5,
    )

    assert signal.class_name == "qpsk"
    assert signal.class_index == 34
    assert signal.snr_db == pytest.approx(12.5)


def test_signal_creates_new_component_signal_list():
    """Signals created without component signals should not share a list."""
    signal1 = Signal(data=np.array([1.0]))
    signal2 = Signal(data=np.array([2.0]))

    assert signal1.component_signals == []
    assert signal2.component_signals == []
    assert signal1.component_signals is not signal2.component_signals

    signal1.component_signals.append(Signal(data=np.array([3.0])))

    assert len(signal1.component_signals) == 1
    assert len(signal2.component_signals) == 0


def test_signal_copies_component_signal_list_from_constructor():
    """Signal should copy the component signal list provided by the caller."""
    component = Signal(data=np.array([1.0]))
    component_list = [component]

    signal = Signal(
        data=np.array([2.0]),
        component_signals=component_list,
    )

    assert signal.component_signals == component_list
    assert signal.component_signals is not component_list

    component_list.append(Signal(data=np.array([3.0])))

    assert len(component_list) == 2
    assert len(signal.component_signals) == 1


def test_signal_repr_contains_metadata_and_component_signals():
    component = Signal(data=np.array([1.0]), class_name="qpsk")
    signal = Signal(
        data=np.array([2.0]),
        component_signals=[component],
        class_name="composite",
    )

    text = repr(signal)

    assert "Signal" in text
    assert "metadata=" in text
    assert "component_signals=" in text
    assert "composite" in text


def test_signal_copy_preserves_metadata_as_top_level_fields():
    """Signal.copy should preserve metadata fields directly on the copied signal."""
    signal = Signal(
        data=np.array([1.0, 2.0], dtype=np.float32),
        class_name="qpsk",
        class_index=34,
        snr_db=12.5,
    )

    copied = signal.copy()

    assert copied is not signal
    assert copied.class_name == "qpsk"
    assert copied.class_index == 34
    assert copied.snr_db == pytest.approx(12.5)
    assert "metadata" not in copied.keys()


def test_signal_copy_deep_copies_data():
    signal = Signal(
        data=np.array([1.0, 2.0], dtype=np.float32),
        class_index=1,
    )

    copied = signal.copy()

    assert np.array_equal(copied.data, signal.data)
    assert copied.data is not signal.data

    copied.data[0] = 99.0

    assert signal.data[0] == 1.0
    assert copied.data[0] == 99.0


def test_signal_copy_deep_copies_component_signals():
    component = Signal(
        data=np.array([3.0, 4.0], dtype=np.float32),
        class_name="ofdm-64",
        class_index=1,
    )
    signal = Signal(
        data=np.array([1.0, 2.0], dtype=np.float32),
        component_signals=[component],
        bandwidth=1000.0,
    )

    copied = signal.copy()

    assert copied is not signal
    assert copied.component_signals is not signal.component_signals
    assert len(copied.component_signals) == 1

    copied_component = copied.component_signals[0]

    assert copied_component is not component
    assert copied_component.class_name == "ofdm-64"
    assert copied_component.class_index == 1
    assert np.array_equal(copied_component.data, component.data)

    copied_component.data[0] = 99.0
    copied_component.class_index = 42

    assert component.data[0] == 3.0
    assert component.class_index == 1


def test_signal_copy_preserves_nested_component_structure():
    grandchild = Signal(
        data=np.array([5.0]),
        class_name="tone",
        class_index=0,
    )
    child = Signal(
        data=np.array([3.0]),
        component_signals=[grandchild],
        class_name="qpsk",
        class_index=34,
    )
    parent = Signal(
        data=np.array([1.0]),
        component_signals=[child],
        class_name="parent",
    )

    copied = parent.copy()

    assert copied is not parent
    assert copied.component_signals[0] is not child
    assert copied.component_signals[0].component_signals[0] is not grandchild

    assert copied.component_signals[0].class_name == "qpsk"
    assert copied.component_signals[0].component_signals[0].class_name == "tone"

    copied.component_signals[0].component_signals[0].class_index = 99

    assert grandchild.class_index == 0


def test_signal_preserves_explicit_duration_in_samples():
    signal = Signal(
        data=np.zeros(100, dtype=np.complex64),
        duration_in_samples=25,
        num_iq_samples_dataset=100,
    )

    assert signal.duration_in_samples == 25


def test_signal_defaults_duration_in_samples_to_data_length():
    signal = Signal(
        data=np.zeros(100, dtype=np.complex64),
        num_iq_samples_dataset=100,
    )

    assert signal.duration_in_samples == 100
