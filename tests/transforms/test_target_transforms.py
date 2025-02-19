"""Unit Tests: transforms/taregt_transforms.py
"""

from torchsig.transforms.target_transforms import (
    TargetTransform,
    CustomLabel,
    PassThrough,
    CenterFreq,
    Bandwidth,
    StartInSamples,
    DurationInSamples,
    SNR,
    ClassName,
    SampleRate,
    NumSamples,
    Start,
    Stop,
    Duration,
    StopInSamples,
    UpperFreq,
    LowerFreq,
    OversamplingRate,
    SamplesPerBaud,
    FamilyName,
    FamilyIndex,
    YOLOLabel
)
from test_transforms_utils import (
    generate_test_dataset_dict
)
from torchsig.signals.signal_types import DatasetDict
from torchsig.signals.signal_lists import TorchSigSignalLists

import pytest
from copy import deepcopy
from typing import Any


TEST_SIGNAL = generate_test_dataset_dict(num_iq_samples = 64, scale = 1.0)


def is_valid(target_transform: TargetTransform, call_output: Any = None) -> bool:
    if not isinstance(target_transform, TargetTransform):
        return False
    if not isinstance(target_transform.required_metadata, list):
        return False
    if not isinstance(target_transform.targets_metadata, list):
        return False
    if not call_output is None:
        if not isinstance(call_output, list):
            return False
        for new_field in target_transform.targets_metadata:
            for m in call_output:
                if new_field not in m.keys():
                    return False

    return True


@pytest.mark.parametrize("is_error", [False])
def test_TargetTransform(is_error: bool) -> None:

    TT = TargetTransform()

    assert is_valid(TT)
    assert len(TT.required_metadata) == 0
    assert len(TT.targets_metadata) == 0

    with pytest.raises(ValueError, match=r".*"):
        signal = TT("error")

    with pytest.raises(NotImplementedError):
        signal = generate_test_dataset_dict(
            num_iq_samples = 64,
            scale = 1.0
        )
        signal.metadata = TT(signal.metadata)


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'label_fields': "error",
            'label_name': "error"
        },
        True
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'label_fields': ["class_index"],
            'label_name': "label"
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'label_fields': ["class_index", "snr_db"],
            'label_name': "label"
        },
        False
    ),
])
def test_CustomLabel(
    signal: DatasetDict,
    params: dict,
    is_error: bool
):

    label_fields = params['label_fields']
    label_name = params['label_name']
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            TT = CustomLabel(label_fields=label_fields, label_name=label_name)
            signal.metadata = TT(signal.metadata)

            assert is_valid(TT, signal.metadata)
    else:
        TT = CustomLabel(label_fields=label_fields, label_name=label_name)
        signal.metadata = TT(signal.metadata)

        assert isinstance(TT, CustomLabel)
        assert TT.required_metadata == label_fields
        assert TT.targets_metadata == [label_name]
        assert is_valid(TT, signal.metadata)

@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'field': "error"
        },
        True
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'field': "class_index"
        },
        True
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'field': ["class_index"]
        },
        False
    )
])
def test_PassThrough(
    signal: DatasetDict,
    params: dict,
    is_error: bool
):
    field = params['field']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            TT = PassThrough(field = field)
            signal.metadata = TT(signal.metadata)

            assert is_valid(TT, signal.metadata)
    else:
        TT = PassThrough(field = field)
        signal.metadata = TT(signal.metadata)

        assert isinstance(TT, PassThrough)
        assert TT.required_metadata == field
        assert TT.targets_metadata == field
        assert is_valid(TT, signal.metadata)

@pytest.mark.parametrize("signal, target_transform, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        CenterFreq,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        Bandwidth,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        StartInSamples,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        DurationInSamples,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        SNR,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        ClassName,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        SampleRate,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        NumSamples,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        Start,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        Stop,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        Duration,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        StopInSamples,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        UpperFreq,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        LowerFreq,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        OversamplingRate,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        SamplesPerBaud,
        False
    ),
])
def test_BuiltInTargetTransforms(
    signal: DatasetDict,
    target_transform: PassThrough,
    is_error: bool
):
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            TT = target_transform()
            signal.metadata = TT(signal.metadata)

            assert is_valid(TT, signal.metadata)
    else:
        TT = target_transform()
        signal.metadata = TT(signal.metadata)

        assert isinstance(TT, target_transform)
        assert is_valid(TT, signal.metadata)

@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'class_family_dict': TorchSigSignalLists.family_dict
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'class_family_dict': "error"
        },
        True
    ),
])
def test_FamilyName(
    signal: DatasetDict,
    params: dict,
    is_error: bool
):
    class_family_dict = params['class_family_dict']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            TT = FamilyName(class_family_dict=class_family_dict)
            signal.metadata = TT(signal.metadata)

            assert is_valid(TT, signal.metadata)
    else:
        TT = FamilyName(class_family_dict=class_family_dict)
        signal.metadata = TT(signal.metadata)

        assert isinstance(TT, FamilyName)
        assert TT.class_family_dict == class_family_dict
        assert is_valid(TT, signal.metadata)

@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'class_family_dict': TorchSigSignalLists.family_dict,
            'family_list': TorchSigSignalLists.family_list
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'class_family_dict': "error",
            'family_list': "error"
        },
        True
    ),
])
def test_FamilyIndex(
    signal: DatasetDict,
    params: dict,
    is_error: bool
):
    class_family_dict = params['class_family_dict']
    family_list = params['family_list']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            TT = FamilyIndex(
                class_family_dict=class_family_dict,
                family_list=family_list
            )
            signal.metadata = TT(signal.metadata)

            assert is_valid(TT, signal.metadata)
    else:
        TT = FamilyIndex(
            class_family_dict=class_family_dict,
            family_list=family_list
        )        
        signal.metadata = TT(signal.metadata)

        assert isinstance(TT, FamilyIndex)
        assert TT.class_family_dict == class_family_dict
        assert TT.family_list == family_list
        assert is_valid(TT, signal.metadata)


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
        },
        False
    ),
])
def test_YOLOLabel(
    signal: DatasetDict,
    params: dict,
    is_error: bool
):

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            TT = YOLOLabel()
            signal.metadata = TT(signal.metadata)

            assert is_valid(TT, signal.metadata)
    else:
        TT = YOLOLabel()       
        signal.metadata = TT(signal.metadata)

        assert isinstance(TT, YOLOLabel)
        assert is_valid(TT, signal.metadata)
        for m in signal.metadata:
            label = m['yolo_label']
            assert isinstance(label, tuple)
            assert len(label) == 5
