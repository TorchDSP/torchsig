"""Unit Tests: transforms/base_transforms
"""

from torchsig.transforms.base_transforms import (
    Transform,
    Compose,
    Lambda,
    Normalize,
    RandomApply,
    RandAugment
)
from torchsig.signals.signal_types import Signal, DatasetSignal
from torchsig.utils.dsp import torchsig_complex_data_type
from test_transforms_utils import (
    generate_test_signal,
    generate_test_dataset_signal
)

# Third Party
from typing import Callable
import numpy as np
from copy import deepcopy
import pytest


AnySignal = Signal | DatasetSignal
RTOL = 1E-6
TEST_SIGNAL = generate_test_signal(num_iq_samples = 64, scale = 1.0)
TEST_DS_SIGNAL = generate_test_dataset_signal(num_iq_samples = 64, scale = 1.0)


@pytest.mark.parametrize("is_error", [False])
def test_Transform(is_error: bool) -> None:
    """Test the base Transform with pytest.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """    
    T = Transform()
    
    assert isinstance(T, Transform) != is_error
    assert isinstance(T.random_generator, np.random.Generator) != is_error


@pytest.mark.parametrize("signal, params, expected, is_error", [
    (deepcopy(TEST_SIGNAL), {'transforms' :  ['invalid_transform']}, AttributeError, True),
    (deepcopy(TEST_SIGNAL), {'transforms' :  [lambda x: x]}, True, False),
    (deepcopy(TEST_DS_SIGNAL), {'transforms' :  []}, True, False)
])
def test_Compose(
    signal: AnySignal,
    params: dict,
    expected: bool | AttributeError, 
    is_error: bool
    ) -> None:
    """Test the Compose Transform with pytest.

    Args:
        signal (AnySignal): input signal.
        params (dict): Transform call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """    
    t_list = params['transforms']

    if is_error:
        with pytest.raises(TypeError):     
            T = Compose(transforms = t_list)
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = Compose(transforms = t_list)
        signal = T(signal)
        
        assert isinstance(T, Compose)
        assert isinstance(T.transforms, list)
        for transform in T.transforms:
            assert callable(transform)
        assert isinstance(signal, AnySignal)
        assert type(signal) == type(signal_test)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        assert np.allclose(signal.data, signal_test.data, RTOL) == expected


@pytest.mark.parametrize("signal, func, expected, is_error", [
    (
        deepcopy(TEST_SIGNAL),
        'invalid_function',
        TypeError,
        True
    ),
    (
        deepcopy(TEST_SIGNAL),
        lambda x: x*42.0,
        generate_test_signal(num_iq_samples=64, scale=42.0),
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL),
        lambda x: x*0.42,
        generate_test_dataset_signal(num_iq_samples=64, scale=0.42),
        False
    ),

])
def test_Lambda(
    signal: AnySignal, 
    func: Callable, 
    expected: AnySignal | TypeError,
    is_error: bool
    ) -> None:
    """Test the Lambda Transform with pytest.

    Args:
        signal (AnySignal): input signal.
        func (Callable): lambda function to execute on signal.
        expected (AnySignal | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    if is_error:
        with pytest.raises(expected):     
            T = Lambda(func)
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = Lambda(func)
        signal = T(signal)

        assert isinstance(T, Lambda)
        assert isinstance(T.func, Callable)
        assert isinstance(signal, AnySignal)
        assert type(signal) == type(signal_test)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        assert np.allclose(signal.data, expected.data, RTOL)


@pytest.mark.parametrize("signal, params, expected, is_error", [
    (
        generate_test_signal(num_iq_samples=64, scale=42.0),
        {'norm': 'invalid_norm', 'flatten': False},
        ValueError,
        True
    ),
    (
        generate_test_signal(num_iq_samples=64, scale=42.0),
        {'norm': 2, 'flatten': False},
        TEST_SIGNAL,
        False
    ),
    (
        generate_test_dataset_signal(num_iq_samples=64, scale=0.42),
        {'norm': 2, 'flatten': False},
        TEST_DS_SIGNAL,
        False
    ),
])
def test_Normalize(
    signal: AnySignal, 
    params: dict, 
    expected: AnySignal | ValueError,
    is_error: bool
    ) -> None:
    """Test the Normalize Transform with pytest.

    Args:
        signal (AnySignal): input signal.
        params (dict): Transform call parameters (see description).
        expected (AnySignal | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    if is_error:
        with pytest.raises(expected):     
            T = Normalize(
                norm = params['norm'], 
                flatten = params['flatten']
            )
            signal = T(signal)
    else:    
        signal_test = deepcopy(signal)

        T = Normalize(
            norm = params['norm'], 
            flatten = params['flatten']
        )
        signal = T(signal)

        assert isinstance(T, Normalize)
        assert isinstance(T.norm, int)
        assert isinstance(T.flatten, bool)
        assert isinstance(signal, AnySignal)
        assert type(signal) == type(signal_test)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        assert np.allclose(signal.data, expected.data, RTOL) 


@pytest.mark.parametrize("signal, params, expected, is_error", [
    (
        generate_test_signal(num_iq_samples=64, scale=42.0),
        {'transform': 'invalid_transform', 'probability': 1.0},
        TypeError,
        True
    ),
    (
        generate_test_signal(num_iq_samples=64, scale=42.0),
        {'transform': Normalize(), 'probability': 0.0},
        generate_test_signal(num_iq_samples=64, scale=42.0),
        False
    ),
    (
        generate_test_dataset_signal(num_iq_samples=64, scale=0.42),
        {'transform': Normalize(), 'probability': 1.0},
        TEST_DS_SIGNAL,
        False
    ),
])
def test_RandomApply(
    signal: AnySignal, 
    params: dict, 
    expected: AnySignal | TypeError,
    is_error: bool
    ) -> None:
    """Test the RandomApply Transform with pytest.

    Args:
        signal (AnySignal): input signal.
        params (dict): Transform call parameters (see description).
        expected (AnySignal | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    if is_error:
        with pytest.raises(expected):  
            T = RandomApply(
                transform = params['transform'], 
                probability = params['probability']
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = RandomApply(
            transform = params['transform'], 
            probability = params['probability']
        )
        signal = T(signal)    

        assert isinstance(T, RandomApply)
        assert isinstance(T.transform, Transform)
        assert isinstance(T.probability, float)
        assert isinstance(signal, AnySignal)
        assert type(signal) == type(signal_test)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type


@pytest.mark.parametrize("signal, params, expected, is_error", [
    (
        generate_test_signal(num_iq_samples=64, scale=42.0),
        {
            'transforms': ['invalid_transform'],
            'choose': 2,
            'replace': False,
        }, 
        ValueError,
        True
    ),
    (
        generate_test_signal(num_iq_samples=64, scale=42.0),
        {
            'transforms': [Normalize(), Normalize(), Normalize()],
            'choose': 2,
            'replace': False,
        }, 
        TEST_SIGNAL,
        False
    ),
    (
        generate_test_dataset_signal(num_iq_samples=64, scale=0.42),
        {
            'transforms': [Normalize()],
            'choose': 1,
            'replace': True,
        }, 
        TEST_DS_SIGNAL,
        False
    ),    
])
def test_RandAugment(
    signal: AnySignal, 
    params: dict, 
    expected: AnySignal | ValueError,
    is_error: bool
    ) -> None:
    """Test the RandomAugment Transform with pytest.

    Args:
        signal (AnySignal): input signal.
        params (dict): Transform call parameters (see description).
        expected (AnySignal | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    if is_error:
        with pytest.raises(expected):  
            T = RandAugment(
                    transforms = params['transforms'],
                    choose = params['choose'],
                    replace = params['replace'] 
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = RandAugment(
                transforms = params['transforms'],
                choose = params['choose'],
                replace = params['replace'] 
        )
        signal = T(signal)

        assert isinstance(T, RandAugment)
        assert isinstance(T.transforms, list)
        for transform in T.transforms:
            assert callable(transform)
        assert isinstance(T.choose, int)
        assert isinstance(T.replace, bool)
        assert isinstance(signal, AnySignal)
        assert type(signal) == type(signal_test)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        assert np.allclose(signal.data, expected.data, RTOL)

