from torchsig.transforms import (
    Compose,
    Identity,
    Normalize,
    RandomPhaseShift,
)
from torchsig.transforms.torch_transforms.transforms import (
    Compose as TorchCompose,
    Identity as TorchIdentity,
    Normalize1D as TorchNormalize1D,
    Normalize2D as TorchNormalize2D,
    PhaseShift as TorchPhaseShift,
)
from torchsig.transforms.torch_transforms.types import (
    Signal, SignalMetadata
)

import torch
import numpy as np
import pytest
from copy import deepcopy
from typing import Literal


###############################################################################
# Random data generation helpers
###############################################################################
def gen_rand_1d_data(
    num_samples: int, 
    complex: bool = True, 
    device: Literal['cpu', 'gpu'] = 'cpu',
) -> torch.Tensor:
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    if complex:
        data: torch.Tensor = (
            torch.rand(num_samples, generator=rng) - 0.5
        ) + 1j * (
            torch.rand(num_samples, generator=rng) - 0.5
        )
    else:
        data: torch.Tensor = (
            torch.rand(num_samples, generator=rng) - 0.5
        )
    return data


def gen_rand_2d_data(
    x: int, 
    y: int, 
    complex: bool = True, 
    device: Literal['cpu', 'gpu'] = 'cpu',
) -> torch.Tensor:
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    if complex:
        data: torch.Tensor = (
            torch.rand(x, y, generator=rng) - 0.5
        ) + 1j * (
            torch.rand(x, y, generator=rng) - 0.5
        )
    else:
        data: torch.Tensor = (
            torch.rand(x, y, generator=rng) - 0.5
        ) 
    return data


###############################################################################
# TorchScript JIT-able tests
###############################################################################
# 
# We want all Transforms that are not Augmentations to be JIT-able such that we
# can optionally embed them within network layers for more robust speed and 
# portability. Augmentations are generally not going to be embedded this way, 
# and since `torch.Generator` is currently unsupported by TorchScript, we only 
# conduct these tests for the deterministic Transforms.
# 
###############################################################################
@pytest.mark.parametrize(
    "transform", 
    [
        TorchNormalize1D,
        TorchIdentity,
    ],
)
def test_jit_able_1d(transform):
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = transform()
    t_jit = torch.jit.script(t)
    with torch.no_grad():
        t_jit(data)
    return


@pytest.mark.parametrize(
    "transform", 
    [
        TorchNormalize2D,
    ],
)
def test_jit_able_2d(transform):
    data: Signal = Signal(
        data=gen_rand_2d_data(16, 16),
        metadata=[SignalMetadata(1.0, 16*16)],
    )
    t = transform()
    t_jit = torch.jit.script(t)
    with torch.no_grad():
        t_jit(data)
    return


###############################################################################
# Identity tests and parity tests
###############################################################################
def test_identity():
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchIdentity()
    new_data = t(data)
    assert new_data == data
    

def test_identity_parity():
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchIdentity()
    new_data = t(data)
    
    data_npy = data.data.numpy()
    t = Identity()
    new_data_npy = t(data_npy)
    
    assert np.allclose(new_data.data.numpy(), new_data_npy)
    

###############################################################################
# Compose tests and parity tests
###############################################################################
def test_compose():
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchCompose([TorchPhaseShift(-1, -1), TorchPhaseShift(1, 1)])
    new_data = t(data)
    assert new_data == data
    
    t = TorchCompose([TorchPhaseShift(-1, -1), TorchIdentity()])
    t_phase_shift = TorchPhaseShift(-1, -1)
    new_data = t(data)
    new_phase_shift_data = t_phase_shift(data)
    assert torch.allclose(new_data.data, new_phase_shift_data.data)
    

def test_identity_parity():
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchCompose([TorchPhaseShift(-1, -1), TorchPhaseShift(1, 1)])
    new_data = t(data)
    
    data_npy = data.data.numpy()
    t = Compose([RandomPhaseShift((-1, -1)), RandomPhaseShift((-1, -1))])
    new_data_npy = t(data_npy)
    
    assert np.allclose(new_data.data.numpy(), new_data_npy)


###############################################################################
# Normalize1D tests and parity tests
###############################################################################
@pytest.mark.parametrize("norm", [2, float('inf')])
def test_normalize1d(norm):
    data: Signal = Signal(
        data=3 * gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchNormalize1D(
        norm=norm,
    )
    new_data = t(deepcopy(data))
    
    norm_vals = torch.linalg.norm(data.data, norm, dim=-1)
    assert torch.allclose(
        new_data.data,
        data.data / norm_vals,
    )
    
    assert not torch.allclose(new_data.data, data.data)
    assert new_data.data.size() == data.data.size()
    assert (
        new_data.metadata[0].sample_rate == data.metadata[0].sample_rate
    ) and (
        new_data.metadata[0].num_samples == data.metadata[0].num_samples
    )

    
@pytest.mark.parametrize("norm", [2, float('inf')])
def test_normalize1d_parity(norm):
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchNormalize1D(
        norm=norm,
    )
    new_data = t(data)
    
    data_npy = data.data.numpy()
    t = Normalize(
        norm=norm,
    )
    new_data_npy = t(data_npy)
    
    assert np.allclose(new_data.data.numpy(), new_data_npy)
    
    
###############################################################################
# Normalize2D tests and parity tests
###############################################################################
@pytest.mark.parametrize("norm", [2, float('inf')])
@pytest.mark.parametrize("flatten", [True, False])
def test_normalize2d(norm, flatten):
    data: Signal = Signal(
        data=3 * gen_rand_2d_data(16, 16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchNormalize2D(
        norm=norm,
        flatten=flatten,
    )
    new_data = t(deepcopy(data))
    
    if flatten:
        flat_tensor = data.data.flatten(start_dim=0)
        norm_vals = torch.linalg.norm(flat_tensor, ord=norm, dim=-1)
        assert torch.allclose(
            new_data.data,
            data.data / norm_vals,
        )
    else:
        norm_vals = torch.linalg.norm(data.data, norm, dim=(-2, -1))
        assert torch.allclose(
            new_data.data,
            data.data / norm_vals,
        )
    
    assert not torch.allclose(new_data.data, data.data)
    assert new_data.data.size() == data.data.size()
    assert (
        new_data.metadata[0].sample_rate == data.metadata[0].sample_rate
    ) and (
        new_data.metadata[0].num_samples == data.metadata[0].num_samples
    )
    
    
@pytest.mark.parametrize("norm", [2, float('inf')])
@pytest.mark.parametrize("flatten", [True, False])
def test_normalize2d_parity(norm, flatten):
    data: Signal = Signal(
        data=gen_rand_2d_data(16, 16),
        metadata=[SignalMetadata(1.0, 16*16)],
    )
    t = TorchNormalize2D(
        norm=norm,
        flatten=flatten,
    )
    new_data = t(data)
    
    data_npy = data.data.numpy()
    t = Normalize(
        norm=norm,
        flatten=flatten,
    )
    new_data_npy = t(data_npy)
    
    assert np.allclose(new_data.data.numpy(), new_data_npy)

    
###############################################################################
# PhaseShift tests and parity tests
###############################################################################
@pytest.mark.parametrize("phase_offset", [
    -1, -0.5, 0, 0.5, 1, (-1, 1), [-0.5, 0.5]
])
def test_phase_shift(phase_offset):
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchPhaseShift(
        phase_offset=phase_offset,
    )
    new_data = t(deepcopy(data))
        
    if isinstance(phase_offset, int) or isinstance(phase_offset, float):
        assert torch.allclose(
            new_data.data, 
            data.data * torch.exp(
                torch.tensor([1j * phase_offset * torch.pi])
            )
        )
        
    assert (
        new_data.metadata[0].sample_rate == data.metadata[0].sample_rate
    ) and (
        new_data.metadata[0].num_samples == data.metadata[0].num_samples
    )

    
@pytest.mark.parametrize("phase_offset", [-1, -0.5, 0, 0.5, 1])
def test_phase_shift_parity(phase_offset):
    data: Signal = Signal(
        data=gen_rand_1d_data(16),
        metadata=[SignalMetadata(1.0, 16)],
    )
    t = TorchPhaseShift(
        phase_offset=phase_offset,
    )
    new_data = t(deepcopy(data))
    
    data_npy = data.data.numpy()
    t = RandomPhaseShift(
        phase_offset=phase_offset,
    )
    new_data_npy = t(data_npy)
    
    assert np.allclose(new_data.data.numpy(), new_data_npy)
    