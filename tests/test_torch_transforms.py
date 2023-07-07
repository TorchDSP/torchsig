from torchsig.transforms import Normalize
from torchsig.transforms.torch_transforms.transforms import (
    Normalize as TorchNormalize,
)

import torch
import numpy as np
import pytest


def gen_rand_data(num_iq_samples: int, device='cpu') -> torch.Tensor:
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    data: torch.Tensor = (
        torch.rand(num_iq_samples, generator=rng) - 0.5
    ) + 1j * (
        torch.rand(num_iq_samples, generator=rng) - 0.5
    )
    return data


@pytest.mark.parametrize("norm", [2, float('inf')])
@pytest.mark.parametrize("flatten", [True, False])
def test_normalize_parity(norm, flatten):
    data = gen_rand_data(16)
    t = TorchNormalize(
        norm=norm,
        flatten=flatten,
    )
    new_data = t(data)
    
    data_npy = data.numpy()
    t = Normalize(
        norm=norm,
        flatten=flatten,
    )
    new_data_npy = t(data_npy)
    
    assert np.allclose(new_data.numpy(), new_data_npy)

@pytest.mark.parametrize("transform", [TorchNormalize])
def test_jit_able(transform):
    data = gen_rand_data(16)
    t = transform()
    t_jit = torch.jit.script(t)
    with torch.no_grad():
        t_jit(data)
    return