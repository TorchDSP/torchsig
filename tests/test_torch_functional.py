from torchsig.transforms.torch_transforms import functional as F

import torch
import numpy as np
import pytest
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


def gen_rand_batched_1d_data(
    batch_size: int, 
    num_samples: int, 
    complex: bool = True,
    device: Literal['cpu', 'gpu'] = 'cpu',
) -> torch.Tensor:
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    if complex:
        data: torch.Tensor = (
            torch.rand(batch_size, num_samples, generator=rng) - 0.5
        ) + 1j * (
            torch.rand(batch_size, num_samples, generator=rng) - 0.5
        )
    else:
        data: torch.Tensor = (
            torch.rand(batch_size, num_samples, generator=rng) - 0.5
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


def gen_rand_batched_2d_data(
    batch_size: int, 
    x: int, 
    y: int,
    complex: bool = True,
    device: Literal['cpu', 'gpu'] = 'cpu',
) -> torch.Tensor:
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    if complex:
        data: torch.Tensor = (
            torch.rand(batch_size, x, y, generator=rng) - 0.5
        ) + 1j * (
            torch.rand(batch_size, x, y, generator=rng) - 0.5
        )
    else:
        data: torch.Tensor = (
            torch.rand(batch_size, x, y, generator=rng) - 0.5
        ) 
    return data


###############################################################################
# Functional tests
###############################################################################
@pytest.mark.parametrize("choices", [[1, 2, 3], [1.2, 2.3, 3.4], ['a', 'b']])
def test_uniform_discrete_distribution(choices):
    rng = torch.Generator()
    rng.manual_seed(0)
    selection_func = F.uniform_discrete_distribution(choices, rng)
    seen = [False] * len(choices)
    num_tests = 100
    for _ in range(num_tests):
        selection = selection_func()
        assert selection in choices
        seen[choices.index(selection)] = True
    assert all(seen)

    
@pytest.mark.parametrize("dist", [(-10, 0), (0, 0), (-10.3, 10.1)])
def test_uniform_continuous_distribution(dist):
    rng = torch.Generator()
    rng.manual_seed(0)
    selection_func = F.uniform_continuous_distribution(dist[0], dist[1], rng)
    prev_val = None
    num_tests = 10
    for _ in range(num_tests):
        selection = selection_func()
        if dist[0] == dist[1]:
            assert selection == dist[0]
        else:
            # Probability of hitting boundary is low enough that we can omit
            # = here to ensure we are returning a random value in the range
            assert dist[0] < selection < dist[1]
            # Also test that we are getting unique values each iteration
            assert selection != prev_val
            prev_val = selection


@pytest.mark.parametrize("dist", [
    [1, 2, 3], 
    [1.2, 2.3, 3.4], 
    ['a', 'b'], 
    (-10, 0), 
    (0, 0), 
    (-10.3, 10.1),
    3,
    4.2,
    'a',
    lambda : 3.1,
])
def test_to_distribution(dist):
    rng = torch.Generator()
    rng.manual_seed(0)
    selection_func = F.to_distribution(dist, rng)
    selection = selection_func()
    if isinstance(dist, list):
        assert selection in dist
    elif isinstance(dist, tuple):
        assert dist[0] <= selection <= dist[1]
    elif isinstance(dist, int) or \
         isinstance(dist, float) or \
         isinstance(dist, str):
        assert selection == dist
    else:
        assert selection == dist()


@pytest.mark.parametrize("norm", [2, float('inf')])
@pytest.mark.parametrize("batched", [False, True])
def test_normalize1d(norm, batched):
    if batched:
        data = 3 * gen_rand_batched_1d_data(4, 16)
        new_data = F.normalize1d(data, norm)
    else:
        data = 3 * gen_rand_1d_data(16)

        # Test batch size of 1 match single
        unsqueezed_data = data.unsqueeze(0)
        unsqueezed_new_data = F.normalize1d(
            unsqueezed_data, 
            norm,
        )
        new_data = F.normalize1d(data, norm)
        assert torch.allclose(unsqueezed_new_data.squeeze(), new_data)
    
    assert not torch.allclose(new_data, data)        
    assert data.shape == new_data.shape
    
    norm_vals = torch.linalg.norm(data, norm, dim=-1)
    norm_vals = norm_vals.unsqueeze(1) if batched else norm_vals
    assert torch.allclose(
        new_data,
        data / norm_vals,
    )

    
@pytest.mark.parametrize("norm", [2, float('inf')])
@pytest.mark.parametrize("flatten", [True, False])
@pytest.mark.parametrize("batched", [False, True])
def test_normalize2d(norm, flatten, batched):
    if batched:
        data = 3 * gen_rand_batched_2d_data(4, 16, 16)
        new_data = F.normalize2d(data, norm, flatten)
    else:
        data = 3 * gen_rand_2d_data(16, 16)

        # Test batch size of 1 match single
        unsqueezed_data = data.unsqueeze(0)
        unsqueezed_new_data = F.normalize2d(
            unsqueezed_data, 
            norm,
            flatten,
        )
        new_data = F.normalize2d(data, norm, flatten)
        assert torch.allclose(unsqueezed_new_data.squeeze(), new_data)
    
    assert not torch.allclose(new_data, data)        
    assert data.shape == new_data.shape
    
    if flatten:
        start_dim = 1 if batched else 0
        flat_tensor = data.flatten(start_dim=start_dim)
        norm_vals = torch.linalg.norm(flat_tensor, ord=norm, dim=-1)
        norm_vals = norm_vals.unsqueeze(-1).unsqueeze(-1) if batched else norm_vals
        assert torch.allclose(
            new_data,
            data / norm_vals,
        )
    else:
        norm_vals = torch.linalg.norm(data, norm, dim=(-2, -1))
        norm_vals = norm_vals.unsqueeze(-1).unsqueeze(-1) if batched else norm_vals
        assert torch.allclose(
            new_data,
            data / norm_vals,
        )
        

@pytest.mark.parametrize("phase_offset", [-1, -0.5, 0, 0.5, 1])
@pytest.mark.parametrize("batched", [False, True])
def test_phase_shift(phase_offset, batched):
    if batched:
        data = gen_rand_batched_1d_data(4, 16)
        new_data = F.phase_shift(data, phase_offset * torch.pi)
    else:
        data = gen_rand_1d_data(16)
        
        # Test batch size of 1 match single
        unsqueezed_data = data.unsqueeze(0)
        unsqueezed_new_data = F.phase_shift(
            unsqueezed_data, 
            phase_offset * torch.pi,
        )
        new_data = F.phase_shift(data, phase_offset * torch.pi)
        assert torch.allclose(unsqueezed_new_data.squeeze(), new_data)
    
    if phase_offset != 0:
        assert not torch.allclose(new_data, data)
    else:
        assert torch.allclose(new_data, data)
        
    assert data.shape == new_data.shape
    assert torch.allclose(
        new_data, 
        data * torch.exp(torch.tensor([1j * phase_offset * torch.pi]))
    )
    