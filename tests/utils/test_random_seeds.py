"""Testing the random seeding functionality of the Seedable class"""

import os
import subprocess
import sys

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import Quantize
from torchsig.utils.random import Seedable, derive_keyed_seed


def test_single_seedable():
    s0 = Seedable()
    s0.seed(1776)
    val_0 = s0.random_generator.random()
    val_1 = s0.random_generator.random()
    val_2 = s0.random_generator.random()

    s0.seed(1776)
    new_val_0 = s0.random_generator.random()
    new_val_1 = s0.random_generator.random()
    new_val_2 = s0.random_generator.random()

    s1 = Seedable()
    s1.seed(1776)
    other_new_val_0 = s1.random_generator.random()
    other_new_val_1 = s1.random_generator.random()
    other_new_val_2 = s1.random_generator.random()

    assert val_0 == new_val_0
    assert val_1 == new_val_1
    assert val_2 == new_val_2

    assert val_0 == other_new_val_0
    assert val_1 == other_new_val_1
    assert val_2 == other_new_val_2


def test_parent_seedable():
    s0 = Seedable()
    s0.seed(1776)
    val_0 = s0.random_generator.random()
    val_1 = s0.random_generator.random()
    val_2 = s0.random_generator.random()

    grandfather_seedable = Seedable()
    father_seedable = Seedable(parent=grandfather_seedable)
    child_seedable = Seedable(parent=father_seedable)

    grandfather_seedable.seed(1776)

    new_val_0 = grandfather_seedable.random_generator.random()
    new_val_1 = child_seedable.random_generator.random()
    new_val_2 = father_seedable.random_generator.random()

    assert val_0 == new_val_0
    assert val_1 == new_val_1
    assert val_2 == new_val_2


def test_derive_keyed_seed_has_stable_version_one_value() -> None:
    seed = derive_keyed_seed(
        12345,
        sample_identity="sample-0042",
        impairment_identity="default-level-2",
        severity=2,
        replicate=0,
    )

    assert seed == 1393683108357541559


def test_key_components_select_independent_deterministic_seeds() -> None:
    base_key = {
        "base_seed": 7,
        "sample_identity": "sample-a",
        "impairment_identity": "quantize",
        "severity": 1,
        "replicate": 0,
    }
    expected = derive_keyed_seed(**base_key)

    assert derive_keyed_seed(**base_key) == expected
    for name, value in (
        ("base_seed", 8),
        ("sample_identity", "sample-b"),
        ("impairment_identity", "clock-jitter"),
        ("severity", 2),
        ("replicate", 1),
    ):
        changed_key = {**base_key, name: value}
        assert derive_keyed_seed(**changed_key) != expected


def test_keyed_replay_is_order_batch_and_worker_assignment_independent() -> None:
    conditions = [(sample, impairment, severity, replicate) for sample in range(8) for impairment in ("quantize", "fading") for severity in (1, 2) for replicate in range(2)]

    def realize(order: list[tuple[int, str, int, int]], batch_size: int, workers: int) -> dict[tuple[int, str, int, int], int]:
        assignments = [order[worker::workers] for worker in range(workers)]
        realized = {}
        for worker_conditions in assignments:
            for offset in range(0, len(worker_conditions), batch_size):
                for sample, impairment, severity, replicate in worker_conditions[offset : offset + batch_size]:
                    key = (sample, impairment, severity, replicate)
                    realized[key] = derive_keyed_seed(
                        91,
                        sample_identity=sample,
                        impairment_identity=impairment,
                        severity=severity,
                        replicate=replicate,
                    )
        return realized

    reference = realize(conditions, batch_size=1, workers=1)

    assert realize(list(reversed(conditions)), batch_size=7, workers=4) == reference
    assert len(set(reference.values())) == len(reference)


def test_keyed_seed_is_stable_across_process_hash_seeds() -> None:
    program = "from torchsig.utils.random import derive_keyed_seed; print(derive_keyed_seed(12345, sample_identity='sample-0042', impairment_identity='default-level-2', severity=2, replicate=0))"
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(result.stdout.strip())

    assert outputs == ["1393683108357541559", "1393683108357541559"]


def test_keyed_replay_restores_sequential_rng_hierarchy() -> None:
    pipeline = Seedable(seed=1776)
    child = Seedable(parent=pipeline)
    control = Seedable(seed=1776)

    assert pipeline.random_generator.random() == control.random_generator.random()
    with pipeline.keyed_replay(
        42,
        sample_identity=11,
        impairment_identity="pipeline",
        severity="high",
        replicate=3,
    ) as first_seed:
        first_values = (pipeline.random_generator.random(), child.random_generator.random())
    assert pipeline.random_generator.random() == control.random_generator.random()

    with pipeline.keyed_replay(
        42,
        sample_identity=11,
        impairment_identity="pipeline",
        severity="high",
        replicate=3,
    ) as repeated_seed:
        repeated_values = (pipeline.random_generator.random(), child.random_generator.random())

    assert repeated_seed == first_seed
    assert repeated_values == first_values


def test_keyed_replay_reproduces_transform_output() -> None:
    transform = Quantize(seed=5)
    clean = Signal(data=(np.linspace(-0.5, 0.5, 256) + 0.25j).astype(np.complex64))

    def apply(replicate: int) -> np.ndarray:
        with transform.keyed_replay(
            123,
            sample_identity="sample-9",
            impairment_identity="quantize",
            severity=1,
            replicate=replicate,
        ):
            return transform(clean.copy()).data

    np.testing.assert_array_equal(apply(0), apply(0))
    assert not np.array_equal(apply(0), apply(1))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"base_seed": -1}, ValueError, "base_seed"),
        ({"sample_identity": True}, TypeError, "sample_identity"),
        ({"impairment_identity": ""}, ValueError, "impairment_identity"),
        ({"severity": float("nan")}, ValueError, "finite"),
        ({"replicate": -1}, ValueError, "replicate"),
    ],
)
def test_derive_keyed_seed_rejects_invalid_keys(kwargs: dict, error: type[Exception], message: str) -> None:
    key = {
        "base_seed": 1,
        "sample_identity": "sample",
        "impairment_identity": "impairment",
        "severity": 1,
        "replicate": 0,
    }
    key.update(kwargs)

    with pytest.raises(error, match=message):
        derive_keyed_seed(**key)
