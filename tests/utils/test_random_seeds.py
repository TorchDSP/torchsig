""" Testing the random seeding functionality of the Seedable class
"""

import pytest

from torchsig.utils.random import Seedable


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

