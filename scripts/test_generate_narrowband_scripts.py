""" Testing Generate Narrowband Scripts

Examples:
    >>> pytest test_generate_narrowband_scripts.py
    >>> pytest test_generate_narrowband_scripts.py --pdb
"""
import generate_narrowband
from torchsig.datasets import conf
import pytest
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import os, sys

configs = [
    conf.NarrowbandCleanTrainConfig,
    conf.NarrowbandCleanValConfig,
    conf.NarrowbandImpairedTrainConfig,
    conf.NarrowbandImpairedValConfig,
    conf.NarrowbandCleanTrainQAConfig,
    conf.NarrowbandCleanValQAConfig,
    conf.NarrowbandImpairedTrainQAConfig,
    conf.NarrowbandImpairedValQAConfig,
]
num_samples_small = 10
num_workers = os.cpu_count() // 2

# @pytest.mark.skip(reason="works")
def test_generate_narrowband_clean_qa_train(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandCleanTrainQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="works")
def test_generate_narrowband_clean_qa_val(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandCleanValQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="works")
def test_generate_narrowband_impaired_qa_train(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandImpairedTrainQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="works")
def test_generate_narrowband_impaired_qa_val(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandImpairedValQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_narrowband_clean_train(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandCleanTrainConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_narrowband_clean_val(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandCleanValConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_narrowband_impaired_train(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandImpairedTrainConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_narrowband_impaired_val(tmp_path):
    generate_narrowband.generate(tmp_path, [conf.NarrowbandImpairedValConfig], num_workers, -1)