""" Testing Generate Wideband Scripts

Examples:
    >>> pytest test_generate_wideband_scripts.py
    >>> pytest test_generate_wideband_scripts.py --pdb
"""
import generate_wideband
from torchsig.datasets import conf
import pytest
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import os, sys

configs = [
        conf.WidebandCleanTrainConfig,
        conf.WidebandCleanValConfig,
        conf.WidebandImpairedTrainConfig,
        conf.WidebandImpairedValConfig,
        conf.WidebandCleanTrainQAConfig,
        conf.WidebandCleanValQAConfig,
        conf.WidebandImpairedTrainQAConfig,
        conf.WidebandImpairedValQAConfig,
]
num_samples_small = 10
num_workers = os.cpu_count() // 2

# @pytest.mark.skip(reason="works")
def test_generate_wideband_clean_qa_train(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandCleanTrainQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="works")
def test_generate_wideband_clean_qa_val(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandCleanValQAConfig], num_workers, -1)

# @pytest.mark.parametrize('execution_number', range(10))
def test_generate_wideband_impaired_qa_train(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandImpairedTrainQAConfig], num_workers, -1)

def test_generate_wideband_impaired_qa_val(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandImpairedValQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_clean_train(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandCleanTrainConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_clean_val(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandCleanValConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_impaired_train(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandImpairedTrainConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_impaired_val(tmp_path):
    generate_wideband.generate(tmp_path, [conf.WidebandImpairedValConfig], num_workers, -1)