""" Testing Generate Wideband Sig53 Scripts

Examples:
    >>> pytest test_generate_wideband_sig53_scripts.py
    >>> pytest test_generate_wideband_sig53_scripts.py --pdb
"""
import generate_wideband_sig53
from torchsig.datasets import conf
import pytest
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import os, sys

configs = [
        conf.WidebandSig53CleanTrainConfig,
        conf.WidebandSig53CleanValConfig,
        conf.WidebandSig53ImpairedTrainConfig,
        conf.WidebandSig53ImpairedValConfig,
        conf.WidebandSig53CleanTrainQAConfig,
        conf.WidebandSig53CleanValQAConfig,
        conf.WidebandSig53ImpairedTrainQAConfig,
        conf.WidebandSig53ImpairedValQAConfig,
]
num_samples_small = 10
num_workers = os.cpu_count() // 2

# @pytest.mark.skip(reason="works")
def test_generate_wideband_sig53_clean_qa_train(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53CleanTrainQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="works")
def test_generate_wideband_sig53_clean_qa_val(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53CleanValQAConfig], num_workers, -1)

# @pytest.mark.parametrize('execution_number', range(10))
def test_generate_wideband_sig53_impaired_qa_train(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53ImpairedTrainQAConfig], num_workers, -1)

def test_generate_wideband_sig53_impaired_qa_val(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53ImpairedValQAConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_sig53_clean_train(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53CleanTrainConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_sig53_clean_val(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53CleanValConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_sig53_impaired_train(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53ImpairedTrainConfig], num_workers, -1)

# @pytest.mark.skip(reason="too big")
def test_generate_wideband_sig53_impaired_val(tmp_path):
    generate_wideband_sig53.generate(tmp_path, [conf.WidebandSig53ImpairedValConfig], num_workers, -1)