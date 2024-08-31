""" Testing Generate Sig53 Scripts
"""
import generate_sig53
from torchsig.datasets import conf
import pytest
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import os, sys

configs = [
    conf.Sig53CleanTrainConfig,
    conf.Sig53CleanValConfig,
    conf.Sig53ImpairedTrainConfig,
    conf.Sig53ImpairedValConfig,
    conf.Sig53CleanTrainQAConfig,
    conf.Sig53CleanValQAConfig,
    conf.Sig53ImpairedTrainQAConfig,
    conf.Sig53ImpairedValQAConfig,
]
num_samples_small = 10
num_workers = os.cpu_count() // 2

def test_generate_sig53_clean_qa_train(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53CleanTrainQAConfig], num_workers, -1)

def test_generate_sig53_clean_qa_val(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53CleanValQAConfig], num_workers, -1)

def test_generate_sig53_impaired_qa_train(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53ImpairedTrainQAConfig], num_workers, -1)

def test_generate_sig53_impaired_qa_val(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53ImpairedValQAConfig], num_workers, -1)

def test_generate_sig53_clean_train(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53CleanTrainConfig], num_workers, -1)

def test_generate_sig53_clean_val(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53CleanValConfig], num_workers, -1)

def test_generate_sig53_impaired_train(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53ImpairedTrainConfig], num_workers, -1)

def test_generate_sig53_impaired_val(tmp_path):
    generate_sig53.generate(tmp_path, [conf.Sig53ImpairedValConfig], num_workers, -1)