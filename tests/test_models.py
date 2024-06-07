from torchsig.models.iq_models.efficientnet.efficientnet import *
from torchsig.models.iq_models.xcit.xcit import *
import pytest


@pytest.mark.parametrize("version", ("b0", "b2", "b4"))
def test_can_instantiate_efficientnet(version: str):
    if version == "b0":
        model = efficientnet_b0(pretrained=False)
    if version == "b2":
        model = efficientnet_b2(pretrained=False)
    if version == "b4":
        model = efficientnet_b4(pretrained=False)


@pytest.mark.parametrize("version", ("nano", "tiny"))
def test_can_instantiate_xcit(version: str):
    if version == "nano":
        model = xcit_nano(pretrained=False)
    if version == "tiny":
        model = xcit_tiny12(pretrained=False)
