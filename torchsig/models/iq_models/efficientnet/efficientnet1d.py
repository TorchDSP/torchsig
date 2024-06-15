import timm
from torch.nn import Linear

from torchsig.models.model_utils.model_utils_1d.conversions_to_1d import convert_2d_model_to_1d

__all__ = ["EfficientNet1d"]

def EfficientNet1d(
    input_channels: int,
    n_features: int,
    efficientnet_version: str = "b0",
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs and returns a 1d version of the EfficientNet model described in
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:

        input_channels (int):
            Number of 1d input channels; e.g., common practice is to split complex number time-series data into 2 channels, representing the real and imaginary parts respectively
        
        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        efficientnet_version (str):
            Specifies the version of efficientnet to use. See the timm efficientnet documentation for details. Examples are 'b0', 'b1', and 'b4'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    mdl = convert_2d_model_to_1d(
        timm.create_model(
            "efficientnet_" + efficientnet_version,
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    mdl.classifier = Linear(mdl.classifier.in_features, n_features)
    return mdl