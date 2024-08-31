import timm
from torch.nn import Linear

from torchsig.models.model_utils.model_utils_1d.conversions_to_1d import convert_2d_model_to_1d

__all__ = ["DenseNet1d"]

def DenseNet1d(
    input_channels: int,
    n_features: int,
    densenet_version: str = "densenet121",
    drop_rate: float = 0.3,
):
    """Constructs and returns a 1d version of the DenseNet model described in
    `"DenseNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:

        input_channels (int):
            Number of 1d input channels; e.g., common practice is to split complex number time-series data into 2 channels, representing the real and imaginary parts respectively
        
        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        densenet_version (str):
            Specifies the version of densenet to use. See the timm densenet documentation for details. Examples are 'densenet121'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    mdl = convert_2d_model_to_1d(
        timm.create_model(
            densenet_version,
            in_chans=input_channels,
            drop_rate=drop_rate,
        )
    )
    mdl.classifier = Linear(mdl.classifier.in_features, n_features)
    return mdl