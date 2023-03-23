import numpy as np
from copy import deepcopy
from typing import Optional, Any, Union, List
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.types import SignalData, SignalDescription
from torchsig.transforms.transforms import SignalTransform
from torchsig.transforms.spectrogram_transforms import functional
from torchsig.transforms.functional import (
    NumericParameter,
    FloatParameter,
    IntParameter,
)
from torchsig.transforms.functional import (
    to_distribution,
    uniform_continuous_distribution,
    uniform_discrete_distribution,
)


class SpectrogramDropSamples(SignalTransform):
    """Randomly drop samples from the input data of specified durations and
    with specified fill techniques:
    * `ffill` (front fill): replace drop samples with the last previous value
    * `bfill` (back fill): replace drop samples with the next value
    * `mean`: replace drop samples with the mean value of the full data
    * `zero`: replace drop samples with zeros
    * `low`: replace drop samples with low power samples
    * `min`: replace drop samples with the minimum of the absolute power
    * `max`: replace drop samples with the maximum of the absolute power
    * `ones`: replace drop samples with ones

    Transform is based off of the
    `TSAug Dropout Transform <https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/dropout.py>`_.

    Args:
         drop_rate (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            drop_rate sets the rate at which to drop samples
            * If Callable, produces a sample by calling drop_rate()
            * If int or float, drop_rate is fixed at the value provided
            * If list, drop_rate is any element in the list
            * If tuple, drop_rate is in range of (tuple[0], tuple[1])

        size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            size sets the size of each instance of dropped samples
            * If Callable, produces a sample by calling size()
            * If int or float, size is fixed at the value provided
            * If list, size is any element in the list
            * If tuple, size is in range of (tuple[0], tuple[1])

        fill (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            fill sets the method of how the dropped samples should be filled
            * If Callable, produces a sample by calling fill()
            * If list, fill is any element in the list
            * If str, fill is fixed at the method provided

    """

    def __init__(
        self,
        drop_rate: NumericParameter = uniform_continuous_distribution(0.001, 0.005),
        size: NumericParameter = uniform_discrete_distribution(np.arange(1, 10)),
        fill: Union[List, str] = uniform_discrete_distribution(
            ["ffill", "bfill", "mean", "zero", "low", "min", "max", "ones"]
        ),
    ):
        super(SpectrogramDropSamples, self).__init__()
        self.drop_rate = to_distribution(drop_rate, self.random_generator)
        self.size = to_distribution(size, self.random_generator)
        self.fill = to_distribution(fill, self.random_generator)

    def __call__(self, data: Any) -> Any:
        drop_rate = self.drop_rate()
        fill = self.fill()

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.float64),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            channels, height, width = data.iq_data.shape
            spec_size = height * width
            drop_instances = int(spec_size * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(
                1, spec_size - max(drop_sizes) - 1, drop_instances
            ).astype(int)

            new_data.iq_data = functional.drop_spec_samples(
                data.iq_data, drop_starts, drop_sizes, fill
            )

        else:
            drop_instances = int(data.shape[0] * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(
                0, data.shape[0] - max(drop_sizes), drop_instances
            ).astype(int)

            new_data = functional.drop_spec_samples(data, drop_starts, drop_sizes, fill)
        return new_data


class SpectrogramPatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `PatchShuffle Regularization <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            patch_size sets the size of each patch to shuffle
            * If Callable, produces a sample by calling patch_size()
            * If int or float, patch_size is fixed at the value provided
            * If list, patch_size is any element in the list
            * If tuple, patch_size is in range of (tuple[0], tuple[1])

        shuffle_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            shuffle_ratio sets the ratio of the patches to shuffle
            * If Callable, produces a sample by calling shuffle_ratio()
            * If int or float, shuffle_ratio is fixed at the value provided
            * If list, shuffle_ratio is any element in the list
            * If tuple, shuffle_ratio is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        patch_size: NumericParameter = uniform_continuous_distribution(2, 16),
        shuffle_ratio: FloatParameter = uniform_continuous_distribution(0.01, 0.10),
    ):
        super(SpectrogramPatchShuffle, self).__init__()
        self.patch_size = to_distribution(patch_size, self.random_generator)
        self.shuffle_ratio = to_distribution(shuffle_ratio, self.random_generator)

    def __call__(self, data: Any) -> Any:
        patch_size = int(self.patch_size())
        shuffle_ratio = self.shuffle_ratio()

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            new_data.iq_data = functional.spec_patch_shuffle(
                data.iq_data, patch_size, shuffle_ratio
            )
        else:
            new_data = functional.spec_patch_shuffle(data, patch_size, shuffle_ratio)
        return new_data


class SpectrogramTranslation(SignalTransform):
    """Transform that inputs a spectrogram and applies a random time/freq
    translation

    Args:
         time_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            time_shift sets the translation along the time-axis
            * If Callable, produces a sample by calling time_shift()
            * If int, time_shift is fixed at the value provided
            * If list, time_shift is any element in the list
            * If tuple, time_shift is in range of (tuple[0], tuple[1])

        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            freq_shift sets the translation along the freq-axis
            * If Callable, produces a sample by calling freq_shift()
            * If int, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        time_shift: IntParameter = uniform_continuous_distribution(-128, 128),
        freq_shift: IntParameter = uniform_continuous_distribution(-128, 128),
    ):
        super(SpectrogramTranslation, self).__init__()
        self.time_shift = to_distribution(time_shift, self.random_generator)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)

    def __call__(self, data: Any) -> Any:
        time_shift = int(self.time_shift())
        freq_shift = int(self.freq_shift())

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            new_data.iq_data = functional.spec_translate(
                data.iq_data, time_shift, freq_shift
            )

            # Update SignalDescription
            new_signal_description = []
            signal_description = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)

                # Update time fields
                new_signal_desc.start = (
                    new_signal_desc.start + time_shift / new_data.iq_data.shape[1]
                )
                new_signal_desc.stop = (
                    new_signal_desc.stop + time_shift / new_data.iq_data.shape[1]
                )
                if new_signal_desc.start >= 1.0 or new_signal_desc.stop <= 0.0:
                    continue
                new_signal_desc.start = (
                    0.0 if new_signal_desc.start < 0.0 else new_signal_desc.start
                )
                new_signal_desc.stop = (
                    1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                )
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start

                # Trim any out-of-capture freq values
                new_signal_desc.lower_frequency = (
                    -0.5
                    if new_signal_desc.lower_frequency < -0.5
                    else new_signal_desc.lower_frequency
                )
                new_signal_desc.upper_frequency = (
                    0.5
                    if new_signal_desc.upper_frequency > 0.5
                    else new_signal_desc.upper_frequency
                )

                # Update freq fields
                new_signal_desc.lower_frequency = (
                    new_signal_desc.lower_frequency
                    + freq_shift / new_data.iq_data.shape[2]
                )
                new_signal_desc.upper_frequency = (
                    new_signal_desc.upper_frequency
                    + freq_shift / new_data.iq_data.shape[2]
                )
                if (
                    new_signal_desc.lower_frequency >= 0.5
                    or new_signal_desc.upper_frequency <= -0.5
                ):
                    continue
                new_signal_desc.lower_frequency = (
                    -0.5
                    if new_signal_desc.lower_frequency < -0.5
                    else new_signal_desc.lower_frequency
                )
                new_signal_desc.upper_frequency = (
                    0.5
                    if new_signal_desc.upper_frequency > 0.5
                    else new_signal_desc.upper_frequency
                )
                new_signal_desc.bandwidth = (
                    new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                )
                new_signal_desc.center_frequency = (
                    new_signal_desc.lower_frequency + new_signal_desc.bandwidth * 0.5
                )

                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)

            # Set output data's SignalDescription to above list
            new_data.signal_description = new_signal_description

        else:
            new_data = functional.spec_translate(data, time_shift, freq_shift)
        return new_data


class SpectrogramMosaicCrop(SignalTransform):
    """The SpectrogramMosaicCrop transform takes the original input tensor and
    inserts it randomly into one cell of a 2x2 grid of 2x the size of the
    orginal spectrogram input. The `dataset` argument is then read 3x to
    retrieve spectrograms to fill the remaining cells of the 2x2 grid. Finally,
    the 2x larger stitched view of 4x spectrograms is randomly cropped to the
    original target size, containing pieces of each of the 4x stitched
    spectrograms.

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the mosaic operation

    """

    def __init__(self, dataset: SignalDataset = None):
        super(SpectrogramMosaicCrop, self).__init__()
        self.dataset = dataset

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Read shapes
            channels, height, width = data.iq_data.shape

            # Randomly decide the new x0, y0 point of the stitched images
            x0 = np.random.randint(0, width)
            y0 = np.random.randint(0, height)

            # Initialize new SignalDescription object
            new_signal_description = []

            # First, create a 2x2 grid of (512+512,512+512) and randomly put the initial data into a grid cell
            cell_idx = np.random.randint(0, 4)
            x_idx = 0 if cell_idx == 0 or cell_idx == 2 else 1
            y_idx = 0 if cell_idx == 0 or cell_idx == 1 else 1
            full_mosaic = np.empty(
                (channels, height * 2, width * 2),
                dtype=data.iq_data.dtype,
            )
            full_mosaic[
                :,
                y_idx * height : (y_idx + 1) * height,
                x_idx * width : (x_idx + 1) * width,
            ] = data.iq_data

            # Update original data's SignalDescription objects given the cell index
            signal_description = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)

                # Update time fields
                if x_idx == 0:
                    if new_signal_desc.stop * width < x0:
                        continue
                    new_signal_desc.start = (
                        0
                        if new_signal_desc.start < (x0 / width)
                        else new_signal_desc.start - (x0 / width)
                    )
                    new_signal_desc.stop = (
                        new_signal_desc.stop - (x0 / width)
                        if new_signal_desc.stop < 1.0
                        else 1.0 - (x0 / width)
                    )
                    new_signal_desc.duration = (
                        new_signal_desc.stop - new_signal_desc.start
                    )

                else:
                    if new_signal_desc.start * width > x0:
                        continue
                    new_signal_desc.start = (width - x0) / width + new_signal_desc.start
                    new_signal_desc.stop = (width - x0) / width + new_signal_desc.stop
                    new_signal_desc.stop = (
                        1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                    )
                    new_signal_desc.duration = (
                        new_signal_desc.stop - new_signal_desc.start
                    )

                # Update frequency fields
                new_signal_desc.lower_frequency = (
                    -0.5
                    if new_signal_desc.lower_frequency < -0.5
                    else new_signal_desc.lower_frequency
                )
                new_signal_desc.upper_frequency = (
                    0.5
                    if new_signal_desc.upper_frequency > 0.5
                    else new_signal_desc.upper_frequency
                )
                if y_idx == 0:
                    if (new_signal_desc.upper_frequency + 0.5) * height < y0:
                        continue
                    new_signal_desc.lower_frequency = (
                        -0.5
                        if (new_signal_desc.lower_frequency + 0.5) < (y0 / height)
                        else new_signal_desc.lower_frequency - (y0 / height)
                    )
                    new_signal_desc.upper_frequency = (
                        new_signal_desc.upper_frequency - (y0 / height)
                        if new_signal_desc.upper_frequency < 0.5
                        else 0.5 - (y0 / height)
                    )
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency
                        + new_signal_desc.bandwidth * 0.5
                    )

                else:
                    if (new_signal_desc.lower_frequency + 0.5) * height > y0:
                        continue
                    new_signal_desc.lower_frequency = (
                        height - y0
                    ) / height + new_signal_desc.lower_frequency
                    new_signal_desc.upper_frequency = (
                        height - y0
                    ) / height + new_signal_desc.upper_frequency
                    new_signal_desc.upper_frequency = (
                        0.5
                        if new_signal_desc.upper_frequency > 0.5
                        else new_signal_desc.upper_frequency
                    )
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency
                        + new_signal_desc.bandwidth * 0.5
                    )

                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)

            # Next, fill in the remaining cells with data randomly sampled from the input dataset
            for cell_i in range(4):
                if cell_i == cell_idx:
                    # Skip if the original data's cell
                    continue
                x_idx = 0 if cell_i == 0 or cell_i == 2 else 1
                y_idx = 0 if cell_i == 0 or cell_i == 1 else 1
                dataset_idx = np.random.randint(len(self.dataset))
                curr_data, curr_signal_desc = self.dataset[dataset_idx]
                full_mosaic[
                    :,
                    y_idx * height : (y_idx + 1) * height,
                    x_idx * width : (x_idx + 1) * width,
                ] = curr_data

                # Update inserted data's SignalDescription objects given the cell index
                signal_description = (
                    [curr_signal_desc]
                    if isinstance(curr_signal_desc, SignalDescription)
                    else curr_signal_desc
                )
                for signal_desc in signal_description:
                    new_signal_desc = deepcopy(signal_desc)

                    # Update time fields
                    if x_idx == 0:
                        if new_signal_desc.stop * width < x0:
                            continue
                        new_signal_desc.start = (
                            0
                            if new_signal_desc.start < (x0 / width)
                            else new_signal_desc.start - (x0 / width)
                        )
                        new_signal_desc.stop = (
                            new_signal_desc.stop - (x0 / width)
                            if new_signal_desc.stop < 1.0
                            else 1.0 - (x0 / width)
                        )
                        new_signal_desc.duration = (
                            new_signal_desc.stop - new_signal_desc.start
                        )

                    else:
                        if new_signal_desc.start * width > x0:
                            continue
                        new_signal_desc.start = (
                            width - x0
                        ) / width + new_signal_desc.start
                        new_signal_desc.stop = (
                            width - x0
                        ) / width + new_signal_desc.stop
                        new_signal_desc.stop = (
                            1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                        )
                        new_signal_desc.duration = (
                            new_signal_desc.stop - new_signal_desc.start
                        )

                    # Update frequency fields
                    new_signal_desc.lower_frequency = (
                        -0.5
                        if new_signal_desc.lower_frequency < -0.5
                        else new_signal_desc.lower_frequency
                    )
                    new_signal_desc.upper_frequency = (
                        0.5
                        if new_signal_desc.upper_frequency > 0.5
                        else new_signal_desc.upper_frequency
                    )
                    if y_idx == 0:
                        if (new_signal_desc.upper_frequency + 0.5) * height < y0:
                            continue
                        new_signal_desc.lower_frequency = (
                            -0.5
                            if (new_signal_desc.lower_frequency + 0.5) < (y0 / height)
                            else new_signal_desc.lower_frequency - (y0 / height)
                        )
                        new_signal_desc.upper_frequency = (
                            new_signal_desc.upper_frequency - (y0 / height)
                            if new_signal_desc.upper_frequency < 0.5
                            else 0.5 - (y0 / height)
                        )
                        new_signal_desc.bandwidth = (
                            new_signal_desc.upper_frequency
                            - new_signal_desc.lower_frequency
                        )
                        new_signal_desc.center_frequency = (
                            new_signal_desc.lower_frequency
                            + new_signal_desc.bandwidth * 0.5
                        )

                    else:
                        if (new_signal_desc.lower_frequency + 0.5) * height > y0:
                            continue
                        new_signal_desc.lower_frequency = (
                            height - y0
                        ) / height + new_signal_desc.lower_frequency
                        new_signal_desc.upper_frequency = (
                            height - y0
                        ) / height + new_signal_desc.upper_frequency
                        new_signal_desc.upper_frequency = (
                            0.5
                            if new_signal_desc.upper_frequency > 0.5
                            else new_signal_desc.upper_frequency
                        )
                        new_signal_desc.bandwidth = (
                            new_signal_desc.upper_frequency
                            - new_signal_desc.lower_frequency
                        )
                        new_signal_desc.center_frequency = (
                            new_signal_desc.lower_frequency
                            + new_signal_desc.bandwidth * 0.5
                        )

                    # Append SignalDescription to list
                    new_signal_description.append(new_signal_desc)

            # After the data has been stitched into the large 2x2 gride, crop using x0, y0
            new_data.iq_data = full_mosaic[:, y0 : y0 + height, x0 : x0 + width]

            # Set output data's SignalDescription to above list
            new_data.signal_description = new_signal_description

        else:
            # Read shapes
            channels, height, width = data.shape

            # Randomly decide the new x0, y0 point of the stitched images
            x0 = np.random.randint(0, width)
            y0 = np.random.randint(0, height)

            # Initialize new SignalDescription object
            new_signal_description = []

            # First, create a 2x2 grid of (512+512,512+512) and randomly put the initial data into a grid cell
            cell_idx = np.random.randint(0, 4)
            x_idx = 0 if cell_idx == 0 or cell_idx == 2 else 1
            y_idx = 0 if cell_idx == 0 or cell_idx == 1 else 1
            full_mosaic = np.empty(
                (channels, height * 2, width * 2),
                dtype=data.dtype,
            )
            full_mosaic[
                :,
                y_idx * height : (y_idx + 1) * height,
                x_idx * width : (x_idx + 1) * width,
            ] = data

            # Next, fill in the remaining cells with data randomly sampled from the input dataset
            for cell_i in range(4):
                if cell_i == cell_idx:
                    # Skip if the original data's cell
                    continue
                x_idx = 0 if cell_i == 0 or cell_i == 2 else 1
                y_idx = 0 if cell_i == 0 or cell_i == 1 else 1
                dataset_idx = np.random.randint(len(self.dataset))
                curr_data, curr_signal_desc = self.dataset[dataset_idx]
                full_mosaic[
                    :,
                    y_idx * height : (y_idx + 1) * height,
                    x_idx * width : (x_idx + 1) * width,
                ] = curr_data

            # After the data has been stitched into the large 2x2 gride, crop using x0, y0
            new_data = full_mosaic[:, y0 : y0 + height, x0 : x0 + width]

        return new_data


class SpectrogramMosaicDownsample(SignalTransform):
    """The SpectrogramMosaicDownsample transform takes the original input
    tensor and inserts it randomly into one cell of a 2x2 grid of 2x the size
    of the orginal spectrogram input. The `dataset` argument is then read 3x to
    retrieve spectrograms to fill the remaining cells of the 2x2 grid. Finally,
    the 2x oversized stitched spectrograms are downsampled by 2 to become the
    desired, original shape

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the mosaic operation

    """

    def __init__(self, dataset: SignalDataset = None):
        super(SpectrogramMosaicDownsample, self).__init__()
        self.dataset = dataset

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Read shapes
            channels, height, width = data.iq_data.shape

            # Initialize new SignalDescription object
            new_signal_description = []

            # First, create a 2x2 grid of (512+512,512+512) and randomly put the initial data into a grid cell
            cell_idx = np.random.randint(0, 4)
            x_idx = 0 if cell_idx == 0 or cell_idx == 2 else 1
            y_idx = 0 if cell_idx == 0 or cell_idx == 1 else 1
            full_mosaic = np.empty(
                (channels, height * 2, width * 2),
                dtype=data.iq_data.dtype,
            )
            full_mosaic[
                :,
                y_idx * height : (y_idx + 1) * height,
                x_idx * width : (x_idx + 1) * width,
            ] = data.iq_data

            # Update original data's SignalDescription objects given the cell index
            signal_description = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)

                # Update time fields
                if x_idx == 0:
                    new_signal_desc.start /= 2
                    new_signal_desc.stop /= 2
                    new_signal_desc.duration = (
                        new_signal_desc.stop - new_signal_desc.start
                    )

                else:
                    new_signal_desc.start = new_signal_desc.start / 2 + 0.5
                    new_signal_desc.stop = new_signal_desc.stop / 2 + 0.5
                    new_signal_desc.duration = (
                        new_signal_desc.stop - new_signal_desc.start
                    )

                # Update frequency fields
                new_signal_desc.lower_frequency = (
                    -0.5
                    if new_signal_desc.lower_frequency < -0.5
                    else new_signal_desc.lower_frequency
                )
                new_signal_desc.upper_frequency = (
                    0.5
                    if new_signal_desc.upper_frequency > 0.5
                    else new_signal_desc.upper_frequency
                )
                if y_idx == 0:
                    new_signal_desc.lower_frequency = (
                        new_signal_desc.lower_frequency + 0.5
                    ) / 2 - 0.5
                    new_signal_desc.upper_frequency = (
                        new_signal_desc.upper_frequency + 0.5
                    ) / 2 - 0.5
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency
                        + new_signal_desc.bandwidth * 0.5
                    )

                else:
                    new_signal_desc.lower_frequency = (
                        new_signal_desc.lower_frequency + 0.5
                    ) / 2
                    new_signal_desc.upper_frequency = (
                        new_signal_desc.upper_frequency + 0.5
                    ) / 2
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency
                        + new_signal_desc.bandwidth * 0.5
                    )

                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)

            # Next, fill in the remaining cells with data randomly sampled from the input dataset
            for cell_i in range(4):
                if cell_i == cell_idx:
                    # Skip if the original data's cell
                    continue
                x_idx = 0 if cell_i == 0 or cell_i == 2 else 1
                y_idx = 0 if cell_i == 0 or cell_i == 1 else 1
                dataset_idx = np.random.randint(len(self.dataset))
                curr_data, curr_signal_desc = self.dataset[dataset_idx]
                full_mosaic[
                    :,
                    y_idx * height : (y_idx + 1) * height,
                    x_idx * width : (x_idx + 1) * width,
                ] = curr_data

                # Update inserted data's SignalDescription objects given the cell index
                signal_description = (
                    [curr_signal_desc]
                    if isinstance(curr_signal_desc, SignalDescription)
                    else curr_signal_desc
                )
                for signal_desc in signal_description:
                    new_signal_desc = deepcopy(signal_desc)

                    # Update time fields
                    if x_idx == 0:
                        new_signal_desc.start /= 2
                        new_signal_desc.stop /= 2
                        new_signal_desc.duration = (
                            new_signal_desc.stop - new_signal_desc.start
                        )

                    else:
                        new_signal_desc.start = new_signal_desc.start / 2 + 0.5
                        new_signal_desc.stop = new_signal_desc.stop / 2 + 0.5
                        new_signal_desc.duration = (
                            new_signal_desc.stop - new_signal_desc.start
                        )

                    # Update frequency fields
                    new_signal_desc.lower_frequency = (
                        -0.5
                        if new_signal_desc.lower_frequency < -0.5
                        else new_signal_desc.lower_frequency
                    )
                    new_signal_desc.upper_frequency = (
                        0.5
                        if new_signal_desc.upper_frequency > 0.5
                        else new_signal_desc.upper_frequency
                    )
                    if y_idx == 0:
                        new_signal_desc.lower_frequency = (
                            new_signal_desc.lower_frequency + 0.5
                        ) / 2 - 0.5
                        new_signal_desc.upper_frequency = (
                            new_signal_desc.upper_frequency + 0.5
                        ) / 2 - 0.5
                        new_signal_desc.bandwidth = (
                            new_signal_desc.upper_frequency
                            - new_signal_desc.lower_frequency
                        )
                        new_signal_desc.center_frequency = (
                            new_signal_desc.lower_frequency
                            + new_signal_desc.bandwidth * 0.5
                        )

                    else:
                        new_signal_desc.lower_frequency = (
                            new_signal_desc.lower_frequency + 0.5
                        ) / 2
                        new_signal_desc.upper_frequency = (
                            new_signal_desc.upper_frequency + 0.5
                        ) / 2
                        new_signal_desc.bandwidth = (
                            new_signal_desc.upper_frequency
                            - new_signal_desc.lower_frequency
                        )
                        new_signal_desc.center_frequency = (
                            new_signal_desc.lower_frequency
                            + new_signal_desc.bandwidth * 0.5
                        )

                    # Append SignalDescription to list
                    new_signal_description.append(new_signal_desc)

            # After the data has been stitched into the large 2x2 gride, downsample by 2
            new_data.iq_data = full_mosaic[:, ::2, ::2]

            # Set output data's SignalDescription to above list
            new_data.signal_description = new_signal_description

        else:
            # Read shapes
            channels, height, width = data.shape

            # Initialize new SignalDescription object
            new_signal_description = []

            # First, create a 2x2 grid of (512+512,512+512) and randomly put the initial data into a grid cell
            cell_idx = np.random.randint(0, 4)
            x_idx = 0 if cell_idx == 0 or cell_idx == 2 else 1
            y_idx = 0 if cell_idx == 0 or cell_idx == 1 else 1
            full_mosaic = np.empty(
                (channels, height * 2, width * 2),
                dtype=data.dtype,
            )
            full_mosaic[
                :,
                y_idx * height : (y_idx + 1) * height,
                x_idx * width : (x_idx + 1) * width,
            ] = data

            # Next, fill in the remaining cells with data randomly sampled from the input dataset
            for cell_i in range(4):
                if cell_i == cell_idx:
                    # Skip if the original data's cell
                    continue
                x_idx = 0 if cell_i == 0 or cell_i == 2 else 1
                y_idx = 0 if cell_i == 0 or cell_i == 1 else 1
                dataset_idx = np.random.randint(len(self.dataset))
                curr_data, curr_signal_desc = self.dataset[dataset_idx]
                full_mosaic[
                    :,
                    y_idx * height : (y_idx + 1) * height,
                    x_idx * width : (x_idx + 1) * width,
                ] = curr_data

            # After the data has been stitched into the large 2x2 gride, downsample by 2
            new_data = full_mosaic[:, ::2, ::2]

        return new_data
