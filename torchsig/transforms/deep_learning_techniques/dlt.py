import numpy as np
from copy import deepcopy
from typing import List, Any, Union, Callable

from torchsig.utils import SignalDescription, SignalData, SignalDataset
from torchsig.transforms.transforms import SignalTransform
from torchsig.transforms.wireless_channel import TargetSNR
from torchsig.transforms.functional import (
    to_distribution,
    uniform_continuous_distribution,
    uniform_discrete_distribution,
)
from torchsig.transforms.functional import (
    NumericParameter,
    FloatParameter,
    IntParameter,
)
from torchsig.transforms.deep_learning_techniques import functional
from torchsig.transforms.expert_feature import functional as eft_f


class DatasetBasebandMixUp(SignalTransform):
    """Signal Transform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using the TargetSNR transform and the
    additional `alpha` input to set the difference in SNRs between the two
    examples with the following relationship:

       mixup_sample_snr = main_sample_snr + alpha

    Note that `alpha` is used as an additive value because the SNR values are
    expressed in log scale. Typical usage will be with with alpha values less
    than zero.

    This transform is loosely based on
    `"mixup: Beyond Emperical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_.


    Args:
        dataset :obj:`SignalDataset`:
            A SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import ModulationsDataset
        >>> # Add signals from the `ModulationsDataset`
        >>> target_transform = SignalDescriptionPassThroughTransform()
        >>> dataset = ModulationsDataset(
                            use_class_idx=True,
                            level=0,
                            num_iq_samples=4096,
                            num_samples=5300,
                            target_transform=target_transform,
                            )
        >>> transform = ST.DatasetBasebandMixUp(dataset=dataset,alpha=(-5,-3))

    """

    def __init__(
        self,
        dataset: SignalDataset = None,
        alpha: NumericParameter = uniform_continuous_distribution(-5, -3),
    ):
        super(DatasetBasebandMixUp, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Input checks
            if len(data.signal_description) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected.".format(
                        len(data.signal_description)
                    )
                )

            # Calculate target SNR of signal to be inserted
            target_snr_db = data.signal_description[0].snr + alpha

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if insert_data.shape[0] != data.iq_data.shape[0]:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_data.shape[0], data.shape[0]
                    )
                )
            insert_signal_data = SignalData(
                data=insert_data,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=insert_signal_description,
            )

            # Set insert data's SNR
            target_snr_transform = TargetSNR(target_snr_db)
            insert_signal_data = target_snr_transform(insert_signal_data)

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data + insert_signal_data.iq_data

            # Update SignalDescription
            new_signal_description = []
            new_signal_description.append(data.signal_description[0])
            new_signal_description.append(insert_signal_data.signal_description[0])
            new_data.signal_description = new_signal_description

            return new_data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `SignalDatasetBasebandMixUp` transform depends on metadata from a `SignalData` object.".format(
                    type(data)
                )
            )


class DatasetBasebandCutMix(SignalTransform):
    """Signal Transform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using the TargetSNR transform to match
    the main dataset's examples' SNR and an additional `alpha` input to set the
    relative quantity in time to occupy, where

       cutmix_num_iq_samples = total_num_iq_samples * alpha

    With this transform, the inserted signal replaces the IQ samples of the
    original signal rather than adding to them as the `DatasetBasebandMixUp`
    transform does above.

    This transform is loosely based on
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" <https://arxiv.org/pdf/1905.04899.pdf>`_.

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import ModulationsDataset
        >>> # Add signals from the `ModulationsDataset`
        >>> target_transform = SignalDescriptionPassThroughTransform()
        >>> dataset = ModulationsDataset(
                            use_class_idx=True,
                            level=0,
                            num_iq_samples=4096,
                            num_samples=5300,
                            target_transform=target_transform,
                            )
        >>> transform = ST.DatasetBasebandCutMix(dataset=dataset,alpha=(0.2,0.5))

    """

    def __init__(
        self,
        dataset: SignalDataset = None,
        alpha: NumericParameter = uniform_continuous_distribution(0.2, 0.5),
    ):
        super(DatasetBasebandCutMix, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Input checks
            if len(data.signal_description) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected.".format(
                        len(data.signal_description)
                    )
                )

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            num_iq_samples = data.iq_data.shape[0]
            if insert_data.shape[0] != num_iq_samples:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_data.shape[0], data.shape[0]
                    )
                )
            insert_signal_data = SignalData(
                data=insert_data,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=insert_signal_description,
            )

            # Set insert data's SNR
            target_snr_transform = TargetSNR(data.signal_description[0].snr)
            insert_signal_data = target_snr_transform(insert_signal_data)

            # Mask both data examples based on alpha and a random start value
            insert_num_iq_samples = int(alpha * num_iq_samples)
            insert_start = np.random.randint(num_iq_samples - insert_num_iq_samples)
            insert_stop = insert_start + insert_num_iq_samples
            data.iq_data[insert_start:insert_stop] = 0
            insert_signal_data.iq_data[:insert_start] = 0
            insert_signal_data.iq_data[insert_stop:] = 0

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data + insert_signal_data.iq_data

            # Update SignalDescription
            new_signal_description = []
            if insert_start != 0 and insert_stop != num_iq_samples:
                # Data description becomes two SignalDescriptions
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = 0.0
                new_signal_desc.stop = insert_start / num_iq_samples
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = insert_stop / num_iq_samples
                new_signal_desc.stop = 1.0
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            elif insert_start == 0:
                # Data description remains one SignalDescription up to end
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = insert_stop / num_iq_samples
                new_signal_desc.stop = 1.0
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            else:
                # Data description remains one SignalDescription at beginning
                new_signal_desc = deepcopy(data.signal_description[0])
                new_signal_desc.start = 0.0
                new_signal_desc.stop = insert_start / num_iq_samples
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            # Repeat for insert's SignalDescription
            new_signal_desc = deepcopy(insert_signal_data.signal_description[0])
            new_signal_desc.start = insert_start / num_iq_samples
            new_signal_desc.stop = insert_stop / num_iq_samples
            new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
            new_signal_description.append(new_signal_desc)

            # Set output data's SignalDescription to above list
            new_data.signal_description = new_signal_description

            return new_data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `SignalDatasetBasebandCutMix` transform depends on metadata from a `SignalData` object.".format(
                    type(data)
                )
            )


class CutOut(SignalTransform):
    """A transform that applies the CutOut transform in the time domain. The
    `cut_dur` input specifies how long the cut region should be, and the
    `cut_type` input specifies what the cut region should be filled in with.
    Options for the cut type include: zeros, ones, low_noise, avg_noise, and
    high_noise. Zeros fills in the region with zeros; ones fills in the region
    with 1+1j samples; low_noise fills in the region with noise with -100dB
    power; avg_noise adds noise at power average of input data, effectively
    slicing/removing existing signals in the most RF realistic way of the
    options; and high_noise adds noise with 40dB power. If a list of multiple
    options are passed in, they are randomly sampled from.

    This transform is loosely based on
    `"Improved Regularization of Convolutional Neural Networks with Cutout" <https://arxiv.org/pdf/1708.04552v2.pdf>`_.

    Args:
         cut_dur (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            cut_dur sets the duration of the region to cut out
            * If Callable, produces a sample by calling cut_dur()
            * If int or float, cut_dur is fixed at the value provided
            * If list, cut_dur is any element in the list
            * If tuple, cut_dur is in range of (tuple[0], tuple[1])

        cut_type (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            cut_type sets the type of data to fill in the cut region with from
            the options: `zeros`, `ones`, `low_noise`, `avg_noise`, and
            `high_noise`
            * If Callable, produces a sample by calling cut_type()
            * If list, cut_type is any element in the list
            * If str, cut_type is fixed at the method provided

    """

    def __init__(
        self,
        cut_dur: NumericParameter = uniform_continuous_distribution(0.01, 0.2),
        cut_type: Union[List, str] = uniform_discrete_distribution(
            ["zeros", "ones", "low_noise", "avg_noise", "high_noise"]
        ),
    ):
        super(CutOut, self).__init__()
        self.cut_dur = to_distribution(cut_dur, self.random_generator)
        self.cut_type = to_distribution(cut_type, self.random_generator)

    def __call__(self, data: Any) -> Any:
        cut_dur = self.cut_dur()
        cut_start = np.random.uniform(0.0, 1.0 - cut_dur)
        cut_type = self.cut_type()

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
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

                # Update labels
                if (
                    new_signal_desc.start > cut_start
                    and new_signal_desc.start < cut_start + cut_dur
                ):
                    # Label starts within cut region
                    if (
                        new_signal_desc.stop > cut_start
                        and new_signal_desc.stop < cut_start + cut_dur
                    ):
                        # Label also stops within cut region --> Remove label
                        continue
                    else:
                        # Push label start to end of cut region
                        new_signal_desc.start = cut_start + cut_dur
                elif (
                    new_signal_desc.stop > cut_start
                    and new_signal_desc.stop < cut_start + cut_dur
                ):
                    # Label stops within cut region but does not start in region --> Push stop to begining of cut region
                    new_signal_desc.stop = cut_start
                elif (
                    new_signal_desc.start < cut_start
                    and new_signal_desc.stop > cut_start + cut_dur
                ):
                    # Label traverse cut region --> Split into two labels
                    new_signal_desc_split = deepcopy(signal_desc)
                    # Update first label region's stop
                    new_signal_desc.stop = cut_start
                    # Update second label region's start & append to description collection
                    new_signal_desc_split.start = cut_start + cut_dur
                    new_signal_description.append(new_signal_desc_split)

                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description

            # Perform data augmentation
            new_data.iq_data = functional.cut_out(
                data.iq_data, cut_start, cut_dur, cut_type
            )

        else:
            new_data = functional.cut_out(data, cut_start, cut_dur, cut_type)
        return new_data


class PatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `"PatchShuffle Regularization" <https://arxiv.org/pdf/1707.07103.pdf>`_.

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
        patch_size: NumericParameter = uniform_continuous_distribution(3, 10),
        shuffle_ratio: FloatParameter = uniform_continuous_distribution(0.01, 0.05),
    ):
        super(PatchShuffle, self).__init__()
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
            new_data.iq_data = functional.patch_shuffle(
                data.iq_data, patch_size, shuffle_ratio
            )

        else:
            new_data = functional.patch_shuffle(data, patch_size, shuffle_ratio)
        return new_data


class DatasetWidebandCutMix(SignalTransform):
    """SignalTransform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using an additional `alpha` input to set
    the relative quantity in time to occupy, where

       cutmix_num_iq_samples = total_num_iq_samples * alpha

    This transform is loosely based on [CutMix: Regularization Strategy to
    Train Strong Classifiers with Localizable Features]
    (https://arxiv.org/pdf/1710.09412.pdf).

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in durations between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling alpha()
            * If int or float, alpha is fixed at the value provided
            * If list, alpha is any element in the list
            * If tuple, alpha is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import WidebandSig53
        >>> # Add signals from the `ModulationsDataset`
        >>> dataset = WidebandSig53('.')
        >>> transform = ST.DatasetWidebandCutMix(dataset=dataset,alpha=(0.2,0.7))

    """

    def __init__(
        self,
        dataset: SignalDataset = None,
        alpha: NumericParameter = uniform_continuous_distribution(0.2, 0.7),
    ):
        super(DatasetWidebandCutMix, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            num_iq_samples = data.iq_data.shape[0]
            if insert_data.shape[0] != num_iq_samples:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_data.shape[0], data.shape[0]
                    )
                )

            # Mask both data examples based on alpha and a random start value
            insert_num_iq_samples = int(alpha * num_iq_samples)
            insert_start = np.random.randint(num_iq_samples - insert_num_iq_samples)
            insert_stop = insert_start + insert_num_iq_samples
            data.iq_data[insert_start:insert_stop] = 0
            insert_data[:insert_start] = 0
            insert_data[insert_stop:] = 0
            insert_start /= num_iq_samples
            insert_dur = insert_num_iq_samples / num_iq_samples

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data + insert_data

            # Update SignalDescription
            new_signal_description = []
            signal_description = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)

                # Update labels
                if (
                    new_signal_desc.start > insert_start
                    and new_signal_desc.start < insert_start + insert_dur
                ):
                    # Label starts within cut region
                    if (
                        new_signal_desc.stop > insert_start
                        and new_signal_desc.stop < insert_start + insert_dur
                    ):
                        # Label also stops within cut region --> Remove label
                        continue
                    else:
                        # Push label start to end of cut region
                        new_signal_desc.start = insert_start + insert_dur
                elif (
                    new_signal_desc.stop > insert_start
                    and new_signal_desc.stop < insert_start + insert_dur
                ):
                    # Label stops within cut region but does not start in region --> Push stop to begining of cut region
                    new_signal_desc.stop = insert_start
                elif (
                    new_signal_desc.start < insert_start
                    and new_signal_desc.stop > insert_start + insert_dur
                ):
                    # Label traverse cut region --> Split into two labels
                    new_signal_desc_split = deepcopy(signal_desc)
                    # Update first label region's stop
                    new_signal_desc.stop = insert_start
                    # Update second label region's start & append to description collection
                    new_signal_desc_split.start = insert_start + insert_dur
                    new_signal_description.append(new_signal_desc_split)

                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)

            # Repeat for inserted example's SignalDescription(s)
            for insert_signal_desc in insert_signal_description:
                # Update labels
                if (
                    insert_signal_desc.stop < insert_start
                    or insert_signal_desc.start > insert_start + insert_dur
                ):
                    # Label is outside inserted region --> Remove label
                    continue
                elif (
                    insert_signal_desc.start < insert_start
                    and insert_signal_desc.stop < insert_start + insert_dur
                ):
                    # Label starts before and ends within region, push start to region start
                    insert_signal_desc.start = insert_start
                elif (
                    insert_signal_desc.start >= insert_start
                    and insert_signal_desc.stop > insert_start + insert_dur
                ):
                    # Label starts within region and stops after, push stop to region stop
                    insert_signal_desc.stop = insert_start + insert_dur
                elif (
                    insert_signal_desc.start < insert_start
                    and insert_signal_desc.stop > insert_start + insert_dur
                ):
                    # Label starts before and stops after, push both start & stop to region boundaries
                    insert_signal_desc.start = insert_start
                    insert_signal_desc.stop = insert_start + insert_dur

                # Append SignalDescription to list
                new_signal_description.append(insert_signal_desc)

            # Set output data's SignalDescription to above list
            new_data.signal_description = new_signal_description

            return new_data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `DatasetWidebandCutMix` transform depends on metadata from a `SignalData` object.".format(
                    type(data)
                )
            )


class DatasetWidebandMixUp(SignalTransform):
    """SignalTransform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using the `alpha` input to set the
    difference in magnitudes between the two examples with the following
    relationship:

       output_sample = main_sample * (1 - alpha) + mixup_sample * alpha

    This transform is loosely based on [mixup: Beyond Emperical Risk
    Minimization](https://arxiv.org/pdf/1710.09412.pdf).

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling alpha()
            * If int or float, alpha is fixed at the value provided
            * If list, alpha is any element in the list
            * If tuple, alpha is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import WidebandSig53
        >>> # Add signals from the `WidebandSig53` Dataset
        >>> dataset = WidebandSig53('.')
        >>> transform = ST.DatasetWidebandMixUp(dataset=dataset,alpha=(0.4,0.6))

    """

    def __init__(
        self,
        dataset: SignalDataset = None,
        alpha: NumericParameter = uniform_continuous_distribution(0.4, 0.6),
    ):
        super(DatasetWidebandMixUp, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if insert_data.shape[0] != data.iq_data.shape[0]:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_data.shape[0], data.shape[0]
                    )
                )

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data * (1 - alpha) + insert_data * alpha

            # Update SignalDescription
            new_signal_description = []
            new_signal_description.extend(data.signal_description)
            new_signal_description.extend(insert_signal_description)
            new_data.signal_description = new_signal_description

            return new_data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `DatasetWidebandMixUp` transform depends on metadata from a `SignalData` object.".format(
                    type(data)
                )
            )


class SpectrogramRandomResizeCrop(SignalTransform):
    """The SpectrogramRandomResizeCrop transforms the input IQ data into a
    spectrogram with a randomized FFT size and overlap. This randomization in
    the spectrogram computation results in spectrograms of various sizes. The
    width and height arguments specify the target output size of the transform.
    To get to the desired size, the randomly generated spectrogram may be
    randomly cropped or padded in either the time or frequency dimensions. This
    transform is meant to emulate the Random Resize Crop transform often used
    in computer vision tasks.

    Args:
        nfft (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The number of FFT bins for the random spectrogram.
            * If Callable, nfft is set by calling nfft()
            * If int, nfft is fixed by value provided
            * If list, nfft is any element in the list
            * If tuple, nfft is in range of (tuple[0], tuple[1])
        overlap_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The ratio of the (nfft-1) value to use as the overlap parameter for
            the spectrogram operation. Setting as ratio ensures the overlap is
            a lower value than the bin size.
            * If Callable, nfft is set by calling overlap_ratio()
            * If float, overlap_ratio is fixed by value provided
            * If list, overlap_ratio is any element in the list
            * If tuple, overlap_ratio is in range of (tuple[0], tuple[1])
        window_fcn (:obj:`str`):
            Window to be used in spectrogram operation.
            Default value is 'np.blackman'.
        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default value is 'complex'.
        width (:obj:`int`):
            Target output width (time) of the spectrogram
        height (:obj:`int`):
            Target output height (frequency) of the spectrogram

    Example:
        >>> import torchsig.transforms as ST
        >>> # Randomly sample NFFT size in range [128,1024] and randomly crop/pad output spectrogram to (512,512)
        >>> transform = ST.SpectrogramRandomResizeCrop(nfft=(128,1024), overlap_ratio=(0.0,0.2), width=512, height=512)

    """

    def __init__(
        self,
        nfft: IntParameter = (256, 1024),
        overlap_ratio: FloatParameter = (0.0, 0.2),
        window_fcn: Callable[[int], np.ndarray] = np.blackman,
        mode: str = "complex",
        width: int = 512,
        height: int = 512,
    ):
        super(SpectrogramRandomResizeCrop, self).__init__()
        self.nfft = to_distribution(nfft, self.random_generator)
        self.overlap_ratio = to_distribution(overlap_ratio, self.random_generator)
        self.window_fcn = window_fcn
        self.mode = mode
        self.width = width
        self.height = height

    def __call__(self, data: Any) -> Any:
        nfft = int(self.nfft())
        nperseg = nfft
        overlap_ratio = self.overlap_ratio()
        noverlap = int(overlap_ratio * (nfft - 1))

        iq_data = data.iq_data if isinstance(data, SignalData) else data

        # First, perform the random spectrogram operation
        spec_data = eft_f.spectrogram(
            iq_data, nperseg, noverlap, nfft, self.window_fcn, self.mode
        )
        if self.mode == "complex":
            new_tensor = np.zeros(
                (2, spec_data.shape[0], spec_data.shape[1]), dtype=np.float32
            )
            new_tensor[0, :, :] = np.real(spec_data).astype(np.float32)
            new_tensor[1, :, :] = np.imag(spec_data).astype(np.float32)
            spec_data = new_tensor

        # Next, perform the random cropping/padding
        channels, curr_height, curr_width = spec_data.shape
        pad_height, crop_height = False, False
        pad_width, crop_width = False, False
        pad_height_samps, pad_width_samps = 0, 0
        if curr_height < self.height:
            pad_height = True
            pad_height_samps = self.height - curr_height
        elif curr_height > self.height:
            crop_height = True
        if curr_width < self.width:
            pad_width = True
            pad_width_samps = self.width - curr_width
        elif curr_width > self.width:
            crop_width = True

        if pad_height or pad_width:

            def pad_func(vector, pad_width, iaxis, kwargs):
                vector[: pad_width[0]] = (
                    np.random.rand(len(vector[: pad_width[0]])) * kwargs["pad_value"]
                )
                vector[-pad_width[1] :] = (
                    np.random.rand(len(vector[-pad_width[1] :])) * kwargs["pad_value"]
                )

            pad_height_start = np.random.randint(0, pad_height_samps // 2 + 1)
            pad_height_end = pad_height_samps - pad_height_start + 1
            pad_width_start = np.random.randint(0, pad_width_samps // 2 + 1)
            pad_width_end = pad_width_samps - pad_width_start + 1

            if self.mode == "complex":
                new_data_real = np.pad(
                    spec_data[0],
                    (
                        (pad_height_start, pad_height_end),
                        (pad_width_start, pad_width_end),
                    ),
                    pad_func,
                    pad_value=np.percentile(np.abs(spec_data[0]), 50),
                )
                new_data_imag = np.pad(
                    spec_data[1],
                    (
                        (pad_height_start, pad_height_end),
                        (pad_width_start, pad_width_end),
                    ),
                    pad_func,
                    pad_value=np.percentile(np.abs(spec_data[1]), 50),
                )
                spec_data = np.concatenate(
                    [
                        np.expand_dims(new_data_real, axis=0),
                        np.expand_dims(new_data_imag, axis=0),
                    ],
                    axis=0,
                )
            else:
                spec_data = np.pad(
                    spec_data,
                    (
                        (pad_height_start, pad_height_end),
                        (pad_width_start, pad_width_end),
                    ),
                    pad_func,
                    min_value=np.percentile(np.abs(spec_data[0]), 50),
                )

        crop_width_start = np.random.randint(0, max(1, curr_width - self.width))
        crop_height_start = np.random.randint(0, max(1, curr_height - self.height))
        spec_data = spec_data[
            :,
            crop_height_start : crop_height_start + self.height,
            crop_width_start : crop_width_start + self.width,
        ]

        # Update SignalData object if necessary, otherwise return
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = spec_data

            # Update SignalDescription
            new_signal_description = []
            signal_description = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)

                # Check bounds for partial signals
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

                # Update labels based on padding/cropping
                if pad_height:
                    new_signal_desc.lower_frequency = (
                        (new_signal_desc.lower_frequency + 0.5) * curr_height
                        + pad_height_start
                    ) / self.height - 0.5
                    new_signal_desc.upper_frequency = (
                        (new_signal_desc.upper_frequency + 0.5) * curr_height
                        + pad_height_start
                    ) / self.height - 0.5
                    new_signal_desc.center_frequency = (
                        (new_signal_desc.center_frequency + 0.5) * curr_height
                        + pad_height_start
                    ) / self.height - 0.5
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )

                if crop_height:
                    if (
                        new_signal_desc.lower_frequency + 0.5
                    ) * curr_height >= crop_height_start + self.height or (
                        new_signal_desc.upper_frequency + 0.5
                    ) * curr_height <= crop_height_start:
                        continue
                    if (
                        new_signal_desc.lower_frequency + 0.5
                    ) * curr_height <= crop_height_start:
                        new_signal_desc.lower_frequency = -0.5
                    else:
                        new_signal_desc.lower_frequency = (
                            (new_signal_desc.lower_frequency + 0.5) * curr_height
                            - crop_height_start
                        ) / self.height - 0.5
                    if (
                        new_signal_desc.upper_frequency + 0.5
                    ) * curr_height >= crop_height_start + self.height:
                        new_signal_desc.upper_frequency = (
                            crop_height_start + self.height
                        )
                    else:
                        new_signal_desc.upper_frequency = (
                            (new_signal_desc.upper_frequency + 0.5) * curr_height
                            - crop_height_start
                        ) / self.height - 0.5
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency + new_signal_desc.bandwidth / 2
                    )

                if pad_width:
                    new_signal_desc.start = (
                        new_signal_desc.start * curr_width + pad_width_start
                    ) / self.width
                    new_signal_desc.stop = (
                        new_signal_desc.stop * curr_width + pad_width_start
                    ) / self.width
                    new_signal_desc.duration = (
                        new_signal_desc.stop - new_signal_desc.start
                    )

                if crop_width:
                    if new_signal_desc.start * curr_width <= crop_width_start:
                        new_signal_desc.start = 0.0
                    elif (
                        new_signal_desc.start * curr_width
                        >= crop_width_start + self.width
                    ):
                        continue
                    else:
                        new_signal_desc.start = (
                            new_signal_desc.start * curr_width - crop_width_start
                        ) / self.width
                    if (
                        new_signal_desc.stop * curr_width
                        >= crop_width_start + self.width
                    ):
                        new_signal_desc.stop = 1.0
                    elif new_signal_desc.stop * curr_width <= crop_width_start:
                        continue
                    else:
                        new_signal_desc.stop = (
                            new_signal_desc.stop * curr_width - crop_width_start
                        ) / self.width
                    new_signal_desc.duration = (
                        new_signal_desc.stop - new_signal_desc.start
                    )

                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description

        else:
            new_data = spec_data

        return new_data
