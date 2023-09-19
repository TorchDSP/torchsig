from typing import Any, Callable, List, Literal, Optional, Tuple, Union
from torchsig.transforms import functional as F
from torchsig.transforms.functional import (
    FloatParameter,
    IntParameter,
    NumericParameter,
)
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.types import (
    SignalData,
    SignalDescription,
    RandomDistribution,
    UniformContinuousRD,
    UniformDiscreteRD,
)
from copy import deepcopy
from scipy import signal
import numpy as np


__all__ = [
    "Transform",
    "Compose",
    "Identity",
    "Lambda",
    "FixedRandom",
    "RandomApply",
    "SignalTransform",
    "Concatenate",
    "TargetConcatenate",
    "RandAugment",
    "RandChoice",
    "Normalize",
    "RandomResample",
    "TargetSNR",
    "AddNoise",
    "TimeVaryingNoise",
    "RayleighFadingChannel",
    "ImpulseInterferer",
    "RandomPhaseShift",
    "InterleaveComplex",
    "ComplexTo2D",
    "Real",
    "Imag",
    "ComplexMagnitude",
    "WrappedPhase",
    "DiscreteFourierTransform",
    "ChannelConcatIQDFT",
    "Spectrogram",
    "ContinuousWavelet",
    "ReshapeTransform",
    "RandomTimeShift",
    "TimeCrop",
    "TimeReversal",
    "AmplitudeReversal",
    "RandomFrequencyShift",
    "RandomDelayedFrequencyShift",
    "LocalOscillatorDrift",
    "GainDrift",
    "AutomaticGainControl",
    "IQImbalance",
    "RollOff",
    "AddSlope",
    "SpectralInversion",
    "ChannelSwap",
    "RandomMagRescale",
    "RandomDropSamples",
    "Quantize",
    "Clip",
    "RandomConvolve",
    "DatasetBasebandMixUp",
    "DatasetBasebandCutMix",
    "CutOut",
    "PatchShuffle",
    "DatasetWidebandCutMix",
    "DatasetWidebandMixUp",
    "SpectrogramRandomResizeCrop",
    "SpectrogramDropSamples",
    "SpectrogramPatchShuffle",
    "SpectrogramTranslation",
    "SpectrogramMosaicCrop",
    "SpectrogramMosaicDownsample",
    "SpectrogramImage",
]


class Transform:
    """An abstract class representing a Transform that can either work on
    targets or data

    """

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to compose.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Compose([ST.AddNoise(noise_power_db=10), ST.InterleaveComplex()])

    """

    def __init__(self, transforms: List[Callable], **kwargs) -> None:
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        return "\n".join([str(t) for t in self.transforms])


class Identity(Transform):
    """Just passes the data -- surprisingly useful in pipelines

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Identity()

    """

    def __init__(self, **kwargs) -> None:
        super(Identity, self).__init__(**kwargs)

    def __call__(self, data: Any) -> Any:
        return data


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Lambda(lambda x: x**2)  # A transform that squares all inputs.

    """

    def __init__(self, func: Callable, **kwargs) -> None:
        super(Lambda, self).__init__(**kwargs)
        self.func = func

    def __call__(self, data: Any) -> Any:
        return self.func(data)


class FixedRandom(Transform):
    """Restricts a randomized transform to apply only a fixed set of seeds.
    For example, this could be used to add noise randomly from among 1000
    possible sets of noise or add fading from 1000 possible channels.

    Args:
        transform (:obj:`Callable`):
            transform to be called

        num_seeds (:obj:`int`):
            number of possible random seeds to use

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.FixedRandom(ST.AddNoise(), num_seeds=10)

    """

    def __init__(self, transform: Transform, num_seeds: int, **kwargs) -> None:
        super(FixedRandom, self).__init__(**kwargs)
        self.transform = transform
        self.seeds = UniformDiscreteRD(np.asarray(range(num_seeds)))
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transform={}, ".format(str(transform))
            + "num_seeds={}".format(num_seeds)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        seed = self.seeds()
        orig_state = (
            np.random.get_state()
        )  # we do not want to somehow fix other random number generation processes.
        np.random.seed(seed)
        data = self.transform(data)
        np.random.set_state(orig_state)  # return numpy back to its previous state
        return data


class RandomApply(Transform):
    """Randomly applies a set of transforms with probability p

    Args:
        transform (``Transform`` objects):
            transform to randomly apply

        probability (:obj:`float`):
            In [0, 1.0], the probability with which to apply a transform

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.RandomApply(ST.AddNoise(noise_power_db=10), probability=.5)  # Add 10dB noise with probability .5

    """

    def __init__(
        self,
        transform: Callable,
        probability: float,
        **kwargs,
    ) -> None:
        super(RandomApply, self).__init__(**kwargs)
        self.transform = transform
        self.probability = probability
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transform={}, ".format(str(transform))
            + "probability={}".format(probability)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        return (
            self.transform(data)
            if RandomDistribution.rng.random() < self.probability
            else data
        )


class SignalTransform(Transform):
    """An abstract base class which explicitly only operates on Signal data

    Args:
        time_dim (:obj:`int`):
            Dimension along which to index time for a signal

    """

    def __init__(self, time_dim: int = 0, **kwargs) -> None:
        super(SignalTransform, self).__init__(**kwargs)
        self.time_dim = time_dim
        self.string: str = (
            self.__class__.__name__ + "(" + "time_dim={}".format(time_dim) + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(
        self,
        data: Union[SignalData, np.ndarray],
    ) -> Union[SignalData, np.ndarray]:
        raise NotImplementedError


class Concatenate(SignalTransform):
    """Inputs a list of SignalTransforms and applies each to the input data
    independently then concatenates the outputs along the specified dimension.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to apply and concatenate.

        concat_dim (:obj:`int`):
            Dimension along which to concatenate the outputs from each
            transform

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = Concatenate([ST.AddNoise(10), ST.DiscreteFourierTransform()], concat_dim=0)

    """

    def __init__(
        self,
        transforms: List[Transform],
        concat_dim: int = 0,
        **kwargs,
    ) -> None:
        super(Concatenate, self).__init__(**kwargs)
        self.transforms = transforms
        self.concat_dim = concat_dim
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + "concat_dim={}".format(concat_dim)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = np.concatenate(
                [transform(deepcopy(data.iq_data)) for transform in self.transforms],
                axis=self.concat_dim,
            )
        else:
            data = np.concatenate(
                [transform(deepcopy(data)) for transform in self.transforms],
                axis=self.concat_dim,
            )
        return data


class TargetConcatenate(SignalTransform):
    """Concatenates Target Transforms into a Tuple

    Args:
        transforms (list of ``Transform`` objects):
            List of transforms to concatenate

    """

    def __init__(self, transforms: List[Transform], **kwargs) -> None:
        super(TargetConcatenate, self).__init__(**kwargs)
        self.transforms = transforms
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, target: Any) -> Any:
        return tuple([transform(target) for transform in self.transforms])


class RandAugment(SignalTransform):
    """RandAugment transform loosely based on:
    `"RandAugment: Practical automated data augmentation with a reduced search space" <https://arxiv.org/pdf/1909.13719.pdf>`_.

    Args:
        transforms (list of `Transform` objects):
            List of transforms to choose from

        num_transforms (:obj: `int`):
            Number of transforms to randomly select

        allow_multiple_same (:obj: `bool`):
            Boolean specifying if multiple of the same transforms can be
            selected from the input list. Implemented as the `replace`
            parameter in numpy's random choice method.

    """

    def __init__(
        self,
        transforms: List[Callable],
        num_transforms: int = 2,
        allow_multiple_same: bool = False,
        **kwargs,
    ) -> None:
        super(RandAugment, self).__init__(**kwargs)
        self.transforms = transforms
        self.num_transforms = num_transforms
        self.allow_multiple_same = allow_multiple_same
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + "num_transforms={}, ".format(num_transforms)
            + "allow_multiple_same={}".format(allow_multiple_same)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        transforms = RandomDistribution.rng.choice(
            self.transforms,  # type: ignore
            size=self.num_transforms,
            replace=self.allow_multiple_same,
        )
        for t in transforms:
            data = t(data)
        return data


class RandChoice(SignalTransform):
    """RandChoice inputs a list of transforms and their associated
    probabilities. When called, a single transform will be sampled from the
    list using the probabilities provided, and then the selected transform
    will operate on the input data.

    Args:
        transforms (:obj:`list`):
            List of transforms to sample from and then apply

        probabilities (:obj:`list`):
            Probabilities used when sampling the above list of transforms

    """

    def __init__(
        self,
        transforms: List[SignalTransform],
        probabilities: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super(RandChoice, self).__init__(**kwargs)
        self.transforms = transforms
        self.probabilities: np.ndarray = (
            probabilities
            if probabilities
            else np.ones(len(self.transforms)) / len(self.transforms)
        )
        if np.sum(self.probabilities) != 1.0:
            self.probabilities /= np.sum(self.probabilities)
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + "probabilities=[{}]".format(self.probabilities)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        t: SignalTransform = RandomDistribution.rng.choice(
            self.transforms,  # type: ignore
            p=self.probabilities,
        )
        return t(data)


class Normalize(SignalTransform):
    """Normalize a IQ vector with mean and standard deviation.

    Args:
        norm :obj:`string`:
            Type of norm with which to normalize

        flatten :obj:`flatten`:
            Specifies if the norm should be calculated on the flattened
            representation of the input tensor

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """

    def __init__(
        self,
        norm: Optional[Union[int, float, Literal["fro", "nuc"]]] = 2,
        flatten: bool = False,
    ) -> None:
        super(Normalize, self).__init__()
        self.norm = norm
        self.flatten = flatten
        self.string: str = (
            self.__class__.__name__
            + "("
            + "norm={}, ".format(norm)
            + "flatten={}".format(flatten)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(
        self, data: Union[SignalData, np.ndarray]
    ) -> Union[SignalData, np.ndarray]:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.normalize(data.iq_data, self.norm, self.flatten)
        else:
            data = F.normalize(data, self.norm, self.flatten)
        return data


class RandomResample(SignalTransform):
    """Resample using poly-phase rational resampling technique.

    Args:
        rate_ratio (:py:class:`~torchsig.types.RandomDistribution`):
            new_rate = rate_ratio*old_rate

        num_iq_samples (:obj:`int`):
            Since resampling changes the number of points in a tensor, it is necessary to designate how
            many samples should be returned. In the case more samples are produced, the last num_iq_samples of
            the resampled tensor are returned.  In the case les samples are produced, the returned tensor is zero-padded
            to have num_iq_samples.

        keep_samples (:obj:`bool`):
            Despite returning a different number of samples being an issue, return however many samples
            are returned from resample_poly

    Note:
        When rate_ratio is > 1.0, the resampling algorithm produces more samples than the original tensor.
        When rate_ratio < 1.0, the resampling algorithm produces less samples than the original tensor. Hence,
        it is necessary to specify a number of samples to return from the newly resampled tensor so that there are
        always enough samples to return

    Example:
        >>> import torchsig.transforms as ST
        >>> # Randomly resample to a new_rate that is between 0.75 and 1.5 times the original rate
        >>> transform = ST.RandomResample((0.75, 1.5), num_iq_samples=4096)
        >>> # Randomly resample to a new_rate that is either 1.5 or 3.0
        >>> transform = ST.RandomResample([1.5, 3.0], num_iq_samples=4096)
        >>> # Resample to a new_rate that is always 1.5
        >>> transform = ST.RandomResample(1.5, num_iq_samples=4096)

    """

    def __init__(
        self,
        rate_ratio: FloatParameter = UniformContinuousRD(1.5, 3.0),
        num_iq_samples: int = 4096,
        keep_samples: bool = False,
    ) -> None:
        super(RandomResample, self).__init__()
        self.rate_ratio: Callable = RandomDistribution.to_distribution(rate_ratio)
        self.num_iq_samples = num_iq_samples
        self.keep_samples = keep_samples
        self.string: str = (
            self.__class__.__name__
            + "("
            + "rate_ratio={}, ".format(rate_ratio)
            + "num_iq_samples={}, ".format(num_iq_samples)
            + "keep_samples={}".format(keep_samples)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        new_rate: float = self.rate_ratio()
        if new_rate == 1.0:
            return data
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Update the SignalDescriptions with the new rate
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            anti_alias_lpf: bool = False
            for signal_desc_idx, signal_desc in enumerate(signal_description_list):
                new_signal_desc: SignalDescription = deepcopy(signal_desc)
                assert new_signal_desc.num_iq_samples is not None
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None
                assert new_signal_desc.samples_per_symbol is not None
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None
                assert new_signal_desc.center_frequency is not None
                assert new_signal_desc.bandwidth is not None

                # Update time descriptions
                new_num_iq_samples: float = new_signal_desc.num_iq_samples * new_rate
                start_iq_sample: float = new_signal_desc.start * new_num_iq_samples
                stop_iq_sample: float = new_signal_desc.stop * new_num_iq_samples
                if new_rate > 1.0:
                    # If the new rate is greater than 1.0, the resampled tensor
                    # is larger than the original tensor and is truncated to be
                    # the last <self.num_iq_samples> only
                    trunc_samples: float = new_num_iq_samples - self.num_iq_samples
                    new_start_iq_sample: float = start_iq_sample - trunc_samples
                    new_stop_iq_sample: float = stop_iq_sample - trunc_samples
                    new_signal_desc.start = (
                        new_start_iq_sample / self.num_iq_samples
                        if new_start_iq_sample > 0.0
                        else 0.0
                    )
                    new_signal_desc.stop = (
                        new_stop_iq_sample / self.num_iq_samples
                        if new_stop_iq_sample < self.num_iq_samples
                        else 1.0
                    )
                else:
                    # If the new rate is less than 1.0, the resampled tensor
                    # is smaller than the original tensor and is zero-padded
                    # at the end to length <self.num_iq_samples>
                    new_signal_desc.start *= new_rate
                    new_signal_desc.stop *= new_rate

                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start

                # Check for signals lost in truncation process
                if new_signal_desc.start > 1.0 or new_signal_desc.stop < 0.0:
                    continue

                # Update frequency descriptions
                new_signal_desc.samples_per_symbol *= new_rate
                # Check freq bounds for cases of partial signals
                # Upsampling these signals will distort them, but at least the label will follow
                if (
                    new_signal_desc.lower_frequency < -0.5
                    and new_signal_desc.upper_frequency / new_rate > -0.5
                    and new_rate > 1.0
                ):
                    new_signal_desc.lower_frequency = -0.5
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency + new_signal_desc.bandwidth / 2
                    )
                if (
                    new_signal_desc.upper_frequency > 0.5
                    and new_signal_desc.lower_frequency / new_rate < 0.5
                    and new_rate > 1.0
                ):
                    new_signal_desc.upper_frequency = 0.5
                    new_signal_desc.bandwidth = (
                        new_signal_desc.upper_frequency
                        - new_signal_desc.lower_frequency
                    )
                    new_signal_desc.center_frequency = (
                        new_signal_desc.lower_frequency + new_signal_desc.bandwidth / 2
                    )
                new_signal_desc.lower_frequency /= new_rate
                new_signal_desc.upper_frequency /= new_rate
                new_signal_desc.center_frequency /= new_rate
                new_signal_desc.bandwidth /= new_rate

                if (
                    new_signal_desc.lower_frequency < -0.45
                    or new_signal_desc.lower_frequency > 0.45
                    or new_signal_desc.upper_frequency < -0.45
                    or new_signal_desc.upper_frequency > 0.45
                ) and new_rate < 1.0:
                    # If downsampling and new signals are near band edge, apply a LPF to handle aliasing
                    anti_alias_lpf = True

                # Check new freqs for inclusion
                if (
                    new_signal_desc.lower_frequency > 0.5
                    or new_signal_desc.upper_frequency < -0.5
                ):
                    continue

                # Append updates to the new description
                new_signal_description.append(new_signal_desc)

            # Apply transform to data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = F.resample(
                data.iq_data,
                np.floor(new_rate * 100).astype(np.int32),
                100,
                self.num_iq_samples,
                self.keep_samples,
                anti_alias_lpf,
            )

            # Update the new data's SignalDescription
            new_data.signal_description = new_signal_description
            return new_data

        else:
            output: np.ndarray = F.resample(
                data,
                np.floor(new_rate * 100).astype(np.int32),
                100,
                self.num_iq_samples,
                self.keep_samples,
            )
            return output


class TargetSNR(SignalTransform):
    """Adds zero-mean complex additive white Gaussian noise to a provided
    tensor to achieve a target SNR. The provided signal is assumed to be
    entirely the signal of interest. Note that this transform relies on
    information contained within the SignalData object's SignalDescription. The
    transform also assumes that only one signal is present in the IQ data. If
    multiple signals' SignalDescriptions are detected, the transform will raise a
    warning.

    Args:
        target_snr (:py:class:`~torchsig.types.RandomDistribution`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

        eb_no (:obj:`bool`):
            Defines SNR as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2))*samples_per_symbol/bits_per_symbol.
            Defining SNR this way effectively normalized the noise level with respect to spectral efficiency and
            bandwidth. Normalizing this way is common in comparing systems in terms of power efficiency.
            If True, bits_per_symbol in the the SignalData will be used in the calculation of SNR. To achieve SNR in
            terms of E_b/N_0, samples_per_symbol must also be provided. Defaults to False.

        linear (:obj:`bool`):
            If True, target_snr and signal_power is on linear scale not dB. Defaults to False.

    """

    def __init__(
        self,
        target_snr: FloatParameter = UniformContinuousRD(-10, 10),
        eb_no: bool = False,
        linear: bool = False,
        **kwargs,
    ) -> None:
        super(TargetSNR, self).__init__(**kwargs)
        self.target_snr = RandomDistribution.to_distribution(target_snr)
        self.eb_no = eb_no
        self.linear = linear
        self.string = (
            self.__class__.__name__
            + "("
            + "target_snr={}, ".format(target_snr)
            + "eb_no={}, ".format(eb_no)
            + "linear={}".format(linear)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        target_snr = self.target_snr()
        target_snr_linear = 10 ** (target_snr / 10) if not self.linear else target_snr
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            if len(signal_description_list) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected.".format(
                        len(signal_description_list)
                    )
                )
            assert signal_description_list[0].class_name is not None
            assert signal_description_list[0].samples_per_symbol is not None
            assert signal_description_list[0].bits_per_symbol is not None
            assert signal_description_list[0].snr is not None

            signal_power = np.mean(np.abs(data.iq_data) ** 2, axis=self.time_dim)
            class_name = signal_description_list[0].class_name
            if "ofdm" not in class_name:
                # EbNo not available for OFDM
                target_snr_linear *= (
                    signal_description_list[0].bits_per_symbol if self.eb_no else 1
                )
            occupied_bw = 1 / signal_description_list[0].samples_per_symbol
            noise_power_linear = signal_power / (target_snr_linear * occupied_bw)
            noise_power_db = 10 * np.log10(noise_power_linear)
            data.iq_data = F.awgn(data.iq_data, noise_power_db)
            signal_description_list[0].snr = target_snr
            return data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `TargetSNR` transform depends on metadata from a `SignalData` object. \n\t\
                Please reference the `AddNoise` transform as an alternative.".format(
                    type(data)
                )
            )


class AddNoise(SignalTransform):
    """Add random AWGN at specified power levels

    Note:
        Differs from the TargetSNR() in that this transform adds
        noise at a specified power level, whereas TargetSNR()
        assumes a basebanded signal and adds noise to achieve a specified SNR
        level for the signal of interest. This transform,
        AddNoise() is useful for simply adding a randomized
        level of noise to either a narrowband or wideband input.

    Args:
        noise_power_db (:py:class:`~torchsig.types.RandomDistribution`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

        input_noise_floor_db (:obj:`float`):
            The noise floor of the input data in dB

        linear (:obj:`bool`):
            If True, target_snr and signal_power is on linear scale not dB.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Added AWGN power range is (-40, -20) dB
        >>> transform = ST.AddNoise((-40, -20))

    """

    def __init__(
        self,
        noise_power_db: FloatParameter = UniformContinuousRD(-80, -60),
        input_noise_floor_db: float = 0.0,
        linear: bool = False,
        **kwargs,
    ) -> None:
        super(AddNoise, self).__init__(**kwargs)
        self.noise_power_db = RandomDistribution.to_distribution(noise_power_db)
        self.input_noise_floor_db = input_noise_floor_db
        self.linear = linear
        self.string = (
            self.__class__.__name__
            + "("
            + "noise_power_db={}, ".format(noise_power_db)
            + "input_noise_floor_db={}, ".format(input_noise_floor_db)
            + "linear={}".format(linear)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Retrieve random noise power value
            noise_power_db = self.noise_power_db()
            noise_power_db = (
                10 * np.log10(noise_power_db) if self.linear else noise_power_db
            )

            if self.input_noise_floor_db:
                noise_floor = self.input_noise_floor_db
            else:
                # TODO: implement fast noise floor estimation technique?
                noise_floor = 0  # Assumes 0dB noise floor

            # Apply data augmentation
            new_data.iq_data = F.awgn(data.iq_data, noise_power_db)

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.snr is not None
                new_signal_desc.snr = (
                    (new_signal_desc.snr - noise_power_db)
                    if noise_power_db > noise_floor
                    else new_signal_desc.snr
                )
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            return new_data

        else:
            noise_power_db = self.noise_power_db(size=data.shape[0])
            noise_power_db = (
                10 * np.log10(noise_power_db) if self.linear else noise_power_db
            )
            output: np.ndarray = F.awgn(data, noise_power_db)
            return output


class TimeVaryingNoise(SignalTransform):
    """Add time-varying random AWGN at specified input parameters

    Args:
        noise_power_db_low (:py:class:`~torchsig.types.RandomDistribution`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

        noise_power_db_high (:py:class:`~torchsig.types.RandomDistribution`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

        inflections (:py:class:`~torchsig.types.RandomDistribution`):
            Number of inflection points in time-varying noise

        random_regions (:py:class:`~torchsig.types.RandomDistribution`):
            If inflections > 0, random_regions specifies whether each
            inflection point should be randomly selected or evenly divided
            among input data

        linear (:obj:`bool`):
            If True, powers input are on linear scale not dB.

    """

    def __init__(
        self,
        noise_power_db_low: FloatParameter = UniformContinuousRD(-80, -60),
        noise_power_db_high: FloatParameter = UniformContinuousRD(-40, -20),
        inflections: IntParameter = UniformDiscreteRD(np.arange(0, 10, dtype=int)),
        random_regions: Union[List, bool] = True,
        linear: bool = False,
        **kwargs,
    ) -> None:
        super(TimeVaryingNoise, self).__init__(**kwargs)
        self.noise_power_db_low = RandomDistribution.to_distribution(noise_power_db_low)
        self.noise_power_db_high = RandomDistribution.to_distribution(
            noise_power_db_high
        )
        self.inflections = RandomDistribution.to_distribution(inflections)
        self.random_regions = RandomDistribution.to_distribution(random_regions)
        self.linear = linear
        self.string = (
            self.__class__.__name__
            + "("
            + "noise_power_db_low={}, ".format(noise_power_db_low)
            + "noise_power_db_high={}, ".format(noise_power_db_high)
            + "inflections={}, ".format(inflections)
            + "random_regions={}, ".format(random_regions)
            + "linear={}".format(linear)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        noise_power_db_low = self.noise_power_db_low()
        noise_power_db_high = self.noise_power_db_high()
        noise_power_db_low = (
            10 * np.log10(noise_power_db_low) if self.linear else noise_power_db_low
        )
        noise_power_db_high = (
            10 * np.log10(noise_power_db_high) if self.linear else noise_power_db_high
        )
        inflections = int(self.inflections())
        random_regions = self.random_regions()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Apply data augmentation
            new_data.iq_data = F.time_varying_awgn(
                data.iq_data,
                noise_power_db_low,
                noise_power_db_high,
                inflections,
                random_regions,
            )

            # Update SignalDescription with average of added noise (Note: this is merely an approximation)
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            noise_power_db_change = np.abs(noise_power_db_high - noise_power_db_low)
            avg_noise_power_db = (
                min(noise_power_db_low, noise_power_db_high) + noise_power_db_change / 2
            )
            for signal_desc in new_signal_description:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.snr is not None
                new_signal_desc.snr -= avg_noise_power_db
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            return new_data

        else:
            output: np.ndarray = F.time_varying_awgn(
                data,
                noise_power_db_low,
                noise_power_db_high,
                inflections,
                random_regions,
            )
            return output


class RayleighFadingChannel(SignalTransform):
    """Applies Rayleigh fading channel to tensor.

    Note:
        A Rayleigh fading channel can be modeled as an FIR filter with Gaussian distributed taps which vary over time.
        The length of the filter determines the coherence bandwidth of the channel and is inversely proportional to
        the delay spread. The rate at which the channel taps vary over time is related to the coherence time and this is
        inversely proportional to the maximum Doppler spread. This time variance is not included in this model.

    Args:
        coherence_bandwidth (:py:class:`~torchsig.types.RandomDistribution`):

        power_delay_profile (:obj:`list`, :obj:`tuple`):
            A list of positive values assigning power to taps of the channel model. When the number of taps
            exceeds the number of items in the provided power_delay_profile, the list is linearly interpolated
            to provide values for each tap of the channel

    Example:
        >>> import torchsig.transforms as ST
        >>> # Rayleigh Fading with coherence bandwidth uniformly distributed between fs/100 and fs/10
        >>> transform = ST.RayleighFadingChannel(lambda size: np.random.uniform(.01, .1, size))
        >>> # Rayleigh Fading with coherence bandwidth normally distributed clipped between .01 and .1
        >>> transform = ST.RayleighFadingChannel(lambda size: np.clip(np.random.normal(0, .1, size), .01, .1))
        >>> # Rayleigh Fading with coherence bandwidth uniformly distributed between fs/100 and fs/10
        >>> transform = ST.RayleighFadingChannel((.01, .1))
        >>> # Rayleigh Fading with coherence bandwidth either .02 or .01
        >>> transform = ST.RayleighFadingChannel([.02, .01])
        >>> # Rayleigh Fading with fixed coherence bandwidth at .1
        >>> transform = ST.RayleighFadingChannel(.1)
        >>> # Rayleigh Fading with fixed coherence bandwidth at .1 and pdp (1.0, .7, .1)
        >>> transform = ST.RayleighFadingChannel((.01, .1), power_delay_profile=(1.0, .7, .1))
    """

    def __init__(
        self,
        coherence_bandwidth: FloatParameter = UniformContinuousRD(0.01, 0.1),
        power_delay_profile: Union[Tuple, List, np.ndarray] = (1, 1),
        **kwargs,
    ) -> None:
        super(RayleighFadingChannel, self).__init__(**kwargs)
        self.coherence_bandwidth = RandomDistribution.to_distribution(
            coherence_bandwidth
        )
        self.power_delay_profile = np.asarray(power_delay_profile)
        self.string = (
            self.__class__.__name__
            + "("
            + "coherence_bandwidth={}, ".format(coherence_bandwidth)
            + "power_delay_profile={}".format(power_delay_profile)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        coherence_bandwidth = self.coherence_bandwidth()
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.rayleigh_fading(
                data.iq_data, coherence_bandwidth, self.power_delay_profile
            )
        else:
            data = F.rayleigh_fading(
                data, coherence_bandwidth, self.power_delay_profile
            )
        return data


class ImpulseInterferer(SignalTransform):
    """Applies an impulse interferer

    Args:
        amp (:py:class:`~torchsig.types.RandomDistribution`):

        pulse_offset (:py:class:`~torchsig.types.RandomDistribution`):

    """

    def __init__(
        self,
        amp: FloatParameter = UniformContinuousRD(0.1, 100.0),
        pulse_offset: FloatParameter = UniformContinuousRD(0.0, 1),
        **kwargs,
    ) -> None:
        super(ImpulseInterferer, self).__init__(**kwargs)
        self.amp = RandomDistribution.to_distribution(amp)
        self.pulse_offset = RandomDistribution.to_distribution(pulse_offset)
        self.string = (
            self.__class__.__name__
            + "("
            + "amp={}, ".format(amp)
            + "pulse_offset={}".format(pulse_offset)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        amp = self.amp()
        pulse_offset = self.pulse_offset()
        pulse_offset = 1.0 if pulse_offset > 1.0 else np.max((0.0, pulse_offset))
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.impulsive_interference(
                data.iq_data, amp, self.pulse_offset
            )
        else:
            data = F.impulsive_interference(data, amp, self.pulse_offset)
        return data


class RandomPhaseShift(SignalTransform):
    """Applies a random phase offset to tensor

    Args:
        phase_offset (:py:class:`~torchsig.types.RandomDistribution`):

    Example:
        >>> import torchsig.transforms as ST
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = ST.RandomPhaseShift(UniformContinuousRD(-1, 1))
        >>> # Phase Offset from [-pi/2, 0, and pi/2]
        >>> transform = ST.RandomPhaseShift(uniform_discrete_distribution([-.5, 0, .5]))
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = ST.RandomPhaseShift((-1, 1))
        >>> # Phase Offset either -pi/4 or pi/4
        >>> transform = ST.RandomPhaseShift([-.25, .25])
        >>> # Phase Offset is fixed at -pi/2
        >>> transform = ST.RandomPhaseShift(-.5)
    """

    def __init__(
        self,
        phase_offset: FloatParameter = UniformContinuousRD(-1, 1),
        **kwargs,
    ) -> None:
        super(RandomPhaseShift, self).__init__(**kwargs)
        self.phase_offset = RandomDistribution.to_distribution(phase_offset)
        self.string = (
            self.__class__.__name__ + "(" + "phase_offset={}".format(phase_offset) + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        phases = self.phase_offset()
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.phase_offset(data.iq_data, phases * np.pi)
        else:
            data = F.phase_offset(data, phases * np.pi)
        return data


class InterleaveComplex(SignalTransform):
    """Converts complex IQ samples to interleaved real and imaginary floating
    point values.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.InterleaveComplex()

    """

    def __init__(self) -> None:
        super(InterleaveComplex, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.interleave_complex(data.iq_data)
        else:
            data = F.interleave_complex(data)
        return data


class ComplexTo2D(SignalTransform):
    """Takes a vector of complex IQ samples and converts two channels of real
    and imaginary parts

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ComplexTo2D()

    """

    def __init__(self) -> None:
        super(ComplexTo2D, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.complex_to_2d(data.iq_data)
        else:
            data = F.complex_to_2d(data)
        return data


class Real(SignalTransform):
    """Takes a vector of complex IQ samples and returns Real portions

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Real()

    """

    def __init__(self) -> None:
        super(Real, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.real(data.iq_data)
        else:
            data = F.real(data)
        return data


class Imag(SignalTransform):
    """Takes a vector of complex IQ samples and returns Imaginary portions

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Imag()

    """

    def __init__(self) -> None:
        super(Imag, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.imag(data.iq_data)
        else:
            data = F.imag(data)
        return data


class ComplexMagnitude(SignalTransform):
    """Takes a vector of complex IQ samples and returns the complex magnitude

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ComplexMagnitude()

    """

    def __init__(self) -> None:
        super(ComplexMagnitude, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.complex_magnitude(data.iq_data)
        else:
            data = F.complex_magnitude(data)
        return data


class WrappedPhase(SignalTransform):
    """Takes a vector of complex IQ samples and returns wrapped phase (-pi, pi)

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.WrappedPhase()

    """

    def __init__(self) -> None:
        super(WrappedPhase, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.wrapped_phase(data.iq_data)
        else:
            data = F.wrapped_phase(data)
        return data


class DiscreteFourierTransform(SignalTransform):
    """Calculates DFT using FFT

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.DiscreteFourierTransform()

    """

    def __init__(self) -> None:
        super(DiscreteFourierTransform, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.discrete_fourier_transform(data.iq_data)
        else:
            data = F.discrete_fourier_transform(data)
        return data


class ChannelConcatIQDFT(SignalTransform):
    """Converts the input IQ into 2D tensor of the real & imaginary components
    concatenated in the channel dimension. Next, calculate the DFT using the
    FFT, convert the complex DFT into a 2D tensor of real & imaginary frequency
    components. Finally, stack the 2D IQ and the 2D DFT components in the
    channel dimension.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ChannelConcatIQDFT()

    """

    def __init__(self) -> None:
        super(ChannelConcatIQDFT, self).__init__()

    def __call__(self, data: Any) -> Any:
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        assert iq_data is not None
        dft_data = F.discrete_fourier_transform(iq_data)
        iq_data = F.complex_to_2d(iq_data)
        dft_data = F.complex_to_2d(dft_data)
        output_data = np.concatenate([iq_data, dft_data], axis=0)
        if isinstance(data, SignalData):
            data.iq_data = output_data
        else:
            data = output_data
        return data


class Spectrogram(SignalTransform):
    """Calculates power spectral density over time

    Args:
        nperseg (:obj:`int`):
            Length of each segment. If window is str or tuple, is set to 256,
            and if window is array_like, is set to the length of the window.

        noverlap (:obj:`int`):
            Number of points to overlap between segments.
            If None, noverlap = nperseg // 8.

        nfft (:obj:`int`):
            Length of the FFT used, if a zero padded FFT is desired.
            If None, the FFT length is nperseg.

        detrend : str or function or False, optional
            Specifies how to detrend each segment. If detrend is a string, it is passed as the type
            argument to the detrend function. If it is a function, it takes a segment and returns a
            detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.

        scaling : { ‘density’, ‘spectrum’ }, optional
            Selects between computing the power spectral density (‘density’) where Sxx has units of
            V**2/Hz and computing the power spectrum (‘spectrum’) where Sxx has units of V**2, if
            x is measured in V and fs is measured in Hz. Defaults to ‘density’.

        window_fcn (:obj:`str`):
            Window to be used in spectrogram operation.
            Default value is 'np.blackman'.

        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default value is 'psd'.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Spectrogram with seg_size=256, overlap=64, nfft=256, window=blackman_harris
        >>> transform = ST.Spectrogram()
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=blackman_harris (2x oversampled in time)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64)
        >>> # Spectrogram with seg_size=128, overlap=0, nfft=128, window=blackman_harris (critically sampled)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=0)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=blackman_harris (2x oversampled in frequency)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=rectangular
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256, window_fcn=np.ones)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=rectangular, detrend=constant, scaling=density
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256, window_fcn=np.ones, detrend='constant', scaling='density')

    """

    def __init__(
        self,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        detrend: Optional[str] = "constant",
        scaling: Optional[str] = "density",
        window_fcn: Callable[[int], np.ndarray] = np.blackman,
        mode: str = "psd",
    ) -> None:
        super(Spectrogram, self).__init__()
        self.nperseg: int = nperseg
        self.noverlap: int = nperseg // 4 if noverlap is None else noverlap
        self.nfft: int = nperseg if nfft is None else nfft
        self.detrend: Optional[str] = None if detrend is None else detrend
        self.scaling: Optional[str] = None if scaling is None else scaling
        self.window_fcn = window_fcn
        self.mode = mode
        self.string = (
            self.__class__.__name__
            + "("
            + "nperseg={}, ".format(nperseg)
            + "noverlap={}, ".format(self.noverlap)
            + "nfft={}, ".format(self.nfft)
            + "detrend={}".format(self.detrend)
            + "scaling={}".format(self.scaling)
            + "window_fcn={}, ".format(window_fcn)
            + "mode={}".format(mode)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.spectrogram(
                data.iq_data,
                self.nperseg,
                self.noverlap,
                self.nfft,
                self.detrend,
                self.scaling,
                self.window_fcn,
                self.mode,
            )
            if self.mode == "complex":
                new_tensor = np.zeros(
                    (2, data.iq_data.shape[0], data.iq_data.shape[1]), dtype=np.float32
                )
                new_tensor[0, :, :] = np.real(data.iq_data).astype(np.float32)
                new_tensor[1, :, :] = np.imag(data.iq_data).astype(np.float32)
                data.iq_data = new_tensor
        else:
            data = F.spectrogram(
                data,
                self.nperseg,
                self.noverlap,
                self.nfft,
                self.detrend,
                self.scaling,
                self.window_fcn,
                self.mode,
            )
            if self.mode == "complex":
                new_tensor = np.zeros(
                    (2, data.shape[0], data.shape[1]), dtype=np.float32
                )
                new_tensor[0, :, :] = np.real(data).astype(np.float32)
                new_tensor[1, :, :] = np.imag(data).astype(np.float32)
                data = new_tensor
        return data


class ContinuousWavelet(SignalTransform):
    """Computes the continuous wavelet transform resulting in a Scalogram of
    the complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        wavelet (:obj:`str`):
            Name of the mother wavelet.
            If None, wavename = 'mexh'.

        nscales (:obj:`int`):
            Number of scales to use in the Scalogram.
            If None, nscales = 33.

        sample_rate (:obj:`float`):
            Sample rate of the signal.
            If None, fs = 1.0.

    Example:
        >>> import torchsig.transforms as ST
        >>> # ContinuousWavelet SignalTransform using the 'mexh' mother wavelet with 33 scales
        >>> transform = ST.ContinuousWavelet()

    """

    def __init__(
        self, wavelet: str = "mexh", nscales: int = 33, sample_rate: float = 1.0
    ) -> None:
        super(ContinuousWavelet, self).__init__()
        self.wavelet = wavelet
        self.nscales = nscales
        self.sample_rate = sample_rate
        self.string = (
            self.__class__.__name__
            + "("
            + "wavelet={}, ".format(wavelet)
            + "nscales={}, ".format(nscales)
            + "sample_rate={}".format(sample_rate)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.continuous_wavelet_transform(
                data.iq_data,
                self.wavelet,
                self.nscales,
                self.sample_rate,
            )
        else:
            data = F.continuous_wavelet_transform(
                data,
                self.wavelet,
                self.nscales,
                self.sample_rate,
            )
        return data


class ReshapeTransform(SignalTransform):
    """Reshapes the input data to the specified shape

    Args:
        new_shape (obj:`tuple`):
            The new shape for the input data

    """

    def __init__(self, new_shape: Tuple, **kwargs) -> None:
        super(ReshapeTransform, self).__init__(**kwargs)
        self.new_shape = new_shape
        self.string = (
            self.__class__.__name__ + "(" + "new_shape={}".format(new_shape) + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = data.iq_data.reshape(*self.new_shape)
        else:
            data = data.reshape(*self.new_shape)
        return data


class RandomTimeShift(SignalTransform):
    """Shifts tensor in the time dimension by shift samples. Zero-padding is applied to maintain input size.

    Args:
        shift (:py:class:`~torchsig.types.RandomDistribution`):

        interp_rate (:obj:`int`):
            Interpolation rate used by internal interpolation filter

        taps_per_arm (:obj:`int`):
            Number of taps per arm used in filter. More is slower, but more accurate.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Shift inputs by range of (-10, 20) samples with uniform distribution
        >>> transform = ST.RandomTimeShift(lambda size: np.random.uniform(-10, 20, size))
        >>> # Shift inputs by normally distributed time shifts
        >>> transform = ST.RandomTimeShift(lambda size: np.random.normal(0, 10, size))
        >>> # Shift by discrete set of values
        >>> transform = ST.RandomTimeShift(lambda size: np.random.choice([-10, 5, 10], size))
        >>> # Shift by 5 or 10
        >>> transform = ST.RandomTimeShift([5, 10])
        >>> # Shift by random amount between 5 and 10 with uniform probability
        >>> transform = ST.RandomTimeShift((5, 10))
        >>> # Shift fixed at 5 samples
        >>> transform = ST.RandomTimeShift(5)

    """

    def __init__(
        self,
        shift: FloatParameter = UniformContinuousRD(-10, 10),
        interp_rate: int = 100,
        taps_per_arm: int = 24,
    ) -> None:
        super(RandomTimeShift, self).__init__()
        self.shift = RandomDistribution.to_distribution(shift)
        self.interp_rate = interp_rate
        num_taps = int(taps_per_arm * interp_rate)
        self.taps = (
            signal.firwin(
                num_taps, 1.0 / interp_rate, 1.0 / interp_rate / 4.0, scale=True
            )
            * interp_rate
        )
        self.string = (
            self.__class__.__name__
            + "("
            + "shift={}, ".format(shift)
            + "interp_rate={}, ".format(interp_rate)
            + "taps_per_arm={}".format(taps_per_arm)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        shift: float = float(self.shift())
        integer_part, decimal_part = divmod(shift, 1)
        integer_time_shift: int = int(integer_part) if integer_part else 0
        float_decimal_part: float = float(decimal_part) if decimal_part else 0.0

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Apply data transformation
            if float_decimal_part != 0:
                new_data.iq_data = F.fractional_shift(
                    data.iq_data,
                    self.taps,
                    self.interp_rate,
                    -float_decimal_part,  # this needed to be negated to be consistent with the previous implementation
                )
            else:
                new_data.iq_data = data.iq_data
            new_data.iq_data = F.time_shift(new_data.iq_data, integer_time_shift)

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None

                new_signal_desc.start += shift / new_data.iq_data.shape[0]
                new_signal_desc.stop += shift / new_data.iq_data.shape[0]
                new_signal_desc.start = (
                    0.0 if new_signal_desc.start < 0.0 else new_signal_desc.start
                )
                new_signal_desc.stop = (
                    1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                )
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                if new_signal_desc.start > 1.0 or new_signal_desc.stop < 0.0:
                    continue
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            return new_data

        else:
            output: np.ndarray = data.copy()
            if float_decimal_part != 0:
                output = F.fractional_shift(
                    output,
                    self.taps,
                    self.interp_rate,
                    -float_decimal_part,  # this needed to be negated to be consistent with the previous implementation
                )
            output = F.time_shift(output, integer_time_shift)
            return output


class TimeCrop(SignalTransform):
    """Crops a tensor in the time dimension to the specified length. Optional
    crop techniques include: start, center, end, & random

    Args:
        crop_type (:obj:`str`):
            Type of cropping to perform. Options are: `start`, `center`, `end`,
            and `random`. `start` crops the input tensor such that the first
            `length` samples are returned. `center` crops the input tensor such
            that the center `length` samples are returned. `end` crops the
            input tensor such that the last `length` samples are returned.
            `random` crops randomly in the range `[0,length-1]`.

        length (:obj:`int`):
            Number of samples to include.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Crop inputs to first 256 samples
        >>> transform = ST.TimeCrop(crop_type='start', length=256)
        >>> # Crop inputs to center 512 samples
        >>> transform = ST.TimeCrop(crop_type='center', length=512)
        >>> # Crop inputs to last 1024 samples
        >>> transform = ST.TimeCrop(crop_type='end', length=1024)
        >>> # Randomly crop any 2048 samples from input
        >>> transform = ST.TimeCrop(crop_type='random', length=2048)

    """

    def __init__(self, crop_type: str = "random", length: int = 256) -> None:
        super(TimeCrop, self).__init__()
        self.crop_type = crop_type
        self.length = length
        self.string = (
            self.__class__.__name__
            + "("
            + "crop_type={}, ".format(crop_type)
            + "length={}".format(length)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        assert iq_data is not None

        if iq_data.shape[0] == self.length:
            return data
        elif iq_data.shape[0] < self.length:
            raise ValueError(
                "Input data length {} is less than requested length {}".format(
                    iq_data.shape[0], self.length
                )
            )

        if self.crop_type == "start":
            start = 0
        elif self.crop_type == "end":
            start = iq_data.shape[0] - self.length
        elif self.crop_type == "center":
            start = (iq_data.shape[0] - self.length) // 2
        elif self.crop_type == "random":
            start = np.random.randint(0, iq_data.shape[0] - self.length)
        else:
            raise ValueError("Crop type must be: `start`, `center`, `end`, or `random`")

        if isinstance(data, SignalData):
            assert data.signal_description is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Perform data augmentation
            new_data.iq_data = F.time_crop(iq_data, start, self.length)
            assert new_data.iq_data is not None

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None

                original_start_sample = signal_desc.start * iq_data.shape[0]
                original_stop_sample = signal_desc.stop * iq_data.shape[0]
                new_start_sample = original_start_sample - start
                new_stop_sample = original_stop_sample - start
                new_signal_desc.start = float(new_start_sample / self.length)
                new_signal_desc.stop = float(new_stop_sample / self.length)
                new_signal_desc.start = (
                    0.0 if new_signal_desc.start < 0.0 else new_signal_desc.start
                )
                new_signal_desc.stop = (
                    1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                )
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_desc.num_iq_samples = self.length
                if new_signal_desc.start > 1.0 or new_signal_desc.stop < 0.0:
                    continue
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            return new_data

        else:
            output: np.ndarray = F.time_crop(data, start, self.length)
            return output


class TimeReversal(SignalTransform):
    """Applies a time reversal to the input. Note that applying a time reversal
    inherently also applies a spectral inversion. If a time-reversal without
    spectral inversion is desired, the `undo_spectral_inversion` argument
    can be set to True. By setting this value to True, an additional, manual
    spectral inversion is applied to revert the time-reversal's inversion
    effect.

    Args:
        undo_spectral_inversion (:obj:`bool`, :obj:`float`):
            * If bool, undo_spectral_inversion is always/never applied
            * If float, undo_spectral_inversion is a probability

    """

    def __init__(
        self,
        undo_spectral_inversion: Union[bool, float] = True,
    ) -> None:
        super(TimeReversal, self).__init__()
        if isinstance(undo_spectral_inversion, bool):
            self.undo_spectral_inversion: float = (
                1.0 if undo_spectral_inversion else 0.0
            )
        else:
            self.undo_spectral_inversion = undo_spectral_inversion
        self.string = (
            self.__class__.__name__
            + "("
            + "undo_spectral_inversion={}".format(undo_spectral_inversion)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        spec_inversion_prob = np.random.rand()
        undo_spec_inversion = spec_inversion_prob <= self.undo_spectral_inversion
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Perform data augmentation
            new_data.iq_data = F.time_reversal(data.iq_data)
            if undo_spec_inversion:
                # If spectral inversion not desired, reverse effect
                new_data.iq_data = F.spectral_inversion(new_data.iq_data)

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None
                assert new_signal_desc.center_frequency is not None

                # Invert time labels
                original_start = new_signal_desc.start
                original_stop = new_signal_desc.stop
                new_signal_desc.start = original_stop * -1 + 1.0
                new_signal_desc.stop = original_start * -1 + 1.0

                if not undo_spec_inversion:
                    # Invert freq labels
                    original_lower = new_signal_desc.lower_frequency
                    original_upper = new_signal_desc.upper_frequency
                    new_signal_desc.lower_frequency = original_upper * -1
                    new_signal_desc.upper_frequency = original_lower * -1
                    new_signal_desc.center_frequency *= -1

                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description
            return new_data

        else:
            output: np.ndarray = F.time_reversal(data)
            if undo_spec_inversion:
                # If spectral inversion not desired, reverse effect
                output = F.spectral_inversion(output)
            return output


class AmplitudeReversal(SignalTransform):
    """Applies an amplitude reversal to the input tensor by applying a value of
    -1 to each sample. Effectively the same as a static phase shift of pi

    """

    def __init__(self) -> None:
        super(AmplitudeReversal, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            new_data.iq_data = F.amplitude_reversal(data.iq_data)
            return new_data
        else:
            output: np.ndarray = F.amplitude_reversal(data)
            return output


class RandomFrequencyShift(SignalTransform):
    """Shifts each tensor in freq by freq_shift along the time dimension.

    Args:
        freq_shift (:py:class:`~torchsig.types.RandomDistribution`):

    Example:
        >>> import torchsig.transforms as ST
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = ST.RandomFrequencyShift(freq_shift=(-0.25, 0.25))
        >>> # Frequency shift inputs always fs/10
        >>> transform = ST.RandomFrequencyShift(freq_shift=0.1)
        >>> # Frequency shift inputs with normal distribution with stdev .1
        >>> transform = ST.RandomFrequencyShift(freq_shift=lambda size: np.random.normal(0, .1, size))
        >>> # Frequency shift inputs with either -fs/4 or fs/4 (discrete)
        >>> transform = ST.RandomFrequencyShift(freq_shift=[-.25, .25])

    """

    def __init__(
        self, freq_shift: FloatParameter = UniformContinuousRD(-0.5, 0.5)
    ) -> None:
        super(RandomFrequencyShift, self).__init__()
        self.freq_shift = RandomDistribution.to_distribution(freq_shift)
        self.string = (
            self.__class__.__name__ + "(" + "freq_shift={}".format(freq_shift) + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        freq_shift = self.freq_shift()
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            avoid_aliasing = False
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None

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

                # Shift freq descriptions
                new_signal_desc.lower_frequency += float(freq_shift)
                new_signal_desc.upper_frequency += float(freq_shift)
                new_signal_desc.center_frequency += float(freq_shift)

                # Check bounds for aliasing
                if (
                    new_signal_desc.lower_frequency >= 0.5
                    or new_signal_desc.upper_frequency <= -0.5
                ):
                    avoid_aliasing = True
                    continue
                if (
                    new_signal_desc.lower_frequency < -0.45
                    or new_signal_desc.upper_frequency > 0.45
                ):
                    avoid_aliasing = True
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

                # Update bw & fc
                new_signal_desc.bandwidth = (
                    new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                )
                new_signal_desc.center_frequency = (
                    new_signal_desc.lower_frequency + new_signal_desc.bandwidth * 0.5
                )

                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description

            # Apply data augmentation
            if avoid_aliasing:
                # If any potential aliasing detected, perform shifting at higher sample rate
                new_data.iq_data = F.freq_shift_avoid_aliasing(data.iq_data, freq_shift)
            else:
                # Otherwise, use faster freq shifter
                new_data.iq_data = F.freq_shift(data.iq_data, freq_shift)
            return new_data
        else:
            output: np.ndarray = F.freq_shift(data, freq_shift)
            return output


class RandomDelayedFrequencyShift(SignalTransform):
    """Apply a delayed frequency shift to the input data

    Args:
        start_shift (:py:class:`~RandomDistribution`):
        freq_shift (:py:class:`~RandomDistribution`):
    """

    def __init__(
        self,
        start_shift: FloatParameter = UniformContinuousRD(0.1, 0.9),
        freq_shift: FloatParameter = UniformContinuousRD(-0.2, 0.2),
    ) -> None:
        super(RandomDelayedFrequencyShift, self).__init__()
        self.start_shift = RandomDistribution.to_distribution(start_shift)
        self.freq_shift = RandomDistribution.to_distribution(freq_shift)
        self.string = (
            self.__class__.__name__
            + "("
            + "start_shift={}, ".format(start_shift)
            + "freq_shift={}".format(freq_shift)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        start_shift: float = self.start_shift()
        # Randomly generate a freq shift that is not near the original fc
        freq_shift: float = 0.0
        while freq_shift < 0.05 and freq_shift > -0.05:
            freq_shift = self.freq_shift()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data
            num_iq_samples = data.iq_data.shape[0]

            # Setup new SignalDescription object
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            avoid_aliasing = False
            for signal_desc in signal_description_list:
                new_signal_desc_first_seg = deepcopy(signal_desc)
                assert new_signal_desc_first_seg.lower_frequency is not None
                assert new_signal_desc_first_seg.upper_frequency is not None
                assert new_signal_desc_first_seg.start is not None
                assert new_signal_desc_first_seg.stop is not None

                new_signal_desc_sec_seg = deepcopy(signal_desc)
                assert new_signal_desc_sec_seg.lower_frequency is not None
                assert new_signal_desc_sec_seg.upper_frequency is not None
                assert new_signal_desc_sec_seg.start is not None
                assert new_signal_desc_sec_seg.stop is not None

                # Check bounds for partial signals
                new_signal_desc_first_seg.lower_frequency = (
                    -0.5
                    if new_signal_desc_first_seg.lower_frequency < -0.5
                    else new_signal_desc_first_seg.lower_frequency
                )
                new_signal_desc_first_seg.upper_frequency = (
                    0.5
                    if new_signal_desc_first_seg.upper_frequency > 0.5
                    else new_signal_desc_first_seg.upper_frequency
                )
                new_signal_desc_first_seg.bandwidth = (
                    new_signal_desc_first_seg.upper_frequency
                    - new_signal_desc_first_seg.lower_frequency
                )
                new_signal_desc_first_seg.center_frequency = (
                    new_signal_desc_first_seg.lower_frequency
                    + new_signal_desc_first_seg.bandwidth * 0.5
                )

                # Update time for original segment if present in segment and add to list
                if new_signal_desc_first_seg.start < start_shift:
                    new_signal_desc_first_seg.stop = (
                        start_shift
                        if new_signal_desc_first_seg.stop > start_shift
                        else new_signal_desc_first_seg.stop
                    )
                    new_signal_desc_first_seg.duration = (
                        new_signal_desc_first_seg.stop - new_signal_desc_first_seg.start
                    )
                    # Append SignalDescription to list
                    new_signal_description.append(new_signal_desc_first_seg)

                # Begin second segment processing
                new_signal_desc_sec_seg.lower_frequency = (
                    -0.5
                    if new_signal_desc_sec_seg.lower_frequency < -0.5
                    else new_signal_desc_sec_seg.lower_frequency
                )
                new_signal_desc_sec_seg.upper_frequency = (
                    0.5
                    if new_signal_desc_sec_seg.upper_frequency > 0.5
                    else new_signal_desc_sec_seg.upper_frequency
                )
                new_signal_desc_sec_seg.bandwidth = (
                    new_signal_desc_sec_seg.upper_frequency
                    - new_signal_desc_sec_seg.lower_frequency
                )
                new_signal_desc_sec_seg.center_frequency = (
                    new_signal_desc_sec_seg.lower_frequency
                    + new_signal_desc_sec_seg.bandwidth * 0.5
                )

                # Update freqs for next segment
                new_signal_desc_sec_seg.lower_frequency += freq_shift
                new_signal_desc_sec_seg.upper_frequency += freq_shift
                new_signal_desc_sec_seg.center_frequency += freq_shift

                # Check bounds for aliasing
                if (
                    new_signal_desc_sec_seg.lower_frequency >= 0.5
                    or new_signal_desc_sec_seg.upper_frequency <= -0.5
                ):
                    avoid_aliasing = True
                    continue
                if (
                    new_signal_desc_sec_seg.lower_frequency < -0.45
                    or new_signal_desc_sec_seg.upper_frequency > 0.45
                ):
                    avoid_aliasing = True
                new_signal_desc_sec_seg.lower_frequency = (
                    -0.5
                    if new_signal_desc_sec_seg.lower_frequency < -0.5
                    else new_signal_desc_sec_seg.lower_frequency
                )
                new_signal_desc_sec_seg.upper_frequency = (
                    0.5
                    if new_signal_desc_sec_seg.upper_frequency > 0.5
                    else new_signal_desc_sec_seg.upper_frequency
                )

                # Update bw & fc
                new_signal_desc_sec_seg.bandwidth = (
                    new_signal_desc_sec_seg.upper_frequency
                    - new_signal_desc_sec_seg.lower_frequency
                )
                new_signal_desc_sec_seg.center_frequency = (
                    new_signal_desc_sec_seg.lower_frequency
                    + new_signal_desc_sec_seg.bandwidth * 0.5
                )

                # Update time for shifted segment if present in segment and add to list
                if new_signal_desc_sec_seg.stop > start_shift:
                    new_signal_desc_sec_seg.start = (
                        start_shift
                        if new_signal_desc_sec_seg.start < start_shift
                        else new_signal_desc_sec_seg.start
                    )
                    new_signal_desc_sec_seg.stop = new_signal_desc_sec_seg.stop
                    new_signal_desc_sec_seg.duration = (
                        new_signal_desc_sec_seg.stop - new_signal_desc_sec_seg.start
                    )
                    # Append SignalDescription to list
                    new_signal_description.append(new_signal_desc_sec_seg)

            # Update with the new SignalDescription
            new_data.signal_description = new_signal_description

            # Perform augmentation
            if avoid_aliasing:
                # If any potential aliasing detected, perform shifting at higher sample rate
                new_data.iq_data[
                    int(start_shift * num_iq_samples) :
                ] = F.freq_shift_avoid_aliasing(
                    data.iq_data[int(start_shift * num_iq_samples) :], freq_shift
                )
            else:
                # Otherwise, use faster freq shifter
                new_data.iq_data[int(start_shift * num_iq_samples) :] = F.freq_shift(
                    data.iq_data[int(start_shift * num_iq_samples) :], freq_shift
                )

        return new_data


class LocalOscillatorDrift(SignalTransform):
    """LocalOscillatorDrift is a transform modelling a local oscillator's drift in frequency by
    a random walk in frequency.

    Args:
        max_drift (FloatParameter, optional):
            [description]. Defaults to UniformContinuousRD(0.005,0.015).
        max_drift_rate (FloatParameter, optional):
            [description]. Defaults to UniformContinuousRD(0.001,0.01).

    """

    def __init__(
        self,
        max_drift: FloatParameter = UniformContinuousRD(0.005, 0.015),
        max_drift_rate: FloatParameter = UniformContinuousRD(0.001, 0.01),
        **kwargs,
    ) -> None:
        super(LocalOscillatorDrift, self).__init__(**kwargs)
        self.max_drift = RandomDistribution.to_distribution(max_drift)
        self.max_drift_rate = RandomDistribution.to_distribution(max_drift_rate)
        self.string = (
            self.__class__.__name__
            + "("
            + "max_drift={}, ".format(max_drift)
            + "max_drift_rate={}".format(max_drift_rate)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        max_drift = self.max_drift()
        max_drift_rate = self.max_drift_rate()

        iq_data = data.iq_data if isinstance(data, SignalData) else data
        assert iq_data is not None

        # Apply drift as a random walk.
        random_walk = RandomDistribution.rng.choice([-1, 1], size=iq_data.shape[0])

        # limit rate of change to at most 1/max_drift_rate times the length of the data sample
        frequency = np.cumsum(random_walk) * max_drift_rate / np.sqrt(iq_data.shape[0])

        # Every time frequency hits max_drift, reset to zero.
        while np.argmax(np.abs(frequency) > max_drift):
            idx = np.argmax(np.abs(frequency) > max_drift)
            offset = max_drift if frequency[idx] < 0 else -max_drift
            frequency[idx:] += offset
        min_offset: float = min(frequency)
        max_offset: float = max(frequency)

        complex_phase = np.exp(2j * np.pi * np.cumsum(frequency))
        iq_data = iq_data * complex_phase

        if isinstance(data, SignalData):
            assert data.signal_description is not None
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None

                # Expand frequency labels
                new_signal_desc.lower_frequency += float(min_offset)
                new_signal_desc.upper_frequency += float(max_offset)
                new_signal_desc.bandwidth = (
                    new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                )

                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description
            new_data.iq_data = iq_data
        else:
            new_data = iq_data

        return new_data


class GainDrift(SignalTransform):
    """GainDrift is a transform modelling a front end gain controller's drift in gain by
    a random walk in gain values.

    Args:
        max_drift (FloatParameter, optional):
            [description]. Defaults to UniformContinuousRD(0.005,0.015).
        min_drift (FloatParameter, optional):
            [description]. Defaults to UniformContinuousRD(0.005,0.015).
        drift_rate (FloatParameter, optional):
            [description]. Defaults to UniformContinuousRD(0.001,0.01).

    """

    def __init__(
        self,
        max_drift: FloatParameter = UniformContinuousRD(0.005, 0.015),
        min_drift: FloatParameter = UniformContinuousRD(0.005, 0.015),
        drift_rate: FloatParameter = UniformContinuousRD(0.001, 0.01),
        **kwargs,
    ) -> None:
        super(GainDrift, self).__init__(**kwargs)
        self.max_drift = RandomDistribution.to_distribution(max_drift)
        self.min_drift = RandomDistribution.to_distribution(min_drift)
        self.drift_rate = RandomDistribution.to_distribution(drift_rate)
        self.string = (
            self.__class__.__name__
            + "("
            + "max_drift={}, ".format(max_drift)
            + "min_drift={}, ".format(min_drift)
            + "drift_rate={}".format(drift_rate)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        max_drift = self.max_drift()
        min_drift = self.min_drift()
        drift_rate = self.drift_rate()

        iq_data = data.iq_data if isinstance(data, SignalData) else data
        assert iq_data is not None

        # Apply drift as a random walk.
        random_walk = RandomDistribution.rng.choice([-1, 1], size=iq_data.shape[0])

        # limit rate of change to at most 1/max_drift_rate times the length of the data sample
        gain = np.cumsum(random_walk) * drift_rate / np.sqrt(iq_data.shape[0])

        # Every time gain hits max_drift, reset to zero
        while np.argmax(gain > max_drift):
            idx = np.argmax(gain > max_drift)
            offset = gain[idx] - max_drift
            gain[idx:] -= offset
        # Every time gain hits min_drift, reset to zero
        while np.argmax(gain < min_drift):
            idx = np.argmax(gain < min_drift)
            offset = min_drift - gain[idx]
            gain[idx:] += offset
        iq_data = iq_data * (1 + gain)

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            new_data.iq_data = iq_data
        else:
            new_data = iq_data

        return new_data


class AutomaticGainControl(SignalTransform):
    """Automatic gain control (AGC) implementation

    Args:
        rand_scale (:py:class:`~torchsig.types.RandomDistribution`):
            Random scaling of alpha values

        initial_gain_db (:obj:`float`):
            Initial gain value in linear units

        alpha_smooth (:obj:`float`):
            Alpha for averaging the measured signal level level_n = level_n*alpha + level_n-1*(1 - alpha)

        alpha_track (:obj:`float`):
            Amount by which to adjust gain when in tracking state

        alpha_overflow (:obj:`float`):
            Amount by which to adjust gain when in overflow state [level_db + gain_db] >= max_level

        alpha_acquire (:obj:`float`):
            Amount by which to adjust gain when in acquire state abs([ref_level_db - level_db - gain_db]) >= track_range_db

        ref_level_db (:obj:`float`):
            Level to which we intend to adjust gain to achieve

        track_range_db (:obj:`float`):
            Range from ref_level_linear for which we can deviate before going into acquire state

        low_level_db (:obj:`float`):
            Level below which we disable AGC

        high_level_db (:obj:`float`):
            Level above which we go into overflow state

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.AutomaticGainControl(rand_scale=(1.0,10.0))

    """

    def __init__(
        self,
        rand_scale: FloatParameter = (1.0, 10.0),
        initial_gain_db: float = 0.0,
        alpha_smooth: float = 0.00004,
        alpha_overflow: float = 0.3,
        alpha_track: float = 0.0004,
        alpha_acquire: float = 0.04,
        ref_level_db: float = 0.0,
        track_range_db: float = 1.0,
        low_level_db: float = -80.0,
        high_level_db: float = 6.0,
    ) -> None:
        super(AutomaticGainControl, self).__init__()
        self.rand_scale = RandomDistribution.to_distribution(rand_scale)
        self.initial_gain_db = initial_gain_db
        self.alpha_smooth = alpha_smooth
        self.alpha_overflow = alpha_overflow
        self.alpha_track = alpha_track
        self.alpha_acquire = alpha_acquire
        self.ref_level_db = ref_level_db
        self.track_range_db = track_range_db
        self.low_level_db = low_level_db
        self.high_level_db = high_level_db
        self.string = (
            self.__class__.__name__
            + "("
            + "rand_scale={}, ".format(rand_scale)
            + "initial_gain_db={}, ".format(initial_gain_db)
            + "alpha_smooth={}, ".format(alpha_smooth)
            + "alpha_overflow={}, ".format(alpha_overflow)
            + "alpha_track={}, ".format(alpha_track)
            + "alpha_acquire={}, ".format(alpha_acquire)
            + "ref_level_db={}, ".format(ref_level_db)
            + "track_range_db={}, ".format(track_range_db)
            + "low_level_db={}, ".format(low_level_db)
            + "high_level_db={}".format(high_level_db)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        assert iq_data is not None
        rand_scale = self.rand_scale()
        alpha_acquire = np.random.uniform(
            self.alpha_acquire / rand_scale, self.alpha_acquire * rand_scale, 1
        )
        alpha_overflow = np.random.uniform(
            self.alpha_overflow / rand_scale, self.alpha_overflow * rand_scale, 1
        )
        alpha_track = np.random.uniform(
            self.alpha_track / rand_scale, self.alpha_track * rand_scale, 1
        )
        alpha_smooth = np.random.uniform(
            self.alpha_smooth / rand_scale, self.alpha_smooth * rand_scale, 1
        )

        ref_level_db = np.random.uniform(
            -0.5 + self.ref_level_db, 0.5 + self.ref_level_db, 1
        )

        iq_data = F.agc(
            np.ascontiguousarray(iq_data, dtype=np.complex64),
            np.float64(self.initial_gain_db),
            np.float64(alpha_smooth),
            np.float64(alpha_track),
            np.float64(alpha_overflow),
            np.float64(alpha_acquire),
            np.float64(ref_level_db),
            np.float64(self.track_range_db),
            np.float64(self.low_level_db),
            np.float64(self.high_level_db),
        )

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            new_data.iq_data = iq_data
        else:
            new_data = iq_data

        return new_data


class IQImbalance(SignalTransform):
    """Applies various types of IQ imbalance to a tensor

    Args:
        iq_amplitude_imbalance_db (:py:class:`~torchsig.types.RandomDistribution`):

        iq_phase_imbalance (:py:class:`~torchsig.types.RandomDistribution`):

        iq_dc_offset_db (:py:class:`~torchsig.types.RandomDistribution`):

    Note:
        For more information about IQ imbalance in RF systems, check out
        https://www.mathworks.com/help/comm/ref/iqimbalance.html

    Example:
        >>> import torchsig.transforms as ST
        >>> # IQ imbalance with default params
        >>> transform = ST.IQImbalance()

    """

    def __init__(
        self,
        iq_amplitude_imbalance_db=UniformContinuousRD(0, 3),
        iq_phase_imbalance=UniformContinuousRD(
            -np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0
        ),
        iq_dc_offset_db=UniformContinuousRD(-0.1, 0.1),
    ) -> None:
        super(IQImbalance, self).__init__()
        self.amp_imbalance = RandomDistribution.to_distribution(
            iq_amplitude_imbalance_db
        )
        self.phase_imbalance = RandomDistribution.to_distribution(iq_phase_imbalance)
        self.dc_offset = RandomDistribution.to_distribution(iq_dc_offset_db)
        self.string = (
            self.__class__.__name__
            + "("
            + "amp_imbalance={}, ".format(iq_amplitude_imbalance_db)
            + "phase_imbalance={}, ".format(iq_phase_imbalance)
            + "dc_offset={}".format(iq_dc_offset_db)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        amp_imbalance = self.amp_imbalance()
        phase_imbalance = self.phase_imbalance()
        dc_offset = self.dc_offset()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.iq_imbalance(
                data.iq_data, amp_imbalance, phase_imbalance, dc_offset
            )
        else:
            data = F.iq_imbalance(data, amp_imbalance, phase_imbalance, dc_offset)
        return data


class RollOff(SignalTransform):
    """Applies a band-edge RF roll-off effect simulating front end filtering

    Args:
        cutoff (:py:class:`~torchsig.types.RandomDistribution`):

        cfo (:py:class:`~torchsig.types.RandomDistribution`):

    """

    def __init__(
        self,
        cutoff: FloatParameter = UniformContinuousRD(0.25, 0.5),
        cfo: FloatParameter = UniformContinuousRD(-0.1, 0.1),
    ) -> None:
        super(RollOff, self).__init__()
        self.cutoff = RandomDistribution.to_distribution(cutoff)
        self.cfo = RandomDistribution.to_distribution(cfo)
        self.string = (
            self.__class__.__name__
            + "("
            + "cutoff={}, ".format(cutoff)
            + "cfo={}, ".format(cfo)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            data.iq_data = F.roll_off(data.iq_data, self.cutoff(), self.cfo())
        else:
            data = F.roll_off(data, self.cutoff(), self.cfo())
        return data


class AddSlope(SignalTransform):
    """Add the slope of each sample with its preceeding sample to itself.
    Creates a weak 0 Hz IF notch filtering effect

    """

    def __init__(self, **kwargs) -> None:
        super(AddSlope, self).__init__(**kwargs)

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Apply data augmentation
            new_data.iq_data = F.add_slope(data.iq_data)
            return new_data
        else:
            output: np.ndarray = F.add_slope(data)
            return output


class SpectralInversion(SignalTransform):
    """Applies a spectral inversion"""

    def __init__(self) -> None:
        super(SpectralInversion, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Perform data augmentation
            new_data.iq_data = F.spectral_inversion(data.iq_data)

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None
                assert new_signal_desc.center_frequency is not None

                # Invert frequency labels
                original_lower = new_signal_desc.lower_frequency
                original_upper = new_signal_desc.upper_frequency
                new_signal_desc.lower_frequency = original_upper * -1
                new_signal_desc.upper_frequency = original_lower * -1
                new_signal_desc.center_frequency *= -1

                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description
            return new_data

        else:
            output: np.ndarray = F.spectral_inversion(data)
            return output


class ChannelSwap(SignalTransform):
    """Transform that swaps the I and Q channels of complex input data"""

    def __init__(self) -> None:
        super(ChannelSwap, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None
                assert new_signal_desc.center_frequency is not None

                # Invert frequency labels
                original_lower = new_signal_desc.lower_frequency
                original_upper = new_signal_desc.upper_frequency
                new_signal_desc.lower_frequency = original_upper * -1
                new_signal_desc.upper_frequency = original_lower * -1
                new_signal_desc.center_frequency *= -1

                new_signal_description.append(new_signal_desc)

            new_data.signal_description = new_signal_description

            # Perform data augmentation
            new_data.iq_data = F.channel_swap(data.iq_data)
            return new_data
        else:
            output: np.ndarray = F.channel_swap(data)
            return output


class RandomMagRescale(SignalTransform):
    """Randomly apply a magnitude rescaling, emulating a change in a receiver's
    gain control

    Args:
        start (:py:class:`~torchsig.types.RandomDistribution`):
            start sets the time when the rescaling kicks in

        scale (:py:class:`~torchsig.types.RandomDistribution`):
            scale sets the magnitude of the rescale

    """

    def __init__(
        self,
        start: FloatParameter = UniformContinuousRD(0.0, 0.9),
        scale: FloatParameter = UniformContinuousRD(-4.0, 4.0),
    ) -> None:
        super(RandomMagRescale, self).__init__()
        self.start = RandomDistribution.to_distribution(start)
        self.scale = RandomDistribution.to_distribution(scale)
        self.string = (
            self.__class__.__name__
            + "("
            + "start={}, ".format(start)
            + "scale={}".format(scale)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        start = self.start()
        scale = self.scale()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            new_data.iq_data = F.mag_rescale(data.iq_data, start, scale)
            return new_data
        else:
            output: np.ndarray = F.mag_rescale(data, start, scale)
            return output


class RandomDropSamples(SignalTransform):
    """Randomly drop IQ samples from the input data of specified durations and
    with specified fill techniques:
    * `ffill` (front fill): replace drop samples with the last previous value
    * `bfill` (back fill): replace drop samples with the next value
    * `mean`: replace drop samples with the mean value of the full data
    * `zero`: replace drop samples with zeros

    Transform is based off of the
    `TSAug Dropout Transform <https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/dropout.py>`_.

    Args:
         drop_rate (:py:class:`~torchsig.types.RandomDistribution`):
            drop_rate sets the rate at which to drop samples

        size (:py:class:`~torchsig.types.RandomDistribution`):
            size sets the size of each instance of dropped samples

        fill (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            fill sets the method of how the dropped samples should be filled

    """

    def __init__(
        self,
        drop_rate: FloatParameter = UniformContinuousRD(0.01, 0.05),
        size: IntParameter = UniformDiscreteRD(np.arange(1, 10, dtype=int)),
        fill: List[str] = (["ffill", "bfill", "mean", "zero"]),
    ) -> None:
        super(RandomDropSamples, self).__init__()
        self.drop_rate = RandomDistribution.to_distribution(drop_rate)
        self.size = RandomDistribution.to_distribution(size)
        self.fill = RandomDistribution.to_distribution(fill)
        self.string = (
            self.__class__.__name__
            + "("
            + "drop_rate={}, ".format(drop_rate)
            + "size={}, ".format(size)
            + "fill={}".format(fill)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        drop_rate = self.drop_rate()
        fill = self.fill()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            drop_instances = int(data.iq_data.shape[0] * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(
                1, data.iq_data.shape[0] - max(drop_sizes) - 1, drop_instances
            ).astype(int)

            new_data.iq_data = F.drop_samples(
                data.iq_data, drop_starts, drop_sizes, fill
            )
            return new_data

        else:
            drop_instances = int(data.shape[0] * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(
                0, data.shape[0] - max(drop_sizes), drop_instances
            ).astype(int)

            output: np.ndarray = F.drop_samples(data, drop_starts, drop_sizes, fill)
            return output


class Quantize(SignalTransform):
    """Quantize the input to the number of levels specified

    Args:
         num_levels (:py:class:`~torchsig.types.RandomDistribution`):
            num_levels sets the number of quantization levels

        round_type (:py:class:`~Callable`, :obj:`str`, :obj:`list`):
            round_type sets the rounding direction of the quantization. Options
            include: 'floor', 'middle', & 'ceiling'

    """

    def __init__(
        self,
        num_levels: IntParameter = UniformDiscreteRD(
            np.asarray([16, 24, 32, 40, 48, 56, 64], dtype=int)
        ),
        round_type: List[str] = (["floor", "middle", "ceiling"]),
    ) -> None:
        super(Quantize, self).__init__()
        self.num_levels = RandomDistribution.to_distribution(num_levels)
        self.round_type = RandomDistribution.to_distribution(round_type)
        self.string = (
            self.__class__.__name__
            + "("
            + "num_levels={}, ".format(num_levels)
            + "round_type={}".format(round_type)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        num_levels = self.num_levels()
        round_type = self.round_type()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            new_data.iq_data = F.quantize(data.iq_data, num_levels, round_type)
            return new_data
        else:
            output: np.ndarray = F.quantize(data, num_levels, round_type)
            return output


class Clip(SignalTransform):
    """Clips the input values to a percentage of the max/min values

    Args:
        clip_percentage (:py:class:`~torchsig.types.RandomDistribution`):
            Specifies the percentage of the max/min values to clip


    """

    def __init__(
        self,
        clip_percentage: FloatParameter = UniformContinuousRD(0.75, 0.95),
        **kwargs,
    ) -> None:
        super(Clip, self).__init__(**kwargs)
        self.clip_percentage = RandomDistribution.to_distribution(clip_percentage)
        self.string = (
            self.__class__.__name__
            + "("
            + "clip_percentage={}".format(clip_percentage)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        clip_percentage = self.clip_percentage()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Apply data augmentation
            new_data.iq_data = F.clip(data.iq_data, clip_percentage)
            return new_data

        else:
            output: np.ndarray = F.clip(data, clip_percentage)
            return output


class RandomConvolve(SignalTransform):
    """Convolve a random complex filter with the input data

    Args:
        num_taps (:py:class:`~torchsig.types.RandomDistribution`):
            Number of taps for the random filter


        alpha (:py:class:`~torchsig.types.RandomDistribution`):
            The effect of the filtered data is dampened using an alpha factor
            that determines the weightings for the summing of the filtered data
            and the original data. `alpha` should be in range `[0,1]` where a
            value of 0 applies all of the weight to the original data, and a
            value of 1 applies all of the weight to the filtered data

    """

    def __init__(
        self,
        num_taps: IntParameter = UniformDiscreteRD(np.arange(2, 5, dtype=int)),
        alpha: FloatParameter = UniformContinuousRD(0.1, 0.5),
        **kwargs,
    ) -> None:
        super(RandomConvolve, self).__init__(**kwargs)
        self.num_taps = RandomDistribution.to_distribution(num_taps)
        self.alpha = RandomDistribution.to_distribution(alpha)
        self.string = (
            self.__class__.__name__
            + "("
            + "num_taps={}, ".format(num_taps)
            + "alpha={}".format(alpha)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        num_taps = int(self.num_taps())
        alpha = self.alpha()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Apply data augmentation
            new_data.iq_data = F.random_convolve(data.iq_data, num_taps, alpha)
            return new_data
        else:
            output: np.ndarray = F.random_convolve(data, num_taps, alpha)
            return output


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

        alpha (:py:class:`~torchsig.types.RandomDistribution`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example


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
        dataset: SignalDataset,
        alpha: FloatParameter = UniformContinuousRD(-5, -3),
    ) -> None:
        super(DatasetBasebandMixUp, self).__init__()
        self.alpha = RandomDistribution.to_distribution(alpha)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Input checks
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            if len(signal_description_list) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected.".format(
                        len(signal_description_list)
                    )
                )
            assert signal_description_list[0].snr is not None

            # Calculate target SNR of signal to be inserted
            target_snr_db = signal_description_list[0].snr + alpha

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if isinstance(insert_data, SignalData):
                assert insert_data.iq_data is not None
                insert_iq_data: np.ndarray = insert_data.iq_data
            else:
                insert_iq_data = insert_data

            if insert_iq_data.shape[0] != data.iq_data.shape[0]:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_iq_data.shape[0], data.iq_data.shape[0]
                    )
                )
            insert_signal_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=insert_signal_description,
            )
            insert_signal_data.iq_data = insert_iq_data

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
            new_signal_description.append(signal_description_list[0])
            assert insert_signal_data.signal_description is not None
            insert_desc: List[SignalDescription] = (
                [insert_signal_data.signal_description]
                if isinstance(insert_signal_data.signal_description, SignalDescription)
                else insert_signal_data.signal_description
            )
            new_signal_description.append(insert_desc[0])
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

        alpha (:py:class:`~torchsig.types.RandomDistribution`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example


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
        dataset: SignalDataset,
        alpha: FloatParameter = UniformContinuousRD(0.2, 0.5),
    ) -> None:
        super(DatasetBasebandCutMix, self).__init__()
        self.alpha = RandomDistribution.to_distribution(alpha)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            # Input checks
            assert data.iq_data is not None
            assert data.signal_description is not None

            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            if len(signal_description_list) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected.".format(
                        len(signal_description_list)
                    )
                )
            assert signal_description_list[0].snr is not None

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if isinstance(insert_data, SignalData):
                assert insert_data.iq_data is not None
                insert_iq_data: np.ndarray = insert_data.iq_data
            else:
                insert_iq_data = insert_data
            num_iq_samples = data.iq_data.shape[0]
            if insert_iq_data.shape[0] != num_iq_samples:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_iq_data.shape[0], data.iq_data.shape[0]
                    )
                )
            insert_signal_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=insert_signal_description,
            )
            insert_signal_data.iq_data = insert_iq_data

            # Set insert data's SNR
            target_snr_transform = TargetSNR(signal_description_list[0].snr)
            insert_signal_data = target_snr_transform(insert_signal_data)
            assert insert_signal_data.iq_data is not None

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
            new_signal_description: List[SignalDescription] = []
            if insert_start != 0 and insert_stop != num_iq_samples:
                # Data description becomes two SignalDescriptions
                new_signal_desc = deepcopy(signal_description_list[0])
                new_signal_desc.start = 0.0
                new_signal_desc.stop = insert_start / num_iq_samples
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
                new_signal_desc = deepcopy(signal_description_list[0])
                new_signal_desc.start = insert_stop / num_iq_samples
                new_signal_desc.stop = 1.0
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            elif insert_start == 0:
                # Data description remains one SignalDescription up to end
                new_signal_desc = deepcopy(signal_description_list[0])
                new_signal_desc.start = insert_stop / num_iq_samples
                new_signal_desc.stop = 1.0
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            else:
                # Data description remains one SignalDescription at beginning
                new_signal_desc = deepcopy(signal_description_list[0])
                new_signal_desc.start = 0.0
                new_signal_desc.stop = insert_start / num_iq_samples
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_description.append(new_signal_desc)
            # Repeat for insert's SignalDescription

            assert insert_signal_data.signal_description is not None
            insert_desc: List[SignalDescription] = (
                [insert_signal_data.signal_description]
                if isinstance(insert_signal_data.signal_description, SignalDescription)
                else insert_signal_data.signal_description
            )
            new_signal_desc = deepcopy(insert_desc[0])
            assert new_signal_desc.start is not None
            assert new_signal_desc.stop is not None
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
         cut_dur (:py:class:`~torchsig.types.RandomDistribution`):
            cut_dur sets the duration of the region to cut out

        cut_type (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            cut_type sets the type of data to fill in the cut region with from
            the options: `zeros`, `ones`, `low_noise`, `avg_noise`, and
            `high_noise`
    """

    def __init__(
        self,
        cut_dur: FloatParameter = UniformContinuousRD(0.01, 0.2),
        cut_type: List[str] = (
            ["zeros", "ones", "low_noise", "avg_noise", "high_noise"]
        ),
    ) -> None:
        super(CutOut, self).__init__()
        self.cut_dur = RandomDistribution.to_distribution(cut_dur)
        self.cut_type = RandomDistribution.to_distribution(cut_type)
        self.string = (
            self.__class__.__name__
            + "("
            + "cut_dur={}, ".format(cut_dur)
            + "cut_type={}".format(cut_type)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        cut_dur = self.cut_dur()
        cut_start = np.random.uniform(0.0, 1.0 - cut_dur)
        cut_type = self.cut_type()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None

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
            new_data.iq_data = F.cut_out(data.iq_data, cut_start, cut_dur, cut_type)
            return new_data
        else:
            output: np.ndarray = F.cut_out(data, cut_start, cut_dur, cut_type)
            return output


class PatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `"PatchShuffle Regularization" <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~torchsig.types.RandomDistribution`):
            patch_size sets the size of each patch to shuffle

        shuffle_ratio (:py:class:`~torchsig.types.RandomDistribution`):
            shuffle_ratio sets the ratio of the patches to shuffle


    """

    def __init__(
        self,
        patch_size: IntParameter = UniformDiscreteRD(np.arange(3, 10, dtype=int)),
        shuffle_ratio: FloatParameter = UniformContinuousRD(0.01, 0.05),
    ) -> None:
        super(PatchShuffle, self).__init__()
        self.patch_size = RandomDistribution.to_distribution(patch_size)
        self.shuffle_ratio = RandomDistribution.to_distribution(shuffle_ratio)
        self.string = (
            self.__class__.__name__
            + "("
            + "patch_size={}, ".format(patch_size)
            + "shuffle_ratio={}".format(shuffle_ratio)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        patch_size = int(self.patch_size())
        shuffle_ratio = self.shuffle_ratio()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            new_data.iq_data = F.patch_shuffle(data.iq_data, patch_size, shuffle_ratio)
            return new_data
        else:
            output: np.ndarray = F.patch_shuffle(data, patch_size, shuffle_ratio)
            return output


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

        alpha (:py:class:`~torchsig.types.RandomDistribution`):
            alpha sets the difference in durations between the main dataset
            example and the inserted example

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets.wideband_sig53 import WidebandSig53
        >>> # Add signals from the `ModulationsDataset`
        >>> dataset = WidebandSig53('.')
        >>> transform = ST.DatasetWidebandCutMix(dataset=dataset,alpha=(0.2,0.7))

    """

    def __init__(
        self,
        dataset: SignalDataset,
        alpha: FloatParameter = UniformContinuousRD(0.2, 0.7),
    ) -> None:
        super(DatasetWidebandCutMix, self).__init__()
        self.alpha = RandomDistribution.to_distribution(alpha)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if isinstance(insert_data, SignalData):
                assert insert_data.iq_data is not None
                insert_iq_data: np.ndarray = insert_data.iq_data
            else:
                insert_iq_data = insert_data
            num_iq_samples = data.iq_data.shape[0]
            if insert_iq_data.shape[0] != num_iq_samples:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_iq_data.shape[0], data.iq_data.shape[0]
                    )
                )

            # Mask both data examples based on alpha and a random start value
            insert_num_iq_samples = int(alpha * num_iq_samples)
            insert_start: int = np.random.randint(
                num_iq_samples - insert_num_iq_samples
            )
            insert_stop = insert_start + insert_num_iq_samples
            data.iq_data[insert_start:insert_stop] = 0
            insert_iq_data[:insert_start] = 0.0
            insert_iq_data[insert_stop:] = 0.0
            insert_start //= num_iq_samples
            insert_dur = insert_num_iq_samples / num_iq_samples

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data + insert_iq_data

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None

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

        alpha (:py:class:`~torchsig.types.RandomDistribution`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets.wideband_sig53 import WidebandSig53
        >>> # Add signals from the `WidebandSig53` Dataset
        >>> dataset = WidebandSig53('.')
        >>> transform = ST.DatasetWidebandMixUp(dataset=dataset,alpha=(0.4,0.6))

    """

    def __init__(
        self,
        dataset: SignalDataset,
        alpha: FloatParameter = UniformContinuousRD(0.4, 0.6),
    ) -> None:
        super(DatasetWidebandMixUp, self).__init__()
        self.alpha = RandomDistribution.to_distribution(alpha)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        alpha = self.alpha()
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Randomly sample from provided dataset
            idx = np.random.randint(self.dataset_num_samples)
            insert_data, insert_signal_description = self.dataset[idx]
            if isinstance(insert_data, SignalData):
                assert insert_data.iq_data is not None
                insert_iq_data: np.ndarray = insert_data.iq_data
            else:
                insert_iq_data = insert_data
            if insert_iq_data.shape[0] != data.iq_data.shape[0]:
                raise ValueError(
                    "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                    Found {}, but expected {} samples".format(
                        insert_iq_data.shape[0], data.iq_data.shape[0]
                    )
                )

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data * (1 - alpha) + insert_iq_data * alpha

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            new_signal_description.extend(signal_description_list)
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

        overlap_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The ratio of the (nfft-1) value to use as the overlap parameter for
            the spectrogram operation. Setting as ratio ensures the overlap is
            a lower value than the bin size.

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
        nfft: IntParameter = UniformDiscreteRD(np.arange(256, 1024, dtype=int)),
        overlap_ratio: FloatParameter = UniformContinuousRD(0.0, 0.2),
        window_fcn: Callable[[int], np.ndarray] = np.blackman,
        mode: str = "complex",
        width: int = 512,
        height: int = 512,
    ) -> None:
        super(SpectrogramRandomResizeCrop, self).__init__()
        self.nfft = RandomDistribution.to_distribution(nfft)
        self.overlap_ratio = RandomDistribution.to_distribution(overlap_ratio)
        self.window_fcn = window_fcn
        self.mode = mode
        self.width = width
        self.height = height
        self.string = (
            self.__class__.__name__
            + "("
            + "nfft={}, ".format(nfft)
            + "overlap_ratio={}, ".format(overlap_ratio)
            + "window_fcn={}, ".format(window_fcn)
            + "mode={}, ".format(mode)
            + "width={}, ".format(width)
            + "height={}".format(height)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        nfft = int(self.nfft())
        nperseg = nfft
        overlap_ratio = self.overlap_ratio()
        noverlap = int(overlap_ratio * (nfft - 1))

        iq_data = data.iq_data if isinstance(data, SignalData) else data
        assert iq_data is not None

        # First, perform the random spectrogram operation
        spec_data = F.spectrogram(
            iq_data,
            nperseg,
            noverlap,
            nfft,
            scaling="density",
            detrend="constant",
            window_fcn=self.window_fcn,
            mode=self.mode,
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
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = spec_data

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None

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
            return new_data

        else:
            output: np.ndarray = spec_data
            return output


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
         drop_rate (:py:class:`~torchsig.types.RandomDistribution`):
            drop_rate sets the rate at which to drop samples

        size (:py:class:`~torchsig.types.RandomDistribution`):
            size sets the size of each instance of dropped samples

        fill (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            fill sets the method of how the dropped samples should be filled

    """

    def __init__(
        self,
        drop_rate: FloatParameter = UniformContinuousRD(0.001, 0.005),
        size: IntParameter = UniformDiscreteRD(np.arange(1, 10, dtype=int)),
        fill: List[str] = (
            ["ffill", "bfill", "mean", "zero", "low", "min", "max", "ones"]
        ),
    ) -> None:
        super(SpectrogramDropSamples, self).__init__()
        self.drop_rate = RandomDistribution.to_distribution(drop_rate)
        self.size = RandomDistribution.to_distribution(size)
        self.fill = RandomDistribution.to_distribution(fill)
        self.string = (
            self.__class__.__name__
            + "("
            + "drop_rate={}, ".format(drop_rate)
            + "size={}, ".format(size)
            + "fill={}".format(fill)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        drop_rate = self.drop_rate()
        fill = self.fill()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
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

            new_data.iq_data = F.drop_spec_samples(
                data.iq_data, drop_starts, drop_sizes, fill
            )
            return new_data

        else:
            drop_instances = int(data.shape[0] * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(
                0, data.shape[0] - max(drop_sizes), drop_instances
            ).astype(int)

            output: np.ndarray = F.drop_spec_samples(
                data,
                drop_starts,
                drop_sizes,
                fill,
            )
            return output


class SpectrogramPatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `PatchShuffle Regularization <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~torchsig.types.RandomDistribution`):
            patch_size sets the size of each patch to shuffle

        shuffle_ratio (:py:class:`~torchsig.types.RandomDistribution`):
            shuffle_ratio sets the ratio of the patches to shuffle

    """

    def __init__(
        self,
        patch_size: IntParameter = UniformDiscreteRD(np.arange(2, 16, dtype=int)),
        shuffle_ratio: FloatParameter = UniformContinuousRD(0.01, 0.10),
    ) -> None:
        super(SpectrogramPatchShuffle, self).__init__()
        self.patch_size = RandomDistribution.to_distribution(patch_size)
        self.shuffle_ratio = RandomDistribution.to_distribution(shuffle_ratio)
        self.string = (
            self.__class__.__name__
            + "("
            + "patch_size={}, ".format(patch_size)
            + "shuffle_ratio={}".format(shuffle_ratio)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        patch_size = int(self.patch_size())
        shuffle_ratio = self.shuffle_ratio()

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            # Perform data augmentation
            new_data.iq_data = F.spec_patch_shuffle(
                data.iq_data, patch_size, shuffle_ratio
            )
            return new_data
        else:
            output: np.ndarray = F.spec_patch_shuffle(
                data,
                patch_size,
                shuffle_ratio,
            )
            return output


class SpectrogramTranslation(SignalTransform):
    """Transform that inputs a spectrogram and applies a random time/freq
    translation

    Args:
         time_shift (:py:class:`~torchsig.types.RandomDistribution`):
            time_shift sets the translation along the time-axis

        freq_shift (:py:class:`~torchsig.types.RandomDistribution`):
            freq_shift sets the translation along the freq-axis

    """

    def __init__(
        self,
        time_shift: IntParameter = UniformDiscreteRD(np.arange(-128, 128, dtype=int)),
        freq_shift: IntParameter = UniformDiscreteRD(np.arange(-128, 128, dtype=int)),
    ) -> None:
        super(SpectrogramTranslation, self).__init__()
        self.time_shift = RandomDistribution.to_distribution(time_shift)
        self.freq_shift = RandomDistribution.to_distribution(freq_shift)
        self.string = (
            self.__class__.__name__
            + "("
            + "time_shift={}, ".format(time_shift)
            + "freq_shift={}".format(freq_shift)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        time_shift = int(self.time_shift())
        freq_shift = int(self.freq_shift())

        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )

            new_data.iq_data = F.spec_translate(data.iq_data, time_shift, freq_shift)

            # Update SignalDescription
            new_signal_description: List[SignalDescription] = []
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None

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
            return new_data

        else:
            output: np.ndarray = F.spec_translate(data, time_shift, freq_shift)
            return output


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

    def __init__(self, dataset: SignalDataset) -> None:
        super(SpectrogramMosaicCrop, self).__init__()
        self.dataset = dataset
        self.string = self.__class__.__name__ + "(" + "dataset={}".format(dataset) + ")"

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data: SignalData = SignalData(
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
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None

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
                signal_description_list = (
                    [curr_signal_desc]
                    if isinstance(curr_signal_desc, SignalDescription)
                    else curr_signal_desc
                )
                for signal_desc in signal_description_list:
                    new_signal_desc = deepcopy(signal_desc)
                    assert new_signal_desc.start is not None
                    assert new_signal_desc.stop is not None
                    assert new_signal_desc.lower_frequency is not None
                    assert new_signal_desc.upper_frequency is not None

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
            return new_data

        else:
            # Read shapes
            channels, height, width = data.shape

            # Randomly decide the new x0, y0 point of the stitched images
            x0 = np.random.randint(0, width)
            y0 = np.random.randint(0, height)

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
            output: np.ndarray = full_mosaic[:, y0 : y0 + height, x0 : x0 + width]
            return output


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

    def __init__(self, dataset: SignalDataset) -> None:
        super(SpectrogramMosaicDownsample, self).__init__()
        self.dataset = dataset
        self.string = self.__class__.__name__ + "(" + "dataset={}".format(dataset) + ")"

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

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
            signal_description_list: List[SignalDescription] = (
                [data.signal_description]
                if isinstance(data.signal_description, SignalDescription)
                else data.signal_description
            )
            for signal_desc in signal_description_list:
                new_signal_desc = deepcopy(signal_desc)
                assert new_signal_desc.start is not None
                assert new_signal_desc.stop is not None
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None

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
                signal_description_list = (
                    [curr_signal_desc]
                    if isinstance(curr_signal_desc, SignalDescription)
                    else curr_signal_desc
                )
                for signal_desc in signal_description_list:
                    new_signal_desc = deepcopy(signal_desc)
                    assert new_signal_desc.start is not None
                    assert new_signal_desc.stop is not None
                    assert new_signal_desc.lower_frequency is not None
                    assert new_signal_desc.upper_frequency is not None

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
            return new_data

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
            output: np.ndarray = full_mosaic[:, ::2, ::2]
            return output


class SpectrogramImage(SignalTransform):
    """Transforms SignalData to spectrogram image PNGs (numpy arrays)

    Args:
        size
        colormap

    """

    def __init__(self, size=512, colormap="viridis"):
        super(SpectrogramImage, self).__init__()
        self.size = size
        self.colormap = colormap
        self.nperseg = self.size
        self.noverlap = 0
        self.nfft = self.size
        self.mode = "psd"

    def __call__(self, data) -> SignalData:
        if isinstance(data, SignalData):
            data.iq_data = F.spectrogram_image(
                data.iq_data,  # type: ignore
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                mode=self.mode,
                colormap=self.colormap,
            )
        else:
            data = F.spectrogram_image(
                data,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                mode=self.mode,
                colormap=self.colormap,
            )

        return data
