import numpy as np
from typing import Optional, Callable, List
from torch.utils.data import ConcatDataset
from torchsig.datasets.synthetic import DigitalModulationDataset, OFDMDataset
from torchsig.transforms.target_transforms.target_transforms import (
    DescToClassIndexSNR,
    DescToClassIndex,
    DescToClassNameSNR,
    DescToClassName,
)
from torchsig.transforms.transforms import (
    Compose,
    RandomApply,
)
from torchsig.transforms.wireless_channel.wce import (
    RandomPhaseShift,
    RayleighFadingChannel,
    TargetSNR,
)
from torchsig.transforms.signal_processing.sp import Normalize, RandomResample
from torchsig.transforms.system_impairment.si import (
    RandomTimeShift,
    RandomFrequencyShift,
    IQImbalance,
)


class ModulationsDataset(ConcatDataset):
    """ModulationsDataset serves as a standard dataset for many RF machine
    learning tasks in the modulation recognition/classification domain.

    Args:
        level (:obj:`str` int):
            * level 0 represents perfect modulations as if they synthesized by a transmitter
            * level 1 represents impairments related to a cabled environment in which a receiver is uncalibrated and unsynchronized
            * level 2 represents impairments related to over-the-air transmission in which the receiver is a non-cooperative receiver

        transform (:class:`torchsig.transforms.Transform`):
            Any additional transforms to append onto the existing transforms of the dataset -- usually to change the
            complex floating point output into another format.

    Impairments:
        * Phase shift (uniform between -pi and pi):
            Phase shifts occur as a result of the carrier wave (complex sinusoid) arriving at different times at the
            receiver.

        * Time shift (uniform between +- half a sample):
            More easily understood as a delay shift or a sample time offset, this is when a signal arrives
            at an ADC at different times. Plus or minus half a sample represents the maximum deviation from a sample
            point of view.

        * Frequency shift (uniform between -.16 and .16 relative to the sample rate):
            Frequency shifts come from Doppler shifts (movement of Tx or Rx) and from local-oscillator offsets.
            The maximum reasonable deviation for a signal of bandwidth fs/2 would be -.25 -- otherwise the kind of
            frequency shift applied here will cause the signal to "wrap-around" in the frequency domain.

        * IQ Imbalance (amplitude, phase, and DC offset):
            It's not clear what these values should be as typical receivers are calibrated to compensate for this.
            However, there are still scenarios in which it is difficult to properly calibrate this.

        * Rayleigh Fading (uniform between 2 and 20 taps with tapered power delay profile):
            This is typical in a multi-path fading environment

        * Additive White Gaussian Noise (uniform between -2 and 18 dB in an Eb/N0 sense):
            AWGN comes from a variety of effects. We include only down to -2 dB because less than this makes samples
            essentially useless for training.

        * Random Resample (uniform between .75 and 1.5):
            This effectively makes the largest bandwidth fs*2/3 and the smallest around fs/3 assuming the original
            signal was at fs/2. This represents a situation in which the bandwidth of the original signal is not
            known and may be the output of a channelizer with some fixed bandwidth.

        * Length of 4096
            For higher order modulations, it may be necessary to have more samples to see everyone symbol
            with high probability

        * Signal bandwidths at ~fs/2
            This is a good compromise between critically samples signals and vastly oversampled signals. Vastly
            oversampled signals will require models to take in perhaps millions of IQ samples to get good classification
            performance on higher order modulations. Critically sampled signals can be easily corrupted by even the
            more simple channel impairments.

    """

    default_classes = [
        "ook",
        "bpsk",
        "4pam",
        "4ask",
        "qpsk",
        "8pam",
        "8ask",
        "8psk",
        "16qam",
        "16pam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32pam",
        "32ask",
        "32psk",
        "64qam",
        "64pam",
        "64ask",
        "64psk",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        use_class_idx: bool = False,
        level: int = 0,
        num_iq_samples: int = 2048,
        num_samples: int = 4500,
        include_snr: bool = False,
        eb_no: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ):
        classes = self.default_classes if classes is None else classes
        # Set the target transform based on input options if none provided
        if not target_transform:
            if use_class_idx:
                if include_snr:
                    target_transform = DescToClassIndexSNR(class_list=classes)
                else:
                    target_transform = DescToClassIndex(class_list=classes)
            else:
                if include_snr:
                    target_transform = DescToClassNameSNR()
                else:
                    target_transform = DescToClassName()
        num_samples_per_class = int(num_samples / len(classes))
        self.class_dict = dict(zip(classes, range(len(classes))))
        self.include_snr = include_snr

        # Extract class info
        ofdm_classes = []
        digital_classes = []
        num_subcarriers = []
        for class_name in classes:
            if "ofdm" in class_name:
                ofdm_classes.append(class_name)
                num_subcarriers.append(int(class_name[5:]))
            else:
                digital_classes.append(class_name)
        num_digital = len(digital_classes)
        num_ofdm = len(ofdm_classes)

        if level == 0:
            random_pulse_shaping = False
            internal_transforms = Compose(
                [
                    TargetSNR((100, 100), eb_no=eb_no),
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 1:
            random_pulse_shaping = True
            internal_transforms = Compose(
                [
                    RandomPhaseShift((-1, 1)),
                    RandomTimeShift((-0.5, 0.5)),
                    RandomFrequencyShift((-0.16, 0.16)),
                    IQImbalance(
                        (-3, 3),
                        (-np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0),
                        (-0.1, 0.1),
                    ),
                    RandomResample((0.75, 1.5), num_iq_samples=num_iq_samples),
                    TargetSNR((80, 80), eb_no=eb_no),
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 2:
            random_pulse_shaping = True
            internal_transforms = Compose(
                [
                    RandomApply(RandomPhaseShift((-1, 1)), 0.9),
                    RandomApply(RandomTimeShift((-32, 32)), 0.9),
                    RandomApply(RandomFrequencyShift((-0.16, 0.16)), 0.7),
                    RandomApply(
                        RayleighFadingChannel(
                            (0.05, 0.5), power_delay_profile=(1.0, 0.5, 0.1)
                        ),
                        0.5,
                    ),
                    RandomApply(
                        IQImbalance(
                            (-3, 3),
                            (-np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0),
                            (-0.1, 0.1),
                        ),
                        0.9,
                    ),
                    RandomApply(
                        RandomResample((0.75, 1.5), num_iq_samples=num_iq_samples),
                        0.5,
                    ),
                    TargetSNR((-2, 30), eb_no=eb_no),
                    Normalize(norm=np.inf),
                ]
            )
        else:
            raise ValueError("Level is unrecognized. Should be 0, 1 or 2.")

        if transform is not None:
            internal_transforms = Compose(
                [
                    internal_transforms,
                    transform,
                ]
            )

        if num_digital > 0:
            digital_dataset = DigitalModulationDataset(
                modulations=digital_classes,  # effectively uses all modulations
                num_iq_samples=num_iq_samples,
                num_samples_per_class=num_samples_per_class,
                iq_samples_per_symbol=2,
                random_data=True,
                random_pulse_shaping=random_pulse_shaping,
                transform=internal_transforms,
                target_transform=target_transform,
            )

        if num_ofdm > 0:
            sidelobe_suppression_methods = ("lpf", "win_start")
            ofdm_dataset = OFDMDataset(
                constellations=(
                    "bpsk",
                    "qpsk",
                    "16qam",
                    "64qam",
                    "256qam",
                    "1024qam",
                ),  # sub-carrier modulations
                num_subcarriers=tuple(
                    num_subcarriers
                ),  # possible number of subcarriers
                num_iq_samples=num_iq_samples,
                num_samples_per_class=num_samples_per_class,
                random_data=True,
                sidelobe_suppression_methods=sidelobe_suppression_methods,
                dc_subcarrier=("on", "off"),
                transform=internal_transforms,
                target_transform=target_transform,
            )

        if num_digital > 0 and num_ofdm > 0:
            super(ModulationsDataset, self).__init__(
                [digital_dataset, ofdm_dataset], **kwargs
            )
        elif num_digital > 0:
            super(ModulationsDataset, self).__init__([digital_dataset], **kwargs)
        elif num_ofdm > 0:
            super(ModulationsDataset, self).__init__([ofdm_dataset], **kwargs)
        else:
            raise ValueError("Input classes must contain at least 1 valid class")

    def __getitem__(self, item):
        return super(ModulationsDataset, self).__getitem__(item)
