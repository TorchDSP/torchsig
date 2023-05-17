from torchsig.datasets.synthetic import (
    ConstellationDataset,
    default_const_map,
    OFDMDataset,
)
from torchsig.datasets.modulations import ModulationsDataset
from unittest import TestCase
import time


class GenerateConstellationsFigures(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.longMessage = True

    def test_modulations_benchmark(self):
        dataset = ModulationsDataset(
            classes=("ofdm-1024",),
            use_class_idx=False,
            level=0,
            num_iq_samples=4096,
            num_samples=10,
            include_snr=True,
        )
        print(len(dataset))

    def test_constellation_benchmark(self):
        for modulation_name in default_const_map.keys():
            dataset = ConstellationDataset(
                [modulation_name],
                num_iq_samples=4096,
                num_samples_per_class=10000,
                iq_samples_per_symbol=2,
                pulse_shape_filter=None,
                random_pulse_shaping=False,
                random_data=False,
                use_gpu=False,
            )
            with self.subTest():
                start_time = time.time()
                for iq_data, label in dataset:
                    pass
                stop_time = time.time()
            total_time = stop_time - start_time
            self.assertTrue(
                True,
                msg="Took {} seconds per sample to generate {}".format(
                    total_time / 10000, modulation_name
                ),
            )
        return True

    def test_ofdm_benchmark(self):
        # num_subcarriers = (
        #     64,
        #     128,
        #     180,
        #     256,
        #     300,
        #     512,
        #     600,
        #     900,
        #     1024,
        #     1200,
        # )
        dataset = OFDMDataset(
            constellations=("bpsk",),  # sub-carrier modulations
            num_subcarriers=(2048,),  # possible number of subcarriers
            num_iq_samples=4096,
            num_samples_per_class=10000,
            random_data=True,
            sidelobe_suppression_methods=("lpf",),
            dc_subcarrier=("on",),
        )
        start_time = time.time()
        for iq_data, label in dataset:
            pass
        stop_time = time.time()
        total_time = stop_time - start_time
        self.assertTrue(
            True,
            msg="Took {} seconds per sample to generate ofdm".format(
                total_time / 10000
            ),
        )
        return True
