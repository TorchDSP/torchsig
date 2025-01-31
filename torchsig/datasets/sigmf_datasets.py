"""Dataset class and reader for SigMF files
"""
from torchsig.utils.types import Signal, ModulatedRFMetadata, SignalData
from torchsig.transforms.transforms import Identity

from torch.utils.data import Dataset
import numpy as np
import glob
from sigmf import SigMFFile, sigmffile
import os
from typing import Optional, Callable, List

class SigMFDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        class_list: List[str],
        metadata_path: str = None,
        is_complex: bool = True,
        shuffle: bool = False,
        transform: Optional[Callable] = Identity(),
        target_transform: Optional[Callable] = Identity(),
        seed: int = None
    ):
        self.data_path = data_path
        self.metadata_path = data_path if metadata_path is None else metadata_path
        self.class_list = class_list
        self.is_complex = is_complex
        self.shuffle = shuffle
        self.T = transform
        self.TT = target_transform
        self.random_generator = np.random.default_rng(seed=seed)
        self.seed = self.random_generator.bit_generator.seed_seq.entropy

        # get list of data filenames
        self.data_filenames = glob.glob(os.path.join(self.data_path, "*.sigmf-data"))
        # shuffle reading in if wanted
        if self.shuffle:
            self.random_generator.shuffle(self.data_filenames)

    def __len__(self) -> int:
        return len(self.data_filenames)

    def __getitem__(self, idx: int) -> tuple:
        # load signal from sigmf file
        data_filename = self.data_filenames[idx]
        base_filename = os.path.splitext(os.path.basename(data_filename))[0]
        metadata_filename = os.path.join(self.metadata_path, f"{base_filename}.sigmf-meta")

        # if not os.path.exists(metadata_filename):
        #     raise OSError(f"{metadata_filename} does not exist.")

        # turn into Signal
        signal = sigmf_reader(metadata_filename, self.class_list, self.is_complex)

        # apply transforms
        signal = self.T(signal)

        # apply target transforms
        target = self.TT(signal)

        return signal["data"]["samples"], target

def sigmf_reader(filename: str, class_list: List[str], complex_data: bool = True) -> Signal:
    signal = sigmffile.fromfile(filename)
    if complex_data:
        samples = signal.read_samples().view(np.complex64).flatten()
    else:
        samples = signal.read_samples().view(np.float32)
    

    sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    sample_count = signal.sample_count
    num_samples = sample_count
    sample_duration = sample_count / sample_rate
    annotations = signal.get_annotations()

    metadatas = []

    for adx, annotation in enumerate(annotations):
        start_idx = annotation[SigMFFile.START_INDEX_KEY]
        start = start_idx / sample_count
        length = annotation[SigMFFile.LENGTH_INDEX_KEY]
        duration = length / sample_count
        stop = start + duration
        
        # capture = signal.get_capture_info(start_idx)
        # center_freq = capture.get(SigMFFile.FREQUENCY_KEY, 0)
        # lower_freq = freq_center - 0.5*sample_rate
        # upper_freq = freq_center + 0.5*sample_rate

        lower_freq = annotation.get(SigMFFile.FLO_KEY)
        upper_freq = annotation.get(SigMFFile.FHI_KEY)
        bandwidth = upper_freq - lower_freq
        center_freq = lower_freq + (bandwidth/2)

        class_name = annotation.get(SigMFFile.LABEL_KEY)
        class_index = class_list.index(class_name)

        # TODO [EO] what should we do about these values?
        snr = None
        bits_per_symbol = None
        samples_per_symbol = None
        excess_bandwidth = None

        metadata = ModulatedRFMetadata(
            sample_rate=sample_rate,
            num_samples=num_samples,
            complex=complex_data,
            lower_freq=lower_freq,
            upper_freq=upper_freq,
            center_freq=center_freq,
            bandwidth=bandwidth,
            start=start,
            stop=stop,
            duration=duration,
            snr=snr,
            bits_per_symbol=bits_per_symbol,
            samples_per_symbol=samples_per_symbol,
            excess_bandwidth=excess_bandwidth,
            class_name=class_name,
            class_index=class_index,
        )
        metadatas.append(metadata)

    return Signal(data=SignalData(samples=samples), metadata=metadatas)
