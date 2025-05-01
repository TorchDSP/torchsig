"""Dataset Base Classes for creation and static loading.
"""

from __future__ import annotations

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.signals.signal_types import DatasetSignal, DatasetDict
from torchsig.signals.builder import SignalBuilder
import torchsig.signals.builders as signal_builders
from torchsig.utils.random import Seedable
from torchsig.utils.dsp import compute_spectrogram
from torchsig.datasets.dataset_utils import (
    to_dataset_metadata, 
    frequency_shift_signal,
    dataset_full_path
)
from torchsig.utils.printing import generate_repr_str
from torchsig.utils.verify import verify_transforms, verify_target_transforms
from torchsig.utils.file_handlers.zarr import ZarrFileHandler
from torchsig.datasets.dataset_utils import dataset_yaml_name, writer_yaml_name
from torchsig.utils.coordinate_system import (
    Coordinate,
    Rectangle,
    is_box_overlap
)

# Third Party
from torch.utils.data import Dataset, IterableDataset
import numpy as np

# Built-In
from typing import Tuple, Dict, TYPE_CHECKING
from pathlib import Path
import yaml
import warnings

if TYPE_CHECKING:
    from torchsig.utils.file_handlers.zarr import TorchSigFileHandler

class TorchsigIterableDataset(IterableDataset, Seedable):
    """Creates a new TorchSig dataset that generates data infinitely unless `num_samples` inside `dataset_metadata` is defined.
    
    This base class provides the functionality to generate signals and write them to disk if necessary. The dataset will continue 
    to generate samples infinitely unless a `num_samples` value is defined in the `dataset_metadata`.

    """ 
    
    def __init__(
        self, 
        dataset_metadata: DatasetMetadata | str | dict,
        **kwargs
    ):
        """
        Initializes the dataset, creates signal builders, and prepares file handlers based on metadata.

        Args:
            dataset_metadata (DatasetMetadata | str | dict): The dataset metadata.
            **kwargs: Additional keyword arguments for initialization.

        """
        Seedable.__init__(self, **kwargs)

        self._dataset_metadata: DatasetMetadata = to_dataset_metadata(dataset_metadata)
        self._dataset_metadata.add_parent(self)
        self.num_samples_generated = 0
        self.builders: Dict[str, SignalBuilder] = self._initialize_builders() # initialize builders


    def __iter__(self):
        return self

    def __next__(self):
        """Returns a dataset sample and corresponding targets for a given index.

        Returns:
            Tuple[np.ndarray, Tuple]: The sample data and the target values.

        Raises:
            IndexError: If the index is out of bounds of the generated samples.
        """
        
        # user requesting another sample at index +1 larger than current list of generates samples
        # generate new sample
        sample = self.__generate_new_signal__()
        
        # apply dataset transforms
        sample = self.dataset_metadata.impairments.dataset_transforms(sample)

        # apply user transforms
        for transform in self.dataset_metadata.transforms:
            sample = transform(sample)

        # convert to DatasetDict
        sample = DatasetDict(signal=sample)

        targets = []
        # apply target transforms
        for target_transform in self.dataset_metadata.target_transforms:
            # apply transform to all metadatas
            sample.metadata = target_transform(sample.metadata)
            # get target outputs
            target_transform_output = []
            for signal_metadata in sample.metadata:
                # extract output from metadata
                # as required by TT target output field name
                signal_output = []
                for field in target_transform.targets_metadata:
                    signal_output.append(signal_metadata[field])
                
                signal_output = tuple(signal_output)
                target_transform_output.append(signal_output)

            targets.append(target_transform_output)

        # convert targets as a list of target transform output ordered by transform
        # to ordered by signal
        # e.g., [(transform 1 output for all signals), (transform 2 output for all signals), ... ] ->
        # [signal 1 outputs, signal 2 outputs, ... ]
        targets = list(zip(*targets))
               
        if len(self.dataset_metadata.target_transforms) == 0:
            # no target transform applied
            targets = sample.metadata
        elif self.dataset_metadata.dataset_type == 'narrowband':
            # only one signal in list for narrowband
            # unwrap targets
            targets = [item[0] if len(item) == 1 else item for row in targets for item in row]
            # unwrap any target transform output that produced a tuple
            targets = targets[0] if len(targets) == 1 else tuple(targets)
        else:
            # wideband
            targets = [tuple([item[0] if len(item) == 1 else item for item in row]) for row in targets]
            # unwrap any target transform output that produced a tuple
            targets = [row[0] if len(row) == 1 else row for row in targets]


        self.num_samples_generated += 1

        return sample.data, targets

    def reset(self):
        """Resets the dataset to its initial state."""
        self._dataset_metadata.num_samples_generated = 0
    
    def _initialize_builders(self) -> Dict[str, SignalBuilder]:
        """
        Initializes signal builders from the class list based on the signal classes supported by the dataset.

        Returns:
            Dict[str, SignalBuilder]: A dictionary where the key is the signal class name, and the value is the corresponding 
            SignalBuilder object.
        """
        builders = {}
        # for each builder
        for builder_name in signal_builders.__all__:
            builder = getattr(signal_builders, builder_name) # get builder class
            # check if class list has any of the builder's supported classes
            matching_classes = set(self._dataset_metadata.class_list) & set(builder.supported_classes)
            if len(matching_classes) > 0: # yes
                for c in matching_classes:
                    # add builder
                    builders[c] = builder(self._dataset_metadata, c,)
                    builders[c].add_parent(self)
        return builders

    def __str__(self) -> str:
        """Returns a string representation of the dataset, including its metadata and the signal builders.

        Returns:
            str: String representation of the dataset.
        """
        max_width = 100
        # first_col_width = 29
        # second_col_width = max_width - first_col_width
        # array_width_indent = first_col_width + 2

        builders_str = "\n".join([f"{key:<15}: {value}" for key, value in self.builders.items()])
        class_str = f"{self.__class__.__name__}"
        center_width = (max_width - len(class_str)) // 2

        return (
            f"\n{'-' * center_width} {self.__class__.__name__} {'-' * center_width}\n"
            f"{self.dataset_metadata}\n"
            f"\nBuilders"
            f"{'-' * max_width}\n"
            f"{builders_str}\n"
        )

    def __repr__(self):
        """Returns a string representation of the object with all its attributes.

        Returns:
            str: String representation of the object with its attributes.
        """
        return generate_repr_str(self)

    def _build_noise_floor(self) -> np.ndarray:
        """Generates the noise floor for the dataset by creating an IQ sample and applying a frequency-domain noise estimation.

        Returns:
            np.ndarray: The generated IQ samples representing the noise floor.
        """   
        real_samples = self.random_generator.normal(
            0,
            1,
            self.dataset_metadata.num_iq_samples_dataset
        )
        imag_samples = self.random_generator.normal(
            0,
            1,
            self.dataset_metadata.num_iq_samples_dataset
        )
        # combine real and imaginary portions of noise
        iq_samples = real_samples + 1j* imag_samples
        # compute an estimate of the noise floor in the frequency domain. use a large stride to process a subset
        # of the data since not many FFTs are needed to be averaged for the noise
        noise_spectrogram_db = compute_spectrogram(iq_samples,self.dataset_metadata.fft_size,self.dataset_metadata.fft_stride*16)
        # average over time
        noise_fft_db = np.mean(noise_spectrogram_db,axis=1)
        # estimate the average noise value in dB in the frequency domain
        noise_avg_db = np.mean(noise_fft_db)
        # compute the correction factor as the distance from the desired level
        correction_db = self.dataset_metadata.noise_power_db-noise_avg_db
        # apply the correction
        correction = 10**(correction_db/10)
        iq_samples = np.sqrt(correction)*iq_samples

        iq_samples = iq_samples.astype(np.complex64)

        return iq_samples


    def __generate_new_signal__(self) -> DatasetSignal:
        """Generates a new dataset signal/sample.

        Args:
            idx (int): The index for the new signal.

        Returns:
            DatasetSignal: A new generated dataset signal containing the data and metadata.
        """     
        
        # build noise floor
        iq_samples = self._build_noise_floor()

        # empty signal list initialization
        signals = []

        # determine number of signals in sample
        num_signals_to_generate = self.random_generator.integers(low=self.dataset_metadata.num_signals_min, high = self.dataset_metadata.num_signals_max+1)

        # generate individual bursts
        for i in range(num_signals_to_generate):

            # choose random signal
            class_name = self._random_signal_class()

            # get builder for signal class
            builder = self.builders[class_name]

            # generate signal at complex baseband
            new_signal = builder.build()

            # apply signal transforms
            new_signal = self.dataset_metadata.impairments.signal_transforms(new_signal)

            # frequency shift signal
            # after signal transforms applied at complex baseband
            new_signal = frequency_shift_signal(
                new_signal,
                center_freq_min=self.dataset_metadata.signal_center_freq_min,
                center_freq_max=self.dataset_metadata.signal_center_freq_max,
                sample_rate=self.dataset_metadata.sample_rate,
                frequency_max=self.dataset_metadata.frequency_max,
                frequency_min=self.dataset_metadata.frequency_min,
                random_generator=self.random_generator,
            )

            # place signal on iq sample cut
            iq_samples[new_signal.metadata.start_in_samples:new_signal.metadata.stop_in_samples] += new_signal.data

            # append the signal on the list
            signals.append(new_signal)

        # form the sample (dataset object)
        sample = DatasetSignal(data=iq_samples, signals=signals)

        return sample

    # Read-Only properties

    @property
    def dataset_metadata(self):
        """Returns the dataset metadata.

        Returns:
            DatasetMetadata: The dataset metadata.
        """    
        return self._dataset_metadata

    # Functions

    def _random_signal_class(self):     
        """Randomly selects which signal to create next.

        Returns:
            str: A signal class name from the available signal classes.
        """
        return self.random_generator.choice(self.dataset_metadata.class_list, p=self.dataset_metadata.class_distribution)



class NewTorchSigDataset(Dataset, Seedable):
    """Creates a new TorchSig dataset that generates data infinitely unless `num_samples` inside `dataset_metadata` is defined.
    
    This base class provides the functionality to generate signals and write them to disk if necessary. The dataset will continue 
    to generate samples infinitely unless a `num_samples` value is defined in the `dataset_metadata`.

    """ 
    
    def __init__(
        self, 
        dataset_metadata: DatasetMetadata | str | dict,
        **kwargs
    ):
        """
        Initializes the dataset, creates signal builders, and prepares file handlers based on metadata.

        Args:
            dataset_metadata (DatasetMetadata | str | dict): The dataset metadata.
            **kwargs: Additional keyword arguments for initialization.

        """
        Seedable.__init__(self, **kwargs)

        self._dataset_metadata: DatasetMetadata = to_dataset_metadata(dataset_metadata)
        self._dataset_metadata.add_parent(self)
        self.num_samples_generated = 0
        self.builders: Dict[str, SignalBuilder] = self._initialize_builders() # initialize builders
        self._current_idx: int = 0  # Internal counter for iterator usage

        warnings.warn("NewTorchSigDataset will become a torch.IterableDataset in the future.",
                      FutureWarning
        )


    def __iter__(self):
        return self

    def __next__(self):
        # Return the next sample
        result = self[self._current_idx]
        self._current_idx += 1
        return result

    def reset(self):
        """Resets the dataset to its initial state."""
        self._dataset_metadata.num_samples_generated = 0
        self._current_idx = 0

    
    def _initialize_builders(self) -> Dict[str, SignalBuilder]:
        """
        Initializes signal builders from the class list based on the signal classes supported by the dataset.

        Returns:
            Dict[str, SignalBuilder]: A dictionary where the key is the signal class name, and the value is the corresponding 
            SignalBuilder object.
        """
        builders = {}
        # for each builder
        for builder_name in signal_builders.__all__:
            builder = getattr(signal_builders, builder_name) # get builder class
            # check if class list has any of the builder's supported classes
            matching_classes = set(self._dataset_metadata.class_list) & set(builder.supported_classes)
            if len(matching_classes) > 0: # yes
                for c in matching_classes:
                    # add builder
                    builders[c] = builder(self._dataset_metadata, c,)
                    builders[c].add_parent(self)
        return builders
    
    def __len__(self) -> int:
        """Returns the number of samples generated in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        # If infinite dataset, return how many samples have been generated
        if self.dataset_metadata.num_samples is None:
            return self.num_samples_generated
        # else:
        return self.dataset_metadata.num_samples

    def __str__(self) -> str:
        """Returns a string representation of the dataset, including its metadata and the signal builders.

        Returns:
            str: String representation of the dataset.
        """
        max_width = 100
        # first_col_width = 29
        # second_col_width = max_width - first_col_width
        # array_width_indent = first_col_width + 2

        builders_str = "\n".join([f"{key:<15}: {value}" for key, value in self.builders.items()])
        class_str = f"{self.__class__.__name__}"
        center_width = (max_width - len(class_str)) // 2

        return (
            f"\n{'-' * center_width} {self.__class__.__name__} {'-' * center_width}\n"
            f"{self.dataset_metadata}\n"
            f"\nBuilders"
            f"{'-' * max_width}\n"
            f"{builders_str}\n"
        )

    def __repr__(self):
        """Returns a string representation of the object with all its attributes.

        Returns:
            str: String representation of the object with its attributes.
        """
        return generate_repr_str(self)

    def _build_noise_floor(self) -> np.ndarray:
        """Generates the noise floor for the dataset by creating an IQ sample and applying a frequency-domain noise estimation.

        Returns:
            np.ndarray: The generated IQ samples representing the noise floor.
        """   
        real_samples = self.random_generator.normal(
            0,
            1,
            self.dataset_metadata.num_iq_samples_dataset
        )
        imag_samples = self.random_generator.normal(
            0,
            1,
            self.dataset_metadata.num_iq_samples_dataset
        )
        # combine real and imaginary portions of noise
        iq_samples = real_samples + 1j* imag_samples
        # compute an estimate of the noise floor in the frequency domain. use a large stride to process a subset
        # of the data since not many FFTs are needed to be averaged for the noise
        noise_spectrogram_db = compute_spectrogram(iq_samples,self.dataset_metadata.fft_size,self.dataset_metadata.fft_stride*16)
        # average over time
        noise_fft_db = np.mean(noise_spectrogram_db,axis=1)
        # estimate the average noise value in dB in the frequency domain
        noise_avg_db = np.mean(noise_fft_db)
        # compute the correction factor as the distance from the desired level
        correction_db = self.dataset_metadata.noise_power_db-noise_avg_db
        # apply the correction
        correction = 10**(correction_db/10)
        iq_samples = np.sqrt(correction)*iq_samples

        iq_samples = iq_samples.astype(np.complex64)

        return iq_samples


    def __generate_new_signal__(self) -> DatasetSignal:
        """Generates a new dataset signal/sample.

        Args:
            idx (int): The index for the new signal.

        Returns:
            DatasetSignal: A new generated dataset signal containing the data and metadata.
        """     
        
        # build noise floor
        iq_samples = self._build_noise_floor()

        # empty signal list initialization
        signals = []

        # determine number of signals in sample
        num_signals_to_generate = self.random_generator.integers(low=self.dataset_metadata.num_signals_min, high = self.dataset_metadata.num_signals_max+1)

        # list of rectangles representing the individual signals within wideband cut
        signal_rectangle_list = []

        # TODO: make dataset_metadata
        #overlap_prob = 0.5
        overlap_prob = 0

        # counter to avoid stuck in infinite loop
        counter = 0
        counter_max = 10*num_signals_to_generate

        # TODO: note that this code needs to be replicated and/or replaced on both this class
        #       AND torchsig iterable

        # generate individual bursts
        num_signals_created = 0
        while (num_signals_created < num_signals_to_generate and counter < counter_max):

            # increment fail-safe counter
            counter += 1

            # choose random signal
            class_name = self._random_signal_class()

            # get builder for signal class
            builder = self.builders[class_name]

            # generate signal at complex baseband
            new_signal = builder.build()

            # apply signal transforms
            new_signal = self.dataset_metadata.impairments.signal_transforms(new_signal)

            # frequency shift signal
            # after signal transforms applied at complex baseband
            new_signal = frequency_shift_signal(
                new_signal,
                center_freq_min=self.dataset_metadata.signal_center_freq_min,
                center_freq_max=self.dataset_metadata.signal_center_freq_max,
                sample_rate=self.dataset_metadata.sample_rate,
                frequency_max=self.dataset_metadata.frequency_max,
                frequency_min=self.dataset_metadata.frequency_min,
                random_generator=self.random_generator,
            )

            # TODO: turn this into a function
            # calculate start and stop time in terms of FFT number
            fft_start_time = np.round(new_signal.metadata.start_in_samples/self.dataset_metadata.fft_size)
            fft_stop_time  = np.round(new_signal.metadata.stop_in_samples/self.dataset_metadata.fft_size)

            # calculate bin position in FFT
            fs = self.dataset_metadata.sample_rate
            fft_start_bin_norm = (new_signal.metadata.lower_freq + (fs/2))/(fs/2)
            fft_stop_bin_norm  = (new_signal.metadata.upper_freq + (fs/2))/(fs/2)

            fft_start_bin_index = np.round(fft_start_bin_norm * self.dataset_metadata.fft_size)
            fft_stop_bin_index  = np.round(fft_stop_bin_norm  * self.dataset_metadata.fft_size)

            # map the position into retangle coordinates
            lower_left_coord = Coordinate(fft_start_time,fft_start_bin_index)
            upper_right_coord = Coordinate(fft_stop_time,fft_stop_bin_index)

            # turn into a rectangle
            new_rectangle = Rectangle(lower_left_coord,upper_right_coord)

            # initialize the boolean value which determines if there is overlap or not
            has_overlap = False

            # determine if overlap
            if (len(signal_rectangle_list) > 0):
                # check to see if the current rectangle overlaps with any signals currently
                # in the spectrogram
                for reference_box in signal_rectangle_list:
                    # check for invidivual overlap
                    individual_overlap = is_box_overlap(new_rectangle,reference_box)
                    # combine with previous potential overlap checks
                    has_overlap = has_overlap or individual_overlap

            # signal is used if there is no overlap OR with some random chance
            if (has_overlap == False or np.random.uniform(0,1) < overlap_prob): # TODO: use RNG
                # store the rectangle for future overlap checking
                signal_rectangle_list.append( new_rectangle )
                # place signal on iq sample cut
                iq_samples[new_signal.metadata.start_in_samples:new_signal.metadata.stop_in_samples] += new_signal.data
                # append the signal on the list
                signals.append(new_signal)
                # update counter
                num_signals_created += 1
            # else:
            #     loop back to top and attempt to recreate another signal

        # form the sample (dataset object)
        sample = DatasetSignal(data=iq_samples, signals=signals)

        return sample

    def _verify_idx(self, idx: int) -> None:
        is_infinite_dataset = self.dataset_metadata.num_samples is None
        idx_in_bounds = idx >= 0
        if not is_infinite_dataset:
            idx_in_bounds = idx_in_bounds and idx < self.dataset_metadata.num_samples
        sample_already_generated = idx < self.num_samples_generated
        # idx_skipping = idx > self.dataset_metadata.num_samples_generated
        

        if idx < 0:
            # idx less than zero
            raise IndexError(f"index {idx} is less than zero and is out of bounds.")
        
        if not is_infinite_dataset and not idx_in_bounds:
            # is finite dataset
            # idx is not between 0 and num_samples
            raise IndexError(f"index {idx} is out of bounds for finite dataset with {self.dataset_metadata.num_samples} num_samples.")
        
        if sample_already_generated:
            # idx < number of generated samples
            # requesting previously generated sample
            raise IndexError(f"cannot access previously generated samples in {self.__class__.__name__} for index {idx}. Ensure you are accessing dataset in order (0, 1, 2,...) or save dataset with DatasetCreator")
        
        # elif idx_skipping:
        #     # idx > number of generated samples
        #     # requesting to generate sample out of order
        #     # e.g., calling dataset[100] without calling dataset[0]...dataset[99] first in order
        #     raise IndexError(f"index {idx} requesting sample out of order. Must request next sample at index {self.dataset_metadata.num_samples_generated}. Ensure you are accessing dataset in order (0, 1, 2,...).")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """Returns a dataset sample and corresponding targets for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, Tuple]: The sample data and the target values.

        Raises:
            IndexError: If the index is out of bounds of the generated samples.
        """

        # verifies idx
        self._verify_idx(idx)
        
        # user requesting another sample at index +1 larger than current list of generates samples
        # generate new sample
        sample = self.__generate_new_signal__()
        
        # apply dataset transforms
        sample = self.dataset_metadata.impairments.dataset_transforms(sample)

        # apply user transforms
        for transform in self.dataset_metadata.transforms:
            sample = transform(sample)

        # convert to DatasetDict
        sample = DatasetDict(signal=sample)

        targets = []
        # apply target transforms
        for target_transform in self.dataset_metadata.target_transforms:
            # apply transform to all metadatas
            sample.metadata = target_transform(sample.metadata)
            # get target outputs
            target_transform_output = []
            for signal_metadata in sample.metadata:
                # extract output from metadata
                # as required by TT target output field name
                signal_output = []
                for field in target_transform.targets_metadata:
                    signal_output.append(signal_metadata[field])
                
                signal_output = tuple(signal_output)
                target_transform_output.append(signal_output)

            targets.append(target_transform_output)

        # convert targets as a list of target transform output ordered by transform
        # to ordered by signal
        # e.g., [(transform 1 output for all signals), (transform 2 output for all signals), ... ] ->
        # [signal 1 outputs, signal 2 outputs, ... ]
        targets = list(zip(*targets))
               
        if len(self.dataset_metadata.target_transforms) == 0:
            # no target transform applied
            targets = sample.metadata
        elif self.dataset_metadata.dataset_type == 'narrowband':
            # only one signal in list for narrowband
            # unwrap targets
            targets = [item[0] if len(item) == 1 else item for row in targets for item in row]
            # unwrap any target transform output that produced a tuple
            targets = targets[0] if len(targets) == 1 else tuple(targets)
        else:
            # wideband
            targets = [tuple([item[0] if len(item) == 1 else item for item in row]) for row in targets]
            # unwrap any target transform output that produced a tuple
            targets = [row[0] if len(row) == 1 else row for row in targets]


        self.num_samples_generated += 1

        return sample.data, targets
            

    # Read-Only properties

    @property
    def dataset_metadata(self):
        """Returns the dataset metadata.

        Returns:
            DatasetMetadata: The dataset metadata.
        """    
        return self._dataset_metadata

    # Functions

    def _random_signal_class(self):     
        """Randomly selects which signal to create next.

        Returns:
            str: A signal class name from the available signal classes.
        """
        return self.random_generator.choice(self.dataset_metadata.class_list, p=self.dataset_metadata.class_distribution)


class StaticTorchSigDataset(Dataset):
    """Static Dataset class, which loads pre-generated data from a directory.

    This class assumes that the dataset has already been generated and saved to disk using a subclass of `NewTorchSigDataset`. 
    It allows loading raw or processed data from disk for inference or analysis.
    
    Args:
        root (str): The root directory where the dataset is stored.
        impairment_level (int): Defines impairment level 0, 1, 2.
        dataset_type (str): Type of the dataset, either "narrowband" or "wideband".
        transforms (list, optional): Transforms to apply to the data (default: []).
        target_transforms (list, optional): Target transforms to apply (default: []).
        file_handler_class (TorchSigFileHandler, optional): Class used for reading the dataset (default: ZarrFileHandler).
    """   

    def __init__(
        self,
        root: str,
        impairment_level: int,
        dataset_type: str,
        transforms: list = [],
        target_transforms: list = [],
        file_handler_class: TorchSigFileHandler = ZarrFileHandler,
        train: bool = None,
        # **kwargs
    ):
        self.root = Path(root)
        self.impairment_level = impairment_level
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.file_handler = file_handler_class
        self.train = train

        # create filepath to saved dataset
        # e.g., root/torchsig_narrowband_clean/
        self.full_root = dataset_full_path(
            dataset_type = self.dataset_type,
            impairment_level = self.impairment_level,
            train = self.train
        )
        self.full_root = f"{self.root}/{self.full_root}"

        # check dataset data type from writer_info.dataset_yaml_name
        with open(f"{self.full_root}/{writer_yaml_name}", 'r') as f:
            writer_info = yaml.load(f, Loader=yaml.FullLoader)
            self.raw = writer_info['save_type'] == "raw"

        # need to create new dataset metadata from dataset_info.yaml
        self.dataset_metadata = to_dataset_metadata(f"{self.full_root}/{dataset_yaml_name}")

        # dataset size
        self.num_samples = self.file_handler.size(self.full_root)

        self._verify()

    def _verify(self):
        # Transforms
        self.transforms = verify_transforms(self.transforms)

        # Target Transforms
        self.target_transforms = verify_target_transforms(self.target_transforms)
        # print(self.target_transforms)
        # print("verify")


    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """Retrieves a sample from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, Tuple]: The data and targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx >= 0 and idx < self.__len__():
            # load data and metadata
            # data: np.ndarray
            # signal_metadatas: List[dict]
            if self.raw:
                # loading in raw IQ data and signal metadata
                data, signal_metadatas = self.file_handler.static_load(self.full_root, idx)

                # convert to DatasetSignal
                sample = DatasetSignal(
                    data = data, 
                    signals = signal_metadatas, 
                    dataset_metadata = self.dataset_metadata,
                )

                # apply user transforms
                for t in self.transforms:
                    sample = t(sample)

                # convert to DatasetDict
                sample = DatasetDict(signal=sample)

                # apply target transforms
                targets = []
                for target_transform in self.target_transforms:
                    # apply transform to all metadatas
                    sample.metadata = target_transform(sample.metadata)
                    # get target outputs
                    target_transform_output = []
                    for signal_metadata in sample.metadata:
                        # extract output from metadata
                        # as required by TT target output field name
                        signal_output = []
                        for field in target_transform.targets_metadata:
                            signal_output.append(signal_metadata[field])
                        
                        signal_output = tuple(signal_output)
                        target_transform_output.append(signal_output)

                    targets.append(target_transform_output)

                # convert targets as a list of target transform output ordered by transform
                # to ordered by signal
                # e.g., [(transform 1 output for all signals), (transform 2 output for all signals), ... ] ->
                # [signal 1 outputs, signal 2 outputs, ... ]
                targets = list(zip(*targets))
                    
                if len(self.target_transforms) == 0:
                    # no target transform applied
                    targets = sample.metadata
                elif self.dataset_type == 'narrowband':
                    # only one signal in list for narrowband
                    # unwrap targets
                    targets = [item[0] if len(item) == 1 else item for row in targets for item in row]
                    # unwrap any target transform output that produced a tuple
                    targets = targets[0] if len(targets) == 1 else tuple(targets)
                else:
                    # wideband
                    targets = [tuple([item[0] if len(item) == 1 else item for item in row]) for row in targets]
                    # unwrap any target transform output that produced a tuple
                    targets = [row[0] if len(row) == 1 else row for row in targets]
                
                return sample.data, targets
            # else:
            # loading in transformed data and targets from target transform
            data, targets = self.file_handler.static_load(self.full_root, idx)

            return data, targets

        else:
            raise IndexError(f"Index {idx} is out of bounds. Must be [0, {self.__len__()}]")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.full_root}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"impairment_level={self.impairment_level}, "
            f"transforms={self.transforms.__repr__()}, "
            f"target_transforms={self.target_transforms.__repr__()}, "
            f"file_handler_class={self.file_handler}, "
            f"train={self.train})"
        )

