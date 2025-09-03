"""Dataset Base Classes for creation and static loading.
"""

from __future__ import annotations

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.dataset_utils import (
    to_dataset_metadata, 
    frequency_shift_signal
)
from torchsig.signals.signal_types import Signal, SignalMetadataExternal
from torchsig.signals.builder import SignalBuilder
import torchsig.signals.builders as signal_builders
from torchsig.transforms.base_transforms import Transform
from torchsig.utils.random import Seedable
from torchsig.utils.dsp import compute_spectrogram
from torchsig.utils.printing import generate_repr_str
from torchsig.utils.verify import verify_transforms
from torchsig.utils.file_handlers.hdf5 import HDF5Reader as DEFAULT_READER
from torchsig.utils.coordinate_system import (
    Coordinate,
    Rectangle,
    is_rectangle_overlap
)

# Third Party
from torch.utils.data import Dataset, IterableDataset
import numpy as np

# Built-In
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import warnings
from pathlib import Path


if TYPE_CHECKING:
    from torchsig.utils.file_handlers import BaseFileHandler, ExternalFileHandler

def apply_transforms_and_labels_to_signal(sample, transforms, target_labels, num_signals_max = None):
    # apply user transforms
    for transform in transforms:
        sample = transform(sample)

    # apply metadata transforms
    # just return data if target_labels is None or empty list
    if target_labels is None:
        # return Signal object
        return sample
    if len(target_labels) < 1:
        # just return np.ndarray data
        return sample.data
    metadatas = sample.get_full_metadata()
    targets = []
    if len(target_labels) == 1:
        # just 1 target label
        # set targets to single item
        # verify metadatas have target_label
        #for metadata in metadatas:
        #    if not hasattr(metadata, target_labels[0]):
        #        raise AttributeError(f"Metadata does not have target label {target_labels[0]}: {metadata}")
        # apply target label
        targets = [getattr(metadata, target_labels[0]) for metadata in metadatas]
    else:
        # multiple target labels
        for metadata in metadatas:
            # for each signal metadata
            # apply all target labels
            #for target_label in target_labels:
            #    # make sure metadata has target label
            #    if not hasattr(metadata, target_label):
            #        raise AttributeError(f"Metadata does not have target label {target_label}: {metadata}")
            # apply target_label
            targets += [[getattr(metadata, target_label) for target_label in target_labels]]

    if num_signals_max == 1 and isinstance(targets, list) and len(targets) > 0:
        targets = targets[0]

    return sample.data, targets

class TorchSigIterableDataset(IterableDataset, Seedable):
    """
    This base class provides the functionality to generate signals and write them to disk if necessary. The dataset will continue 
    to generate samples infinitely.
    """ 
    # pylint: disable=abstract-method
    
    def __init__(
        self, 
        dataset_metadata: DatasetMetadata | str | dict,
        component_transforms: list = [],
        transforms: list = [],
        target_labels: list = None,
        **kwargs
    ):
        """
        Initializes the dataset, creates signal builders, and prepares file handlers based on metadata.

        Args:
            dataset_metadata (DatasetMetadata | str | dict): The dataset metadata.
            **kwargs: Additional keyword arguments for initialization.

        """
        Seedable.__init__(self, **kwargs)
        self.transforms = transforms
        for transform in self.transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)
        self.component_transforms = component_transforms
        for component_transform in self.component_transforms:
            if isinstance(component_transform, Seedable):
                component_transform.add_parent(self)
        self.dataset_metadata: DatasetMetadata = to_dataset_metadata(dataset_metadata)
        self.builders: Dict[str, SignalBuilder] = self._initialize_builders() # initialize builders
        self.target_labels = target_labels


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
        return apply_transforms_and_labels_to_signal(sample, self.transforms, self.target_labels, num_signals_max=self.dataset_metadata.num_signals_max)
    
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
            matching_classes = set(self.dataset_metadata.class_list) & set(builder.supported_classes)
            if len(matching_classes) > 0: # yes
                for c in matching_classes:
                    # add builder
                    builders[c] = builder(self.dataset_metadata, c,)
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

        transforms_str = [f"{t}" for t in self.transforms]

        return (
            f"\n{'-' * center_width} {self.__class__.__name__} {'-' * center_width}\n"
            f"\nTransforms\n"
            f"{'-' * max_width}\n"
            f"{list(transforms_str)}\n"
            f"\nTarget Labels = {self.target_labels}\n"
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


    def __generate_new_signal__(self) -> Signal:
        """Generates a new dataset signal/sample.

        Returns:
            DatasetSignal: A new generated dataset signal containing the data and metadata.
        """     
        
        # build noise floor
        iq_samples = self._build_noise_floor()

        # empty signal list initialization
        signals = []

        # determine number of signals in sample
        num_signals_to_generate = self.random_generator.integers(low=self.dataset_metadata.num_signals_min, high = self.dataset_metadata.num_signals_max+1)

        # list of rectangles representing the individual signals within the dataset IQ
        signal_rectangle_list = []

        # counter to avoid stuck in infinite loop
        infinite_loop_counter = 0
        infinite_loop_counter_max = 10*num_signals_to_generate

        # generate individual bursts
        num_signals_created = 0
        while (num_signals_created < num_signals_to_generate and infinite_loop_counter < infinite_loop_counter_max):

            # increment fail-safe counter
            infinite_loop_counter += 1

            # choose random signal
            class_name = self._random_signal_class()

            # get builder for signal class
            builder = self.builders[class_name]

            # generate signal at complex baseband
            new_signal = builder.build()

            # apply signal transforms
            for component_transform in self.component_transforms:
                new_signal = component_transform(new_signal)

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

            # map the signal bounding box into a rectangle in cartesian coordinate system
            new_rectangle = self._map_to_coordinates(new_signal)

            # check if the new_rectangle overlaps with any others in spectrogram
            has_overlap = self._check_if_overlap ( new_rectangle, signal_rectangle_list )

            # signal is used if there is no overlap OR with some random chance
            if (has_overlap is False or self.random_generator.uniform(0,1) < self.dataset_metadata.cochannel_overlap_probability):
                num_signals_created += 1
                # store the rectangle for future overlap checking
                signal_rectangle_list.append( new_rectangle )
                # place signal on iq sample cut
                iq_samples[new_signal.metadata.start_in_samples:new_signal.metadata.stop_in_samples] += new_signal.data
                # append the signal on the list
                signals.append(new_signal)
            # else:
            #     loop back to top and attempt to recreate another signal
        
        # form the sample (dataset object)
        sample = Signal(data=iq_samples, component_signals=signals)

        return sample

    def _map_to_coordinates ( self, new_signal:Signal ) -> Rectangle:

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

        return new_rectangle

    def _check_if_overlap ( self, new_rectangle:Rectangle, signal_rectangle_list:list ) -> bool:

        # initialize the boolean value which determines if there is overlap or not
        has_overlap = False

        # determine if overlap
        if len(signal_rectangle_list) > 0:
            # check to see if the current rectangle overlaps with any signals currently
            # in the spectrogram
            for reference_box in signal_rectangle_list:
                # check for invidivual overlap
                individual_overlap = is_rectangle_overlap(new_rectangle,reference_box)
                # combine with previous potential overlap checks
                has_overlap = has_overlap or individual_overlap

        return has_overlap

    # Functions

    def _random_signal_class(self):     
        """Randomly selects which signal to create next.

        Returns:
            str: A signal class name from the available signal classes.
        """
        return self.random_generator.choice(self.dataset_metadata.class_list, p=self.dataset_metadata.class_distribution)





class StaticTorchSigDataset(Dataset, Seedable):
    """Static Dataset class, which loads pre-generated data from a directory.
    
    Args:
        root (str): The root directory where the dataset is stored.
        transforms (list, optional): Transforms to apply to the data (default: []).
        file_handler_class (BaseFileHandler, optional): Class used for reading the dataset (default: HDF5FileHandler).
    """   

    def __init__(
        self,
        root: str,
        file_handler_class: BaseFileHandler = DEFAULT_READER,
        transforms: list = [],
        target_labels: list = None,
        **kwargs
    ):
        self.root = Path(root)
        self.reader = file_handler_class(root = self.root)

        Seedable.__init__(self, **kwargs)
        self.transforms = transforms
        for transform in self.transforms:
            transform.add_parent(self)
        self.target_labels = target_labels

        # dataset size
        self.dataset_length = len(self.reader)

        self.dataset_metadata = self.reader.dataset_metadata

        self._verify()

    def _verify(self):
        # check root

        if not self.root.exists():
            raise ValueError(f"root does not exist: {self.root}")


    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """Retrieves a sample from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, Tuple]file_handler: The data and targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if 0 <= idx < len(self):
            sample = self.reader.read(idx=idx)
            return apply_transforms_and_labels_to_signal(sample, self.transforms, self.target_labels, num_signals_max=self.dataset_metadata.num_signals_max)
        
        raise IndexError(f"Index {idx} is out of bounds. Must be [0, {self.__len__() - 1}]")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.root}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"file_handler_class={self.reader}"
        )





class ExternalTorchSigDataset(Dataset):
    """
    Lightweight static dataset for importing external (not TorchSig generated) data and metadata from files.
    
    Args:
        root (str): The root directory where the dataset is stored.
        file_handler_class (ExternalFileHandler): Class used for reading dataset.
        transforms (list, optional): Transforms to apply to the data (default: []).
        target_transforms (list, optional): Target transforms to apply (default: []).        
        
    """
    def __init__(
        self, 
        file_handler: ExternalFileHandler,
        transforms: List[Transform] = [],  
        target_labels: List[str] = []             
    ):
        self.transforms = transforms
        self.target_labels = target_labels
        self.file_handler = file_handler
        self.dataset_length = self.file_handler.size()
        self.dataset_metadata = self.file_handler.load_dataset_metadata()
        self._verify()

    
    def _verify(self):
        # Transforms
        self.transforms = verify_transforms(self.transforms)   

    
    def __len__(self) -> int: 
        return self.dataset_length
            
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        """
        Retrieves a sample from the static dataset by index.

        Args:
            idx: sample index.

        Returns:
            Tuple[data, targets] returned data array and metadata.
        """
        if 0 <= idx < len(self):
            data, signal_metadatas = self.file_handler.load(idx)
            component_signals = []
            for signal_metadata in signal_metadatas:
                if not isinstance(signal_metadata, dict):
                    raise ValueError(f"Signal metadata is not a dict: {type(signal_metadata)}.")
                # create external signal metadata
                esm = SignalMetadataExternal(
                    self.dataset_metadata,
                    **signal_metadata
                )
                # create component signal
                component_signal = Signal(
                    data = np.array([]),
                    metadata = esm,
                )
                # add to component signals
                component_signals.append(component_signal)
            
            # create Signal from component signals
            sample = Signal(
                data = data,
                component_signals = component_signals
            )

            # apply user transforms
            for transform in self.transforms:
                sample = transform(sample)

            # apply metadata transforms
            # just return data if target_labels is None or empty list
            if self.target_labels is None:
                return sample
            if len(self.target_labels) < 1:
                return sample.data

            metadatas = sample.get_full_metadata()
            targets = []
            if len(self.target_labels) == 1:
                # just 1 target label
                # set targets to single item
                targets = [getattr(metadata, self.target_labels[0]) for metadata in metadatas]
            else:
                # multiple target labels
                for metadata in metadatas:
                    # for each signal metadata
                    # apply all target labels
                    targets += [[getattr(metadata, target_label) for target_label in self.target_labels]]

            return sample.data, targets
            
        raise IndexError(f"Index {idx} is out of bounds. Must be [0, {self.__len__()}]")          