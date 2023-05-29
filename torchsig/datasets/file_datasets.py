import os
import xml
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Optional, Any

from torchsig.utils.types import SignalDescription
from torchsig.datasets.wideband import BurstSourceDataset, SignalBurst
from torchsig.transforms.functional import (
    to_distribution, 
    uniform_continuous_distribution, 
    uniform_discrete_distribution,
    FloatParameter, 
    NumericParameter,
)


class TargetInterpreter:
    """The TargetInterpreter base class is meant to be inherited and modified 
    for specific interpreters such that each sub-class implements a transform
    from a file containing target information into a BurstCollection containing
    SignalBursts.
    
    Args:
        target_file: (:obj:`str`):
            The file containing label/target/annotation information

        num_iq_samples: (:obj:`int`):
            The number of IQ samples for each example in the dataset being
            generated

        capture_duration_samples: (:obj:`int`):
            The total number of IQ samples in the original capture file
            
        class_list: (:obj:`list`):
            List of class names for class to binary encoding

    """
    def __init__(
        self, 
        target_file: str,
        class_list: List[str],
        num_iq_samples: int = int(512*512),
        capture_duration_samples: int = int(512*512),
    ):
        self.target_file = target_file
        self.num_iq_samples = num_iq_samples
        self.capture_duration_samples = capture_duration_samples
        self.class_list = class_list
        # Initialize relevant capture parameters to be overwritten by interpreter
        self.sample_rate = 1.0
        self.is_complex = True
        # Initialize the detections dataframe using sub-class's interpreters
        self.detections_df = self._convert_to_dataframe()
        self.detections_df.sort_values(by=['start'])
        self.num_labels = len(self.detections_df)
        self.detections_df = self._convert_class_name_to_index()

    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Meant to be implemented by a sub-class specific to the label file,
        converting labels into a dataframe with the following columns for a 
        uniform setup prior to generalized burst conversions.
        
        """
        detection_columns = ["start", "stop", "center_freq", "bandwidth", "class_name"]
        self.detections_df = pd.DataFrame(columns = detection_columns)
        return self.detections_df
    
    def _convert_class_name_to_index(self) -> pd.DataFrame:
        """Append a column to the dataframe containing the class index 
        associated with the class names
        
        """
        # If no input class_list ordering provided, read from dataframe
        if self.class_list == []:
            self.class_list = list(self.detections_df['class_name'].unique())
        # Append class index column
        self.detections_df['class_index'] = [self.class_list.index(self.detections_df['class_name'][i]) for i in range(self.num_labels)]
        return self.detections_df
        
    def convert_to_signalburst(
        self, 
        start_sample: int = 0, 
        df_indicies: Optional[np.ndarray] = None, 
    ) -> List[WidebandFileSignalBurst]:
        """Inputs a start sample and an array of indicies to convert into a 
        list of `SignalBursts` for the `WidebandFileSignalBurst`s
        
        """
        signal_bursts = []
        if df_indicies is None:
            # Defaults to full dataframe
            df_indicies = np.arange(self.num_labels)
                
        for label in self.detections_df.iloc[df_indicies].itertuples():
            # Determine cut vs full capture relationship
            startInWindow = bool(label.start >= start_sample and label.start < start_sample + self.num_iq_samples)
            stopInWindow = bool(label.stop > start_sample and label.stop <= start_sample + self.num_iq_samples)
            spansFullWindow = bool(label.start <= start_sample and label.stop >= start_sample + self.num_iq_samples)
            fullyContainedInWindow = bool(startInWindow and stopInWindow)

            # Normalize freq information
            center_freq = label.center_freq / self.sample_rate
            center_freq = center_freq if self.is_complex else center_freq / 2
            bandwidth = label.bandwidth / self.sample_rate
            bandwidth = bandwidth if self.is_complex else bandwidth / 2
            
            # If label present, normalize with respect to requested window and append
            if fullyContainedInWindow:
                start = (label.start - start_sample) / self.num_iq_samples
                stop = (label.stop - start_sample) / self.num_iq_samples
                signal_bursts.append(
                    WidebandFileSignalBurst(
                        num_iq_samples = self.num_iq_samples,
                        start = start,
                        stop = stop,
                        center_frequency = center_freq,
                        bandwidth = bandwidth,
                        class_name = label.class_name,
                        class_index = label.class_index,
                        random_generator = np.random.RandomState,
                    )
                )
                
            elif startInWindow:
                start = (label.start - start_sample) / self.num_iq_samples
                stop = 1
                signal_bursts.append(
                    WidebandFileSignalBurst(
                        num_iq_samples = self.num_iq_samples,
                        start = start,
                        stop = stop,
                        center_frequency = center_freq,
                        bandwidth = bandwidth,
                        class_name = label.class_name,
                        class_index = label.class_index,
                        random_generator = np.random.RandomState,
                    )
                )
                
            elif stopInWindow:
                start = 0
                stop = (label.stop - start_sample) / self.num_iq_samples
                signal_bursts.append(
                    WidebandFileSignalBurst(
                        num_iq_samples = self.num_iq_samples,
                        start = start,
                        stop = stop,
                        center_frequency = center_freq,
                        bandwidth = bandwidth,
                        class_name = label.class_name,
                        class_index = label.class_index,
                        random_generator = np.random.RandomState,
                    )
                )
                
            elif spansFullWindow:
                start = 0
                stop = 1
                signal_bursts.append(
                    WidebandFileSignalBurst(
                        num_iq_samples = self.num_iq_samples,
                        start = start,
                        stop = stop,
                        center_frequency = center_freq,
                        bandwidth = bandwidth,
                        class_name = label.class_name,
                        class_index = label.class_index,
                        random_generator = np.random.RandomState,
                    )
                )
        return signal_bursts


class CSVFileInterpreter(TargetInterpreter):
    """The CSVFileInterpreter implements the transformation from a CSV-
    formatted signal annotation into a List of WidebandFileSignalBursts. Expected 
    input is a CSV file where each row contains a separate signal annotation, 
    and the annotation details are separated by commas for each column. 
    Information about how the CSV was generated and the original capture file 
    are passed in such that the normalized labels for the requested dataset can
    be calculated
    
    Example CSV format:
        ```
        index,start_sample,stop_sample,center_freq,bandwidth,class_name
        0,200,14000,800000000,5000000,signal_0
        1,1300,4000,425000000,1000000,signal_1
        ```
    Input args:
        start_column=1
        stop_column=2
        center_freq_column=3
        bandwidth_column=4
        class_column=5
    
    Args:
        target_file: (:obj:`str`):
            The file containing label/target/annotation information
        
        num_iq_samples: (:obj:`int`):
            Number of IQ samples being requested at the TorchSig SignalDataset side
        
        capture_duration_samples: (:obj:`int`):
            Total number of IQ samples present in the original data file

        class_list: (:obj:`list`):
            List of class names for class to binary encoding

        sample_rate: (:obj:`float`):
            Sample rate of data capture
            
        is_complex: (:obj:`bool`):
            Specify whether data capture is complex or real
        
        start_column: (:obj:`int`):
            Column index for start sample
            
        stop_column: (:obj:`int`):
            Column index for stop sample
        
        center_freq_column: (:obj:`int`):
            Column index for center frequency in Hz
        
        bandwidth_column: (:obj:`int`):
            Column index for bandwidth in Hz
            
        class_column: (:obj:`int`):
            Column index for class name
        
    """
    def __init__(
        self, 
        target_file: str,
        class_list: List[str],
        num_iq_samples: int = int(512*512),
        capture_duration_samples: int = int(512*512),
        sample_rate: float = 25e6,
        is_complex: bool = True,
        start_column: int = 1,
        stop_column: int = 2,
        center_freq_column: int = 3,
        bandwidth_column: int = 4,
        class_column: int = 5,
        **kwargs
    ):
        self.target_file = target_file
        self.num_iq_samples = num_iq_samples
        self.capture_duration_samples = capture_duration_samples
        self.class_list = class_list
        self.sample_rate = sample_rate
        self.is_complex = is_complex
        self.start_column = start_column
        self.stop_column = stop_column
        self.center_freq_column = center_freq_column
        self.bandwidth_column = bandwidth_column
        self.class_column = class_column
        # Generate dataframe
        self.detections_df = self._convert_to_dataframe()
        self.detections_df = self.detections_df.sort_values(by=['start']).reset_index(drop=True)
        self.num_labels = len(self.detections_df)
        self.detections_df = self._convert_class_name_to_index()
        
    def _convert_to_dataframe(self) -> pd.DataFrame:
        # Initialize dataframe
        detection_columns = ["start", "stop", "center_freq", "bandwidth", "class_name"]
        self.detections_df = pd.DataFrame(columns = detection_columns)
        
        # Read CSV into temporary dataframe
        df = pd.read_csv(self.target_file)
        
        # Store information into detections dataframe
        self.detections_df["class_name"] = df.iloc[:,self.class_column].tolist()
        self.detections_df["class_indices"] = [self.class_list.index(n) for n in self.detections_df["class_name"]]
        self.detections_df["start"] = df.iloc[:,self.start_column].tolist()
        self.detections_df["stop"] = df.iloc[:,self.stop_column].tolist()
        self.detections_df["center_freq"] = df.iloc[:,self.center_freq_column].tolist()
        self.detections_df["bandwidth"] = df.iloc[:,self.bandwidth_column].tolist()
        
        return self.detections_df
    
    
class SigMFInterpreter(TargetInterpreter):
    """The SigMFInterpreter reads in SigMF meta file information and maps the 
    annotations into SignalBursts
    
    Args:
        target_file: (:obj:`str`):
            The file containing label/target/annotation information
        
        num_iq_samples: (:obj:`int`):
            Number of IQ samples being requested at the TorchSig SignalDataset side
        
        capture_duration_samples: (:obj:`int`):
            Total number of IQ samples present in the original data file

        class_list: (:obj:`list`):
            List of class names for class to binary encoding

        class_target: (:obj:`str`):
            Annotation label for the field containing the class name
        
    """
    def __init__(
        self, 
        target_file: str,
        class_list: List[str],
        num_iq_samples: int = int(512*512),
        capture_duration_samples: int = int(512*512),
        class_target: str = 'core:description',
        **kwargs
    ):
        self.target_file = target_file
        self.num_iq_samples = num_iq_samples
        self.capture_duration_samples = capture_duration_samples
        self.class_list = class_list
        self.class_target = class_target
        # Generate dataframe
        self.detections_df = self._convert_to_dataframe()
        self.detections_df = self.detections_df.sort_values(by=['start']).reset_index(drop=True)
        self.num_labels = len(self.detections_df)
        self.detections_df = self._convert_class_name_to_index()
        
    def _convert_to_dataframe(self) -> pd.DataFrame:
        # Initialize dataframe
        detection_columns = ["start", "stop", "center_freq", "bandwidth", "class_name", "class_index"]
        self.detections_df = pd.DataFrame(columns = detection_columns)
        
        # Read SigMF meta file
        meta = json.load(open(self.target_file, "r"))
        
        # Read global SigMF data
        self.sample_rate = int(meta["global"]["core:sample_rate"])
        data_type = meta["global"]["core:datatype"]
        self.is_complex = True if "c" in data_type else False
        capture_center_freq = float(meta["captures"][0]["core:frequency"])
        
        # Loop through annotations
        class_names = []
        class_indices = []
        starts = []
        stops = []
        center_freqs = []
        bandwidths = []
        for annotation_idx, annotation in enumerate(meta['annotations']):
            # Read annotation details
            class_names.append(annotation[self.class_target])
            class_indices.append(self.class_list.index(annotation[self.class_target]))
            lower_freq = annotation['core:freq_lower_edge']
            upper_freq = annotation['core:freq_upper_edge']
            bandwidth = upper_freq - lower_freq
            bandwidths.append(bandwidth)
            center_freqs.append(lower_freq - capture_center_freq + bandwidth/2)
            start = annotation['core:sample_start']
            starts.append(start)
            stops.append(start + annotation['core:sample_count'])

        # Store information into detections dataframe
        self.detections_df["class_name"] = class_names
        self.detections_df["class_index"] = class_indices
        self.detections_df["start"] = starts
        self.detections_df["stop"] = stops
        self.detections_df["center_freq"] = center_freqs
        self.detections_df["bandwidth"] = bandwidths
        
        return self.detections_df
    
    
class WidebandFileSignalBurst(SignalBurst):
    """A sub-class of SignalBurst that takes a wideband file input along with 
    signal annotation parameters and reads the specified data from the file
    
    Args:
        data_file (:obj:`str`):
            The file containing the IQ data to read from
            
        start_sample (:obj:`int`):
            The IQ sample to start reading from within the IQ data file
            
        is_complex (:obj:`bool`):
            Boolean specifying if the data file contains complex data (True)
            or real data (False)
            
        capture_type (:obj:`numpy.dtype`):
            The precision of the data capture. Defaults to int16
            
        **kwargs
    """

    def __init__(
        self, 
        data_file: Optional[str] = None,
        start_sample: int = 0,
        is_complex: bool = True,
        capture_type: np.dtype = np.dtype(np.int16),
        **kwargs
    ):
        super(WidebandFileSignalBurst, self).__init__(**kwargs)
        assert self.center_frequency is not None
        assert self.bandwidth is not None
        self.lower_frequency = self.center_frequency - self.bandwidth / 2
        self.upper_frequency = self.center_frequency + self.bandwidth / 2
        self.data_file = data_file
        self.start_sample = start_sample
        self.is_complex = is_complex
        self.capture_type = capture_type        
        capture_type_is_complex = 'complex' in str(self.capture_type)
        if self.is_complex and not capture_type_is_complex:
            self.bytes_per_sample = int(self.capture_type.itemsize * 2)
        else:
            self.bytes_per_sample = self.capture_type.itemsize
            
    def generate_iq(self):
        if self.data_file is not None:
            with open(self.data_file, "rb") as file_object:
                # Apply desired offset
                file_object.seek(int(self.start_sample)*self.bytes_per_sample)
                # Read desired number of samples from file
                iq_data = np.frombuffer(
                    file_object.read(int(self.num_iq_samples)*self.bytes_per_sample), 
                    dtype=self.capture_type
                ).astype(np.float64).view(np.complex128)
        else:
            # Since only the first burst is given data information, the 
            # remaining bursts are set to all 0's to avoid reading the data
            # file repetitively and summing with itself
            iq_data = np.zeros(self.num_iq_samples, dtype=np.complex128)
        return iq_data[:self.num_iq_samples]
    
    
class FileBurstSourceDataset(BurstSourceDataset):
    """The FileBurstSourceDataset complements the SyntheticBurstSourceDataset 
    but rather than generating synthetic bursts and adding them together, the 
    FileBurstSourceDataset inputs information from files and returns labeled
    SignalBursts for the capture files. The conversions from the label files to the
    SignalBursts is done through an input TargetInterpreter such that the
    FileBurstSourceDataset can be used with any data type coming from files
    provided the interpretation class is built.
    
    Args:
        data_files: (:obj:`List`):
            List of data files to read the IQ data from
        
        target_files: (:obj:`List`):
            List of target files to read the signal annotations from. Note that
            these files should be ordered to match the data_files accordingly
        
        capture_type: (:obj:`np.dtype`):
            Specify the data type of the capture data_files (ex: np.int16)
        
        is_complex: (:obj:`bool`):
            Specify whether the data files are complex or real
            
        sample_policy: (:obj:`str`):
            Specify the policy defining how samples are retrieved from the data
            and annotation files. Options include: `random`, `sequential_label`,
            and `sequential_iq`. Details for each below:
                - `random_labels`: Randomly sample files and then labels and then read
                    IQ samples around the randomly sampled label
                - `sequential_labels`: Sequentially iterate over the files and 
                    labels, retrieving IQ samples around each sequential label
                - `random_iq`: Randomly sample files and starting IQ samples,
                    regardless of labels
                - `sequential_iq`: Sequentially iterate over the files, 
                    directly iterating over IQ samples regardless of labels
            
        null_ratio: (:obj:`float`):
            Selects the ratio of examples without labeled bursts present. Only 
            valid for the `random_labels` and `sequential_labels` sample 
            policies. For example, a ratio of 0.2 would have 0.2*num_samples
            examples without bursts (noise-only) and 0.8*samples containing 
            labeled bursts
            
        target_interpreter: (:obj:`TargetInterpreter`):
            TargetInterpreter class that maps teh target_files' annotations
            into a BurstCollection of FileSignalBursts

        class_list: (:obj:`list`):
            List of class names for class to binary encoding

        num_iq_samples: (:obj:`int`):
            Number of IQ samples for each example in the dataset
            
        num_samples: (:obj:`int`):
            Number of samples/examples to read for creating the dataset
            
        seed: (:obj:`Optional`):
            Initialize the random seed
        
    """

    def __init__(
        self,
        data_files: List[str],
        target_files: List[str],
        class_list: List[str],
        capture_type: np.dtype = np.dtype(np.int16),
        is_complex: bool = True,
        sample_policy: str = "random_labels",
        null_ratio: float = 0.0,
        target_interpreter: TargetInterpreter = SigMFInterpreter,  # type: ignore
        num_iq_samples: int = int(512*512),
        num_samples: int = 100,
        seed: Optional[int] = None,
        **kwargs
    ):
        super(FileBurstSourceDataset, self).__init__(
            num_iq_samples=num_iq_samples,
            num_samples=num_samples,
        )
        self.data_files = data_files
        self.target_files = target_files
        self.capture_type = capture_type
        self.is_complex = is_complex
        self.sample_policy = sample_policy
        self.null_ratio = null_ratio
        self.target_interpreter = target_interpreter
        self.class_list = class_list
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples
        self.seed = seed
        
        capture_type_is_complex = 'complex' in str(self.capture_type)
        if self.is_complex and not capture_type_is_complex:
            self.bytes_per_sample = int(self.capture_type.itemsize * 2)
        else:
            self.bytes_per_sample = self.capture_type.itemsize

        if "labels" in self.sample_policy:
            # Set number of samples containing no bursts
            self.num_null_samples = int(self.num_samples * self.null_ratio)
            self.num_valid_samples = self.num_samples - self.num_null_samples
            
            # Distribute randomness evenly over labels, rather than files then labels
            # If more than 10,000 files, omit this step for speed
            if self.sample_policy == "random_labels" and len(self.target_files) < 10_000:
                annotations_per_file = []
                for file_index, target_file in enumerate(self.target_files):
                    # Read total file size
                    capture_duration_samples = os.path.getsize(
                        os.path.join(self.data_files[file_index])
                    ) // self.bytes_per_sample
                    # Interpret annotations for file
                    interpreter = self.target_interpreter(  # type: ignore
                        target_file = target_file,
                        num_iq_samples = self.num_iq_samples,
                        capture_duration_samples = capture_duration_samples,
                        class_list = self.class_list,
                    )
                    # Read all annotations
                    annotations = interpreter.detections_df
                    # Track number of annotations
                    annotations_per_file.append(len(annotations))
                total_annotations = sum(annotations_per_file)
                self.file_probabilities = np.asarray(annotations_per_file) / total_annotations
        
        # Generate the index by creating a set of bursts.
        self.index = [(collection, idx) for idx, collection in enumerate(self._generate_burst_collections())]

    def _generate_burst_collections(self) -> List[List[SignalBurst]]:
        dataset = []
        
        if "iq" in self.sample_policy:
            file_index = 0
            data_index = 0
            for sample_idx in range(self.num_samples):
                if self.sample_policy == "random_iq":
                    # Sample random file
                    file_index = np.random.randint(len(self.data_files))
                # Read total file size
                capture_duration_samples = os.path.getsize(
                    os.path.join(self.data_files[file_index])
                ) // self.bytes_per_sample

                # Instantiate target interpreter
                interpreter = self.target_interpreter(  # type: ignore
                    target_file=self.target_files[file_index],
                    num_iq_samples = self.num_iq_samples,
                    capture_duration_samples = capture_duration_samples,
                    class_list = self.class_list,
                )
                # Read all annotations
                annotations = interpreter.detections_df
                
                # Determine data start index
                if self.sample_policy == "random_iq":
                    if (capture_duration_samples - self.num_iq_samples) > 0:
                        data_index = np.random.randint(0, capture_duration_samples - self.num_iq_samples)

                # Convert labels to SignalBursts
                sample_burst_collection = interpreter.convert_to_signalburst(
                    start_sample=data_index, 
                    df_indicies=None,
                )

                if len(sample_burst_collection) > 0:
                    # Add data file information to the first SignalBurst only
                    sample_burst_collection[0].data_file = self.data_files[file_index]
                    sample_burst_collection[0].start_sample = data_index
                    sample_burst_collection[0].is_complex = self.is_complex
                    sample_burst_collection[0].capture_type = self.capture_type
                    capture_type_is_complex = 'complex' in str(self.capture_type)
                    if self.is_complex and not capture_type_is_complex:
                        sample_burst_collection[0].bytes_per_sample = int(self.capture_type.itemsize * 2)
                    else:
                        sample_burst_collection[0].bytes_per_sample = self.capture_type.itemsize
                else:
                    # Create invalid SignalBurst for data file information only
                    sample_burst_collection = []
                    sample_burst_collection.append(
                        WidebandFileSignalBurst(
                            num_iq_samples=self.num_iq_samples,
                            start=0,
                            stop=0,
                            center_frequency=0,
                            bandwidth=0,
                            class_name=None,
                            class_index=None,
                            data_file=self.data_files[file_index],
                            start_sample=data_index,
                            is_complex=self.is_complex,
                            capture_type=self.capture_type,
                            random_generator=np.random.RandomState,
                        )
                    )
        
                # If sequentially sampling, increment
                if self.sample_policy == "sequential_iq":
                    data_index += self.num_iq_samples
                    # Check for end of lables and end of files
                    if (data_index + self.num_iq_samples) > capture_duration_samples:
                        data_index = 0
                        file_index += 1
                        if file_index >= len(self.data_files):
                            file_index = 0

                # Save SignalBursts to dataset
                dataset.append(sample_burst_collection)
            
        else:
            # First, handle null samples
            null_fail_counter = 0
            for sample_idx in range(self.num_null_samples):
                # Sample random file
                file_index = np.random.randint(len(self.data_files))
                # Read total file size
                capture_duration_samples = os.path.getsize(
                    os.path.join(self.data_files[file_index])
                ) // self.bytes_per_sample

                # Instantiate target interpreter
                interpreter = self.target_interpreter(  # type: ignore
                    target_file=self.target_files[file_index],
                    num_iq_samples=self.num_iq_samples,
                    capture_duration_samples=capture_duration_samples,
                    class_list = self.class_list,
                )
                # Read all annotations
                annotations = interpreter.detections_df

                # Decide data_index based on annotation locations
                null_interval = 0
                null_start_index = 0
                null_attempts = 0
                null_fail = False
                while null_interval < self.num_iq_samples:
                    # Randomly sample label index to search around
                    label_index = np.random.randint(interpreter.num_labels)
                    if interpreter.num_labels > 1 and label_index+1 <= interpreter.num_labels-1:
                        # Max over previous annotation stop and previous null start to handle cases of long signals
                        null_start_index = max(annotations.iloc[label_index].stop, null_start_index)
                        null_stop_index = annotations.iloc[label_index+1].start
                    elif interpreter.num_labels > 1 and label_index + 1 > interpreter.num_labels-1:
                        # Start start index at end of final label
                        null_start_index = max(annotations.iloc[label_index].stop, null_start_index)
                        null_stop_index = capture_duration_samples
                    elif interpreter.num_labels == 1:
                        # Sample from before or after the only label
                        before = True if np.random.rand() >= 0.5 else False
                        null_start_index = 0 if before else annotations.iloc[0].stop
                        null_stop_index = annotations.iloc[0].start if before else capture_duration_samples
                    else:
                        # Sample from anywhere in file
                        null_start_index = 0
                        null_stop_index = capture_duration_samples
                    null_interval = null_stop_index - null_start_index
                    null_attempts += 1
                    if null_attempts > 100:
                        null_fail = True
                        break
                if null_fail:
                    sample_idx -= 1
                    null_fail_counter += 1
                    if null_fail_counter > 100:
                        # Not enough null examples across files
                        self.num_valid_samples = self.num_samples - sample_idx
                        break
                    continue

                # Random value within null start and stop indicies - IQ samples
                data_index = np.random.randint(
                    null_start_index, 
                    null_stop_index-self.num_iq_samples
                )

                # Create invalid SignalBurst for data file information only
                null_sample_burst_collection = []
                null_sample_burst_collection.append(
                    WidebandFileSignalBurst(
                        num_iq_samples=self.num_iq_samples,
                        start=0,
                        stop=0,
                        center_frequency=0,
                        bandwidth=0,
                        class_name=None,
                        class_index=None,
                        data_file=self.data_files[file_index],
                        start_sample=data_index,
                        is_complex=self.is_complex,
                        capture_type=self.capture_type,
                        random_generator=np.random.RandomState,
                    )
                )

                # Append to dataset
                dataset.append(null_sample_burst_collection)

            # Next, handle the valid bursts
            file_index = 0
            label_index = 0
            for sample_idx in range(self.num_valid_samples):
                if self.sample_policy == "random_labels":
                    # Sample random file, weighted by number of annotations
                    file_index = np.random.choice(len(self.data_files), p=self.file_probabilities)
                # Read total file size
                capture_duration_samples = os.path.getsize(
                    os.path.join(self.data_files[file_index])
                ) // self.bytes_per_sample

                # Instantiate target interpreter
                interpreter = self.target_interpreter(  # type: ignore
                    target_file=self.target_files[file_index],
                    num_iq_samples = self.num_iq_samples,
                    capture_duration_samples = capture_duration_samples,
                    class_list = self.class_list,
                )
                # Read all annotations
                annotations = interpreter.detections_df

                if self.sample_policy == "random_labels":
                    # Randomly sample specific burst label
                    label_index = np.random.randint(interpreter.num_labels)
                burst_start_index = annotations.iloc[label_index].start

                # Step back a random number of IQ samples from the burst start index
                burst_duration = annotations.iloc[label_index].stop - burst_start_index
                if burst_duration < self.num_iq_samples:
                    if (burst_duration / self.num_iq_samples) <= 0.2:
                        # Very short burst: Ensure full burst is present in window
                        earliest_sample_index = burst_start_index - (self.num_iq_samples - burst_duration)
                        latest_sample_index = burst_start_index
                    else:
                        # Short burst: Ensure at least half of the burst is present in window
                        earliest_sample_index = burst_start_index - (self.num_iq_samples - burst_duration / 2)
                        latest_sample_index = burst_start_index + burst_duration / 2
                else:
                    # Long burst: Ensure at least a quarter of the window is occupied
                    earliest_sample_index = burst_start_index - (0.75 * self.num_iq_samples)
                    latest_sample_index = annotations.iloc[label_index].stop - (0.25 * self.num_iq_samples)
                data_index = max(0,np.random.randint(earliest_sample_index, latest_sample_index))

                # Check duration
                if capture_duration_samples - data_index < self.num_iq_samples:
                    sample_idx -= 1
                    continue

                # Convert labels to SignalBursts
                sample_burst_collection = interpreter.convert_to_signalburst(
                    start_sample=data_index, 
                    df_indicies=None,
                )

                # Add data file information to the first SignalBurst only
                sample_burst_collection[0].data_file = self.data_files[file_index]
                sample_burst_collection[0].start_sample = data_index
                sample_burst_collection[0].is_complex = self.is_complex
                sample_burst_collection[0].capture_type = self.capture_type
                capture_type_is_complex = 'complex' in str(self.capture_type)
                if self.is_complex and not capture_type_is_complex:
                    sample_burst_collection[0].bytes_per_sample = int(self.capture_type.itemsize * 2)
                else:
                    sample_burst_collection[0].bytes_per_sample = self.capture_type.itemsize

                # If sequentially sampling, increment
                if self.sample_policy == "sequential_labels":
                    label_index += len(sample_burst_collection)
                    # Check for end of lables and end of files
                    if label_index >= interpreter.num_labels:
                        label_index = 0
                        file_index += 1
                        if file_index >= len(self.data_files):
                            file_index = 0

                # Save SignalBursts to dataset
                dataset.append(sample_burst_collection)

        return dataset
