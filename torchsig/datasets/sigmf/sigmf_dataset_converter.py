
from pathlib import Path
from typing import Any, Dict, List, Literal
import warnings

import numpy as np
import sigmf
import yaml
from sigmf import SigMFFile

from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler
from torchsig.datasets.dataset_utils import dataset_yaml_name
from torchsig.utils.random import Seedable


class SigMFDatasetConverter(Seedable):

    def __init__(
        self,
        root: str,
        dataset: Literal["narrowband", "wideband"] = "wideband",
        overwrite: bool = False,
        fft_size: int = 512,
        num_iq_samples: int = 512**2,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        batch_size: int = 1,
        overlap_factor: float = 0.5,  # only for wideband conversion
        target_snr_db: float = 10.0,  # only for narrowband conversion
    ):
        Seedable.__init__(self)

        self.root = Path(root)

        self.path_converted = self.root / "torchsig"
        self.path_converted.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset
        self.overwrite = overwrite
        self.batch_size = batch_size

        self.fft_size = fft_size
        self.num_iq_samples = num_iq_samples
        self.overlap_factor = overlap_factor

        self.writer: TorchSigFileHandler = file_handler(
            root=str(self.path_converted),
            batch_size=batch_size,  # Single sample loading
        )

        self.sigmf_datasets = self._get_all_sigmf_datasets()
        self.label_mapping = self._create_label_mapping()
        self.sample_rate = self._get_dataset_sample_rate()
        self.target_snr_db = target_snr_db

        self.dataset_stats = {
            'num_signals_per_chunk': [],
            'signal_durations': [],
            'signal_durations_samples': [],
            'signal_bandwidths': [],
            'signal_center_freqs': [],
        }

    def convert(self) -> None:
        """
        Main method to convert SigMF files to TorchSig Zarr format.
        """
        if self.dataset == "narrowband":
            self._convert_narrowband()
        elif self.dataset == "wideband":
            self._convert_wideband()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset}")

    def _convert_narrowband(self) -> None:
        """Convert SigMF files to TorchSig narrowband format"""

        if not self.sigmf_datasets:
            raise FileNotFoundError(
                f"No SigMF datasets found in {self.root}. Please ensure the directory contains .sigmf-meta and .sigmf-data files.")

        if self.writer.exists() and not self.overwrite:
            print(
                f"Dataset already exists in {self.path_converted}. Skipping conversion.")
            return

        print(
            f"Converting {len(self.sigmf_datasets)} SigMF files to narrowband Zarr format...")

        # Create writer_info.yaml file
        self._create_writer_info()

        # Setup writer
        self.writer._setup()

        all_data = []
        all_targets = []
        batch_idx = 0
        total_annotations = 0

        for sigmf_base in self.sigmf_datasets:
            print(f"Processing {sigmf_base}...")

            # Load SigMF file
            sigmf_file = sigmf.sigmffile.fromfile(
                str(self.root / f"{sigmf_base}"))

            # Read the data
            samples = sigmf_file.read_samples()
            annotations = sigmf_file.get_annotations()
            sample_rate = sigmf_file.get_global_field(
                SigMFFile.SAMPLE_RATE_KEY)

            print(f"Found {len(annotations)} annotations")
            total_annotations += len(annotations)

            for ann_idx, ann in enumerate(annotations):
                # Extract annotation boundaries
                ann_start = ann.get('core:sample_start', 0)
                ann_count = ann.get('core:sample_count', self.num_iq_samples)

                # Skip if annotation goes beyond file
                if ann_start >= len(samples):
                    print(
                        f"Warning: Annotation {ann_idx} starts beyond file length, skipping")
                    continue

                # Get frequency information for filtering
                freq_lower = ann.get(SigMFFile.FLO_KEY)
                freq_upper = ann.get(SigMFFile.FHI_KEY)

                # Get capture info
                capture = self._get_capture_for_sample(sigmf_file, ann_start)
                if capture is None:
                    print(
                        f"Warning: No capture found for annotation at sample {ann_start}")
                    continue

                capture_center_freq = capture.get(SigMFFile.FREQUENCY_KEY, 0.0)

                # Extract signal data with frequency filtering
                signal_data = self._extract_narrowband_signal(
                    samples,
                    ann_start,
                    ann_count,
                    freq_lower=freq_lower,
                    freq_upper=freq_upper,
                    capture_center_freq=capture_center_freq,
                    sample_rate=sample_rate
                )

                if signal_data is None:
                    continue

                # Create single annotation for this narrowband sample
                torchsig_annotation = self._create_narrowband_annotation(
                    sigmf_file, ann, signal_data)

                if torchsig_annotation is None:
                    continue

                # Update statistics
                self._update_dataset_stats([torchsig_annotation])

                all_data.append(signal_data)
                # Single annotation in a list
                all_targets.append([torchsig_annotation])

                # Write batch when full
                if len(all_data) >= self.batch_size:
                    self.writer.write(batch_idx, (all_data, all_targets))
                    batch_idx += 1
                    all_data = []
                    all_targets = []

        # Write remaining data
        if all_data:
            self.writer.write(batch_idx, (all_data, all_targets))

        # Create YAML config
        self._create_yaml()

        total_samples = self.writer.size(str(self.path_converted))
        print(
            f"Conversion complete! Created {total_samples} narrowband samples from {total_annotations} annotations in {str(self.path_converted)}")

    def _extract_narrowband_signal(
        self,
        samples: np.ndarray,
        ann_start: int,
        ann_count: int,
        freq_lower: float = None,
        freq_upper: float = None,
        capture_center_freq: float = 0.0,
        sample_rate: float = None
    ) -> np.ndarray:
        """
        Extract and prepare narrowband signal data with optional frequency filtering.

        Args:
            samples: Full IQ sample array
            ann_start: Annotation start index
            ann_count: Annotation sample count
            freq_lower: Lower frequency bound (Hz) for filtering
            freq_upper: Upper frequency bound (Hz) for filtering
            capture_center_freq: Capture center frequency (Hz)
            sample_rate: Sample rate (Hz)

        Returns:
            Processed signal data of length num_iq_samples, or None if invalid
        """
        # Ensure we don't go beyond file boundaries
        ann_end = min(ann_start + ann_count, len(samples))
        actual_count = ann_end - ann_start

        if actual_count <= 0:
            return None

        # Extract the actual signal
        signal_segment = samples[ann_start:ann_end]

        if (freq_lower is not None and freq_upper is not None and
                sample_rate is not None):
            signal_segment = self._apply_frequency_filter(
                signal_segment,
                freq_lower,
                freq_upper,
                capture_center_freq,
                sample_rate
            )

        # Add the extracted signal to a realistic noise floor with SNR control
        final_signal = self._prepare_signal_with_noise_floor(signal_segment)

        return final_signal

    def _prepare_signal_with_noise_floor(self, signal_segment: np.ndarray) -> np.ndarray:
        """
        Takes a signal segment, places it within a sample of `num_iq_samples`
        (with padding or truncation), and adds it to a generated noise floor
        with a target SNR.

        Args:
            signal_segment (np.ndarray): The extracted and filtered signal.

        Returns:
            np.ndarray: The final signal of length `num_iq_samples` with noise.
        """
        # 1. Generate noise floor
        noise_floor = self._generate_noise_floor(self.num_iq_samples)
        noise_power = np.mean(np.abs(noise_floor)**2)

        # 2. Scale signal for target SNR
        signal_power = np.mean(np.abs(signal_segment)**2)
        if signal_power > 1e-12:  # Avoid division by zero
            target_snr_linear = 10**(self.target_snr_db / 10.0)

            required_signal_power = noise_power * target_snr_linear
            scaling_factor = np.sqrt(required_signal_power / signal_power)
            scaled_signal = signal_segment * scaling_factor
        else:
            scaled_signal = signal_segment

        # 3. Place signal within the noise floor (padding/truncation)
        final_signal = noise_floor.copy()  # Start with the noise floor

        if len(scaled_signal) == self.num_iq_samples:
            # Perfect fit, add directly
            final_signal += scaled_signal
        elif len(scaled_signal) < self.num_iq_samples:
            # Signal is shorter, needs padding. Add it to the center of the noise.
            padding_needed = self.num_iq_samples - len(scaled_signal)
            start_idx = padding_needed // 2
            end_idx = start_idx + len(scaled_signal)
            final_signal[start_idx:end_idx] += scaled_signal
        else:
            # Signal is longer, needs truncation from the center.
            excess = len(scaled_signal) - self.num_iq_samples
            start_trim = excess // 2
            end_trim = start_trim + self.num_iq_samples
            truncated_signal = scaled_signal[start_trim:end_trim]
            final_signal += truncated_signal

        return final_signal.astype(np.complex64)

    def _generate_noise_floor(self, length: int) -> np.ndarray:
        """
        Generates complex white Gaussian noise.

        Args:
            length (int): The number of samples to generate.

        Returns:
            np.ndarray: The complex noise signal.
        """
        # Using a small standard deviation for the noise to act as a base floor
        std_dev = 0.05
        noise_real = self.random_generator.normal(0, std_dev, length)
        noise_imag = self.random_generator.normal(0, std_dev, length)
        return (noise_real + 1j * noise_imag).astype(np.complex64)

    def _apply_frequency_filter(
        self,
        signal: np.ndarray,
        freq_lower: float,
        freq_upper: float,
        capture_center_freq: float,
        sample_rate: float
    ) -> np.ndarray:
        """
        Apply frequency domain filtering to isolate the signal of interest.

        Args:
            signal: Input IQ signal
            freq_lower: Lower frequency bound (Hz)
            freq_upper: Upper frequency bound (Hz) 
            capture_center_freq: Capture center frequency (Hz)
            sample_rate: Sample rate (Hz)

        Returns:
            Frequency-filtered signal
        """
        # Convert to baseband frequencies
        freq_lower_bb = freq_lower - capture_center_freq
        freq_upper_bb = freq_upper - capture_center_freq

        # Calculate center frequency and bandwidth
        signal_center_freq = (freq_lower_bb + freq_upper_bb) / 2.0
        signal_bandwidth = abs(freq_upper_bb - freq_lower_bb)

        # Add some margin to the filter (10% on each side)
        filter_margin = signal_bandwidth * 0.1
        filter_bandwidth = signal_bandwidth + 2 * filter_margin

        # Take FFT
        fft_signal = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/sample_rate))

        # Create frequency mask
        mask = np.zeros_like(freqs, dtype=bool)
        filter_lower = signal_center_freq - filter_bandwidth/2
        filter_upper = signal_center_freq + filter_bandwidth/2

        mask = (freqs >= filter_lower) & (freqs <= filter_upper)

        # Apply filter
        filtered_fft = fft_signal * mask

        # Convert back to time domain
        filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_fft))

        return filtered_signal

    def _create_narrowband_annotation(
        self,
        sigmf_file: sigmf.SigMFFile,
        ann: Dict[str, Any],
        signal_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create a single TorchSig annotation for narrowband sample.

        Args:
            sigmf_file: SigMF file object
            ann: Original SigMF annotation
            signal_data: Processed signal data

        Returns:
            TorchSig-formatted annotation or None if invalid
        """
        # Get sample rate
        sample_rate = sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)

        # Get original annotation info
        ann_start = ann.get(SigMFFile.START_INDEX_KEY, 0)
        ann_count = ann.get(SigMFFile.LENGTH_INDEX_KEY, len(signal_data))

        # Get capture info for frequency reference
        capture = self._get_capture_for_sample(sigmf_file, ann_start)
        if capture is None:
            print(
                f"Warning: No capture found for annotation at sample {ann_start}")
            return None

        capture_center_freq = capture.get(SigMFFile.FREQUENCY_KEY, 0.0)

        # Extract frequency information
        freq_lower = ann.get(SigMFFile.FLO_KEY)
        freq_upper = ann.get(SigMFFile.FHI_KEY)

        if freq_lower is None or freq_upper is None:
            print(f"Warning: Missing frequency information in annotation")
            return None

        # Convert to baseband (relative to capture center frequency)
        freq_lower_baseband = freq_lower - capture_center_freq
        freq_upper_baseband = freq_upper - capture_center_freq

        # Calculate center frequency and bandwidth
        center_freq = (freq_lower_baseband + freq_upper_baseband) / 2.0
        bandwidth = abs(freq_upper_baseband - freq_lower_baseband)

        if bandwidth == 0:
            warnings.warn(
                f"Zero bandwidth for annotation {ann}. Setting to 1e-12 Hz.")
            bandwidth = 1e-12

        # Extract class information
        sigmf_label = ann.get(SigMFFile.LABEL_KEY, 'unknown')

        # Calculate actual signal bounds within the narrowband sample
        signal_length = len(signal_data)

        # For narrowband, we need to determine where the signal actually sits
        # within the padded/truncated sample
        if ann_count <= signal_length:
            # Signal was padded or exact fit
            padding_needed = signal_length - ann_count
            padding_left = padding_needed // 2

            # Signal bounds within the sample
            signal_start_samples = padding_left
            signal_stop_samples = padding_left + ann_count
        else:
            # Signal was truncated
            signal_start_samples = 0
            signal_stop_samples = signal_length

        # Convert to time
        signal_start_seconds = signal_start_samples / sample_rate
        signal_stop_seconds = signal_stop_samples / sample_rate
        signal_duration_seconds = (
            signal_stop_samples - signal_start_samples) / sample_rate

        # Create TorchSig annotation with actual signal bounds
        torchsig_ann = {
            "bandwidth": float(bandwidth),
            "lower_freq": float(freq_lower_baseband),
            "upper_freq": float(freq_upper_baseband),
            "center_freq": float(center_freq),
            "class_index": self.label_mapping.get(sigmf_label),
            "class_name": sigmf_label,
            "duration": signal_duration_seconds,
            "duration_in_samples": signal_stop_samples - signal_start_samples,
            "num_samples": signal_stop_samples - signal_start_samples,
            "sample_rate": float(sample_rate),
            "start": signal_start_seconds,
            "start_in_samples": signal_start_samples,
            "stop": signal_stop_seconds,
            "stop_in_samples": signal_stop_samples,
            "snr_db": 0.0,  # Placeholder, not in SigMF
        }

        return torchsig_ann

    def _convert_wideband(self) -> None:
        """Convert SigMF files to TorchSig Zarr format with overlap"""

        if not self.sigmf_datasets:
            raise FileNotFoundError(
                f"No SigMF datasets found in {self.root}. Please ensure the directory contains .sigmf-meta and .sigmf-data files.")

        if self.writer.exists() and not self.overwrite:
            print(
                f"Dataset already exists in {self.path_converted}. Skipping conversion.")
            return

        print(
            f"Converting {len(self.sigmf_datasets)} SigMF files to Zarr format...")
        print(
            f"Using overlap factor: {self.overlap_factor} ({self.overlap_factor*100:.1f}%)")

        # Create writer_info.yaml file
        self._create_writer_info()

        # Setup writer
        self.writer._setup()

        all_data = []
        all_targets = []
        batch_idx = 0

        for sigmf_base in self.sigmf_datasets:
            print(f"Processing {sigmf_base}...")

            # Load SigMF file
            sigmf_file = sigmf.sigmffile.fromfile(
                str(self.root / f"{sigmf_base}"))
            # Read the data
            samples = sigmf_file.read_samples()

            # Calculate step size with overlap
            step_size = int(self.num_iq_samples * (1 - self.overlap_factor))

            # Calculate number of chunks with overlap
            num_chunks = max(
                1, (len(samples) - self.num_iq_samples) // step_size + 1)

            for chunk_idx in range(num_chunks):
                # Calculate start index with overlap
                start_idx = chunk_idx * step_size
                end_idx = start_idx + self.num_iq_samples

                # Skip if we go beyond the file
                if end_idx > len(samples):
                    break

                chunk_data = samples[start_idx:end_idx]

                # Ensure we have the correct chunk size
                if len(chunk_data) != self.num_iq_samples:
                    print(
                        f"  Warning: Chunk {chunk_idx} has {len(chunk_data)} samples, expected {self.num_iq_samples}")
                    continue

                # Create TorchSig-style annotations using the new function
                torchsig_annotations = self._create_torchsig_annotations(
                    sigmf_file, chunk_data, start_idx, end_idx
                )

                self._update_dataset_stats(torchsig_annotations)

                all_data.append(chunk_data)
                all_targets.append(torchsig_annotations)

                # Write batch when full
                if len(all_data) >= self.batch_size:
                    self.writer.write(batch_idx, (all_data, all_targets))
                    batch_idx += 1
                    all_data = []
                    all_targets = []

        # Write remaining data
        if all_data:
            self.writer.write(batch_idx, (all_data, all_targets))

        # Create YAML config
        self._create_yaml()

        total_samples = self.writer.size(str(self.path_converted))
        print(
            f"Conversion complete! Created {total_samples} samples in {str(self.path_converted)}")

    def _get_all_sigmf_datasets(self):
        meta_files = list(self.root.glob("*.sigmf-meta"))
        data_files = list(self.root.glob("*.sigmf-data"))

        # Extract base names (without extensions) for comparison
        meta_bases = {f.stem.replace('.sigmf-meta', '') for f in meta_files}
        data_bases = {f.stem.replace('.sigmf-data', '') for f in data_files}

        # Check for mismatches
        missing_data = meta_bases - data_bases
        missing_meta = data_bases - meta_bases

        if missing_data:
            warnings.warn(
                f"Found .sigmf-meta files without corresponding .sigmf-data files: {missing_data}")

        if missing_meta:
            warnings.warn(
                f"Found .sigmf-data files without corresponding .sigmf-meta files: {missing_meta}")

        # Return only matched pairs
        matched_bases = meta_bases & data_bases

        print(f"Found {len(matched_bases)} matched SigMF file pairs")

        return sorted(matched_bases)

    def _create_writer_info(self):
        """Create writer_info.yaml file required by TorchSig"""
        writer_info = {
            'root': str(self.root),
            'full_root': str(self.path_converted),
            'overwrite': self.overwrite,
            'batch_size': self.batch_size,
            'num_workers': 1,  # Not relevant for conversion
            'file_handler': 'ZarrFileHandler',
            'save_type': 'raw',
            'complete': True
        }

        writer_yaml = self.path_converted / "writer_info.yaml"
        with open(writer_yaml, 'w') as f:
            yaml.dump(writer_info, f, default_flow_style=False)

    def _create_torchsig_annotations(
        self,
        sigmf_file: sigmf.SigMFFile,
        chunk_data: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Convert SigMF annotations to TorchSig format efficiently.

        Args:
            sigmf_file: Loaded SigMF file object
            chunk_data: The actual IQ data chunk
            start_idx: Start sample index of the chunk
            end_idx: End sample index of the chunk

        Returns:
            List of TorchSig-formatted annotations
        """
        # Get SigMF metadata
        annotations: list[dict] = sigmf_file.get_annotations()

        # Extract sample rate from global info
        sample_rate = sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        chunk_length = len(chunk_data)

        # Pre-filter annotations that could overlap with this chunk
        # This is more efficient than checking every annotation
        relevant_annotations: list[dict] = [
            ann for ann in annotations
            if ann.get('core:sample_start', 0) < end_idx and
            (ann.get('core:sample_start', 0) +
             ann.get('core:sample_count', chunk_length)) > start_idx
        ]

        torchsig_annotations: List[Dict[str, Any]] = []

        for ann in relevant_annotations:
            # Extract SigMF annotation fields
            ann_start: int = ann.get('core:sample_start', 0)
            ann_count: int = ann.get('core:sample_count', chunk_length)
            ann_end: int = ann_start + ann_count

            # Calculate overlap with current chunk
            rel_start: int = max(0, ann_start - start_idx)
            rel_end: int = min(chunk_length, ann_end - start_idx)
            rel_count: int = rel_end - rel_start

            # Skip if no actual overlap
            if rel_count <= 0:
                continue

            capture = self._get_capture_for_sample(
                sigmf_file=sigmf_file, sample_index=ann_start)
            capture_center_freq = capture.get(SigMFFile.FREQUENCY_KEY)

            # Extract frequency information
            freq_lower: float = ann.get(SigMFFile.FLO_KEY)
            freq_upper: float = ann.get(SigMFFile.FHI_KEY)

            # Convert to relative frequencies
            freq_lower -= capture_center_freq
            freq_upper -= capture_center_freq

            # Calculate center frequency and bandwidth
            center_freq = (freq_lower + freq_upper) / 2.0
            bandwidth = abs(freq_upper - freq_lower)

            if bandwidth == 0:
                warnings.warn(
                    f"Zero bandwidth for annotation {ann}. Setting to 1e-12 Hz.")
                bandwidth = 1e-12

            # Extract class information
            sigmf_label: str = ann.get('core:label', 'unknown')

            # Calculate timing information
            duration_seconds: float = rel_count / sample_rate
            start_seconds: float = rel_start / sample_rate
            stop_seconds: float = rel_end / sample_rate

            # Create TorchSig annotation
            torchsig_ann: Dict[str, Any] = {
                "bandwidth": float(bandwidth),
                "lower_freq": float(freq_lower),
                "upper_freq": float(freq_upper),
                "center_freq": float(center_freq),
                "class_index": self.label_mapping.get(sigmf_label),
                "class_name": sigmf_label,
                "duration": duration_seconds,
                "duration_in_samples": rel_count,
                "num_samples": rel_count,
                "sample_rate": float(sample_rate),
                "start": start_seconds,
                "start_in_samples": rel_start,
                "stop": stop_seconds,
                "stop_in_samples": rel_end,
                "snr_db": 0.0,  # Placeholder, not in SigMF
            }

            torchsig_annotations.append(torchsig_ann)

        return torchsig_annotations

    def _create_label_mapping(self) -> Dict[str, int]:
        """
        Create a label mapping from SigMF annotations to TorchSig classes.

        Returns:
            Dict[str, int]: Mapping of class names to indices.
        """
        # Collect unique labels from all SigMF files
        unique_labels = set()
        for sigmf_base in self.sigmf_datasets:
            sigmf_file = sigmf.sigmffile.fromfile(
                str(self.root / f"{sigmf_base}"))
            annotations = sigmf_file.get_annotations()
            for ann in annotations:
                label = ann.get('core:label', 'unknown')
                unique_labels.add(label)

        # Create mapping
        label_mapping = {label: idx for idx,
                         label in enumerate(sorted(unique_labels))}
        return label_mapping

    def _get_dataset_sample_rate(self) -> float:
        """
        Get the sample rate from the first SigMF file.

        Returns:
            float: Sample rate of the dataset.
        """
        if not self.sigmf_datasets:
            raise ValueError("No SigMF datasets found.")

        sigmf_file = sigmf.sigmffile.fromfile(
            str(self.root / f"{self.sigmf_datasets[0]}"))
        return sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)

    def _create_yaml(self):

        computed_stats = self._compute_final_stats()

        # Custom class to mark lists that should be in flow style
        class FlowList(list):
            pass

        # Custom representer for flow-style lists
        def represent_flow_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        # Register the custom representer
        yaml.add_representer(FlowList, represent_flow_list)

        data = {
            "required": {
                "dataset_type": self.dataset,
                "num_iq_samples_dataset": self.num_iq_samples,
                "fft_size": self.fft_size,
                "impairment_level": 2,
                "num_signals_max": computed_stats.get('num_signals_max', 0),
            },
            "overrides": {
                "impairment_level": 2,
                "num_iq_samples_dataset": self.num_iq_samples,
                "fft_size": self.fft_size,
                "class_list": FlowList(list(self.label_mapping.keys())),
                "sample_rate": self.sample_rate,
                **computed_stats
            },
            "read_only": {
                "info": {
                    "dataset_type": self.dataset,
                    "num_iq_samples_dataset": self.num_iq_samples,
                    "fft_size": self.fft_size,
                    "sample_rate": self.sample_rate,
                },
                "signals": {
                    "class_list": FlowList(list(self.label_mapping.keys())),
                    **computed_stats
                }
            },
        }

        if self.dataset == "narrowband":
            data["required"].pop("num_signals_max")

        yaml_path = self.path_converted / dataset_yaml_name
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _get_capture_for_sample(
        self,
        sigmf_file: SigMFFile,
        sample_index: int,
    ) -> Dict[str, Any]:
        """
        Retrieves the most relevant capture metadata for a given sample index from SigMF file.

        :param sigmf_file: SigMF file object containing capture metadata
        :type sigmf_file: sigmffile.SigMFFile
        :param sample_index: Sample index for which to find corresponding capture
        :type sample_index: int
        :returns: Capture dictionary with greatest sample_start ≤ sample_index, or None if no captures
        :rtype: Optional[Dict[str, Any]]
        """

        captures = sigmf_file.get_captures()
        if not captures:
            return None

        # Find the capture with the greatest sample_start that is ≤ sample_index
        chosen_capture = captures[0]
        chosen_start = chosen_capture.get(SigMFFile.START_INDEX_KEY, 0)

        for cap in captures[1:]:
            cap_start = cap.get(SigMFFile.START_INDEX_KEY, 0)
            if cap_start <= sample_index and cap_start > chosen_start:
                chosen_capture = cap
                chosen_start = cap_start

        return chosen_capture

    def _update_dataset_stats(self, annotations: List[Dict[str, Any]]) -> None:
        """Update dataset statistics with annotations from current chunk"""

        # Track number of signals per chunk
        self.dataset_stats['num_signals_per_chunk'].append(len(annotations))

        # Track individual signal statistics
        for ann in annotations:
            # Duration in seconds
            duration = ann.get('duration', 0.0)
            self.dataset_stats['signal_durations'].append(duration)

            # Duration in samples
            duration_samples = ann.get('duration_in_samples', 0)
            self.dataset_stats['signal_durations_samples'].append(
                duration_samples)

            # Bandwidth
            bandwidth = ann.get('bandwidth', 0.0)
            self.dataset_stats['signal_bandwidths'].append(bandwidth)

            # Center frequency (baseband)
            center_freq = ann.get('center_freq', 0.0)
            self.dataset_stats['signal_center_freqs'].append(center_freq)

    def _compute_final_stats(self) -> Dict[str, Any]:
        """Compute final statistics from collected data"""

        stats = {}

        # Number of signals statistics
        # if self.dataset_stats['num_signals_per_chunk']:
        #    stats.update({
        #        'num_signals_min': min(self.dataset_stats['num_signals_per_chunk']),
        #        'num_signals_max': max(self.dataset_stats['num_signals_per_chunk']),
        #    })

        # Signal duration statistics (seconds)
        # if self.dataset_stats['signal_durations']:
        #    stats.update({
        #        'signal_duration_min': min(self.dataset_stats['signal_durations']),
        #        'signal_duration_max': max(self.dataset_stats['signal_durations']),
        #    })

        # Signal duration statistics (samples)
        # if self.dataset_stats['signal_durations_samples']:
        #    stats.update({
        #        'signal_duration_in_samples_min': min(self.dataset_stats['signal_durations_samples']),
        #        'signal_duration_in_samples_max': max(self.dataset_stats['signal_durations_samples']),
        #    })

        # Signal bandwidth statistics
        # if self.dataset_stats['signal_bandwidths']:
        #    stats.update({
        #        'signal_bandwidth_min': min(self.dataset_stats['signal_bandwidths']),
        #        'signal_bandwidth_max': max(self.dataset_stats['signal_bandwidths']),
        #    })

        # Signal center frequency statistics (baseband)
        # if self.dataset_stats['signal_center_freqs']:
        #    stats.update({
        #        'signal_center_freq_min': min(self.dataset_stats['signal_center_freqs']),
        #        'signal_center_freq_max': max(self.dataset_stats['signal_center_freqs']),
        #    })

        return stats
