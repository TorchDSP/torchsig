"""File handler for stereo WAV IQ datasets."""

from pathlib import Path
from typing import Self

import soundfile as sf

from torchsig.signals.signal_types import Signal

from .audio import AudioFidelity, AudioHandleCache, AudioRecordLayout
from .metadata_reader import MetadataReader

__all__ = ["WAVReader"]


class WAVReader(MetadataReader):
    """Read fixed-length or manifest-indexed stereo WAV IQ records.

    An explicit manifest adds ``file_path``, ``start_frame``, and
    ``num_frames`` columns to ``metadata.csv``. Records may then have variable
    lengths and arbitrary distribution across nested WAV files. Legacy
    datasets continue to use ``num_iq_samples`` and ``elements_per_file``.
    """

    def __init__(
        self,
        root: str | Path,
        audio_handle_cache_size: int = 8,
        normalization: str = "none",
        record_normalization_scale: bool = False,
    ) -> None:
        """Initialize the reader and its fidelity and handle-cache policies.

        ``normalization`` accepts ``"none"`` (the default), ``"rms"``, or
        ``"peak"``. If ``record_normalization_scale`` is true, reads include
        the applied scalar in metadata as ``normalization_scale``.
        """
        super().__init__(root)
        self._audio_fidelity = AudioFidelity(normalization)
        self.normalization = self._audio_fidelity.normalization
        self.record_normalization_scale = bool(record_normalization_scale)
        self.audio_handle_cache_size = audio_handle_cache_size
        self._audio_handles = AudioHandleCache(audio_handle_cache_size)
        self.wav_files = sorted(self.root.rglob("*.wav"), key=str)
        if not self.wav_files:
            raise FileNotFoundError(f"No .wav files found in {self.root}")

        if not self._has_explicit_manifest() and (self.num_iq_samples == 0 or self.elements_per_file == 0):
            first_frames = int(sf.info(self.wav_files[0]).frames)
            if self.elements_per_file == 0:
                if self.dataset_size > 0 and self.dataset_size % len(self.wav_files) == 0:
                    self.elements_per_file = self.dataset_size // len(self.wav_files)
                else:
                    self.elements_per_file = 1
            self.num_iq_samples = first_frames // self.elements_per_file

        if not self._has_explicit_manifest():
            legacy_total = len(self.wav_files) * self.elements_per_file
            if self.dataset_size != legacy_total:
                raise ValueError(f"Metadata contains {self.dataset_size} elements, but WAV layout implies {legacy_total}.")

        self.record_layout = AudioRecordLayout(
            self.root,
            self._metadata_rows,
            self.wav_files,
            num_iq_samples=self.num_iq_samples,
            elements_per_file=self.elements_per_file,
        )
        AudioFidelity.validate_layout(self.record_layout, self._metadata_rows, self.sample_rate, reject_lossy_ogg=False)
        self.total_elements = len(self.record_layout)
        self.file_start_indices = [] if self.record_layout.is_manifest else list(range(0, self.total_elements, self.elements_per_file))

    def _has_explicit_manifest(self) -> bool:
        """Return whether metadata selects explicit record descriptors."""
        return any("file_path" in row for row in self._metadata_rows)

    def read(self, idx: int) -> Signal:
        """Return the IQ record at global index ``idx``."""
        if idx < 0 or idx >= self.dataset_size:
            raise IndexError(f"index {idx} out of range (size={self.dataset_size})")
        record = self.record_layout[idx]
        stereo = self._audio_handles.read(record.path, record.start_frame, record.num_frames)
        complex_vec, scale = self._audio_fidelity.convert(stereo, record)
        metadata = self.load_row(idx, self.class_list)
        if self.record_normalization_scale:
            metadata["normalization_scale"] = scale
        return Signal(data=complex_vec, component_signals=[], metadata=metadata)

    def setup(self) -> None:
        """Prepare the reader for use after an earlier teardown."""
        self._audio_handles.setup()

    def teardown(self) -> None:
        """Close all process-local audio handles."""
        self._audio_handles.close()

    def __enter__(self) -> Self:
        """Prepare and return this reader as a context manager."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Close cached handles when leaving a context."""
        self.teardown()
        return False

    def __len__(self) -> int:
        """Return the number of indexed records."""
        return self.dataset_size
