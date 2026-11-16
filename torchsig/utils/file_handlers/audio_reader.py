"""Shared reader implementation for audio-backed IQ datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import soundfile as sf

from torchsig.signals.signal_types import Signal

from .audio import AudioFidelity, AudioHandleCache, AudioRecordLayout
from .metadata_reader import MetadataReader

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["AudioReader"]


class AudioReader(MetadataReader):
    """Read manifest-indexed or legacy fixed-length stereo IQ audio."""

    suffix: ClassVar[str]
    reject_lossy: ClassVar[bool] = False
    file_attribute: ClassVar[str] = "audio_files"
    missing_files_message: ClassVar[str]
    mismatch_message: ClassVar[str]

    def __init__(
        self,
        root: str | Path,
        audio_handle_cache_size: int = 8,
        normalization: str = "none",
        record_normalization_scale: bool = False,
        *,
        startup_validation: bool = True,
    ) -> None:
        super().__init__(root)
        self._audio_fidelity = AudioFidelity(normalization)
        self.normalization = self._audio_fidelity.normalization
        self.record_normalization_scale = bool(record_normalization_scale)
        self.audio_handle_cache_size = audio_handle_cache_size
        self._audio_handles = AudioHandleCache(audio_handle_cache_size)
        self.audio_files = sorted(self.root.rglob(f"*{self.suffix}"), key=str)
        setattr(self, self.file_attribute, self.audio_files)
        if not self.audio_files:
            raise FileNotFoundError(self.missing_files_message.format(root=self.root))

        if not self._has_explicit_manifest() and (self.num_iq_samples == 0 or self.elements_per_file == 0):
            first_frames = int(sf.info(self.audio_files[0]).frames)
            if self.elements_per_file == 0:
                self.elements_per_file = self.dataset_size // len(self.audio_files) if self.dataset_size > 0 and self.dataset_size % len(self.audio_files) == 0 else 1
            self.num_iq_samples = first_frames // self.elements_per_file

        if not self._has_explicit_manifest():
            legacy_total = len(self.audio_files) * self.elements_per_file
            if self.dataset_size != legacy_total:
                raise ValueError(self.mismatch_message.format(size=self.dataset_size, total=legacy_total))

        self.record_layout = AudioRecordLayout(
            self.root,
            self._metadata_rows,
            self.audio_files,
            num_iq_samples=self.num_iq_samples,
            elements_per_file=self.elements_per_file,
        )
        self._layout_validated = False
        if startup_validation:
            self._validate_layout()
        self.total_elements = len(self.record_layout)
        self.file_start_indices = [] if self.record_layout.is_manifest else list(range(0, self.total_elements, self.elements_per_file))

    def _validate_layout(self) -> None:
        if not self._layout_validated:
            AudioFidelity.validate_layout(self.record_layout, self._metadata_rows, self.sample_rate, reject_lossy_ogg=self.reject_lossy)
            self._layout_validated = True

    def _has_explicit_manifest(self) -> bool:
        return any("file_path" in row for row in self._metadata_rows)

    def read(self, idx: int) -> Signal:
        """Return one complex IQ signal and its metadata."""
        self.validate_index(idx)
        self._validate_layout()
        record = self.record_layout[idx]
        stereo = self._audio_handles.read(record.path, record.start_frame, record.num_frames)
        complex_vec, scale = self._audio_fidelity.convert(stereo, record)
        metadata = self.load_row(idx, self.class_list)
        if self.record_normalization_scale:
            metadata["normalization_scale"] = scale
        return Signal(data=complex_vec, component_signals=[], metadata=metadata)

    def setup(self) -> None:
        """Prepare process-local audio resources."""
        self._audio_handles.setup()

    def teardown(self) -> None:
        """Close all cached process-local handles."""
        self._audio_handles.close()

    def __len__(self) -> int:
        """Return the number of indexed audio records."""
        return self.dataset_size
