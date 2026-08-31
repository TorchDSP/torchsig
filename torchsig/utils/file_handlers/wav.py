"""File-handler that reads a directory of **stereo** WAV recordings.
Each WAV file stores *elements_per_file* whole IQ records; every record
contains *num_iq_samples* complex samples (I/Q pairs).
"""

import bisect
from pathlib import Path

import numpy as np
import soundfile as sf

from torchsig.signals.signal_types import Signal

from .metadata_reader import MetadataReader

__all__ = ["WAVReader"]


class WAVReader(MetadataReader):
    """Read a directory that contains a TorchSig-compatible WAV dataset.

    Required files
    ---------------
    * ``*.wav`` - **stereo** audio files.  Each file must store *N*
      whole IQ records, i.e. ``N x num_iq_samples`` frames.  The reader
      interprets the *left* channel as the **I** (in-phase) component and
      the *right* channel as the **Q** (quadrature) component, so the audio
      stream is effectively a *pair of interleaved real-valued tracks*.
    * ``metadata.csv`` - one row **per element** (not per frame).  The row
      supplies the label, modcod, sample-rate, etc.  No header is expected
      because :class:`WAVReader` supplies the field names internally.
    * Optional ``info.json`` - must contain the keys ``num_iq_samples`` and
      ``elements_per_file`` (or they can be inferred from the first WAV file).
      The JSON also holds the total dataset size, class list, sample-rate,
      and any other user-defined metadata.

    Public API
    ----------
    The reader operates on **element indices**:
    ``reader.read(i)`` → returns the *i-th* IQ record as a
    ``torchsig.signals.Signal`` whose ``data`` attribute is a
    ``complex64`` array of shape ``(num_iq_samples,)``.  The associated
    metadata row is attached to ``Signal.metadata``.
    """

    def __init__(self, root: str | Path) -> None:
        # 1. Initialise base class
        super().__init__(root)
        self.root = Path(self.root).resolve()

        # 2. Locate and sort all WAV files
        self.wav_files = sorted(list(self.root.rglob("*.wav")), key=lambda p: str(p))
        if not self.wav_files:
            raise FileNotFoundError(f"No .wav files found in {self.root}")

        # ------------------------------------------------------------------
        # Robustly recalculate dataset_size from the CSV
        # ------------------------------------------------------------------
        csv_path = self.root / "metadata.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                # Count only non-empty lines to avoid trailing newline errors
                rows = [line for line in f if line.strip()]
                self.dataset_size = len(rows)
                # Also update the base class rows if it's using a different list
                if hasattr(self, "metadata_rows"):
                    self.metadata_rows = rows

        # ------------------------------------------------------------------
        # Inference of layout
        # ------------------------------------------------------------------
        if self.num_iq_samples == 0 or self.elements_per_file == 0:
            first_info = sf.info(self.wav_files[0])
            total_frames = int(first_info.frames)

            if self.elements_per_file == 0:
                # Only trust dataset_size if it's a perfect multiple of files
                if self.dataset_size > 0 and self.dataset_size % len(self.wav_files) == 0:
                    self.elements_per_file = self.dataset_size // len(self.wav_files)
                else:
                    # Default to 1 element per file if CSV is missing/wrong
                    self.elements_per_file = 1

            self.num_iq_samples = total_frames // self.elements_per_file

        # ------------------------------------------------------------------
        # Build the indexing table
        # ------------------------------------------------------------------
        self.file_start_indices = []
        cum = 0
        for _ in self.wav_files:
            self.file_start_indices.append(cum)
            cum += self.elements_per_file
        self.total_elements = cum

        # FINAL SANITY CHECK:
        # If the CSV was too short, we force dataset_size to match the files
        # so that reader.read(idx) doesn't throw IndexError.
        self.dataset_size = max(self.dataset_size, self.total_elements)

    def read(self, idx: int) -> Signal:
        if idx < 0 or idx >= self.dataset_size:
            raise IndexError(f"index {idx} out of range")

        # Use bisect to find which file contains the idx-th element
        file_idx = bisect.bisect_right(self.file_start_indices, idx) - 1
        element_offset = idx - self.file_start_indices[file_idx]

        # Ensure we are using the SAME sorted list as __init__
        wav_path = self.wav_files[file_idx]

        # ... (read and convert to complex) ...
        # Ensure dtype is float32 to avoid the 0.9999 vs 1.0 issue
        pcm, _ = sf.read(wav_path, dtype="float32", always_2d=True)

        # Extract the specific record
        start_frame = element_offset * self.num_iq_samples
        end_frame = start_frame + self.num_iq_samples
        stereo = pcm[start_frame:end_frame, :]

        complex_vec = (stereo[:, 0] + 1j * stereo[:, 1]).astype(np.complex64)
        # --------------------------------------------------------------
        # Pull the CSV row that belongs to this element.
        # --------------------------------------------------------------
        metadata = self.load_row(idx, self.class_list)
        # --------------------------------------------------------------
        # Assemble and return the Signal.
        # --------------------------------------------------------------
        return Signal(
            data=complex_vec,
            component_signals=[],
            metadata=metadata,
        )

    def __len__(self) -> int:
        """Number of elements (rows in metadata.csv)."""
        return self.dataset_size
