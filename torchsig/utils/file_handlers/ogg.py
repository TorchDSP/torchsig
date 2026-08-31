"""File-handler that reads a directory of **stereo** OGG recordings.
Each OGG file stores *elements_per_file* whole IQ records; every record
contains *num_iq_samples* complex samples (I/Q pairs).
"""

import bisect
from pathlib import Path

import numpy as np
import soundfile as sf

from torchsig.signals.signal_types import Signal

from .metadata_reader import MetadataReader

__all__ = ["OGGReader"]


class OGGReader(MetadataReader):
    """Read a directory that contains a TorchSig-compatible OGG dataset.

    Required files
    ---------------
    * ``*.ogg`` - **stereo** audio files.  Each file must store *N*
      whole IQ records, i.e. ``N x num_iq_samples`` frames.  The reader
      interprets the *left* channel as the **I** (in-phase) component and
      the *right* channel as the **Q** (quadrature) component, so the audio
      stream is effectively a *pair of interleaved real-valued tracks*.

    * ``metadata.csv`` - one row **per element** (not per frame).  The row
      supplies the label, modcod, sample-rate, etc.  No header is expected
      because :class:`OGGReader` supplies the field names internally.

    * Optional ``info.json`` - must contain the keys ``num_iq_samples`` and
      ``elements_per_file`` (or they can be inferred from the first OGG file).
      The JSON also holds the total dataset size, class list, sample-rate,
      and any other user-defined metadata.

    **Current limitation** - The implementation works **only with loss-less,
    un-compressed audio streams**.  In practice this means the OGG files must
    contain either:

    * a **WAV-PCM-FLOAT** stream (32-bit floating-point PCM), or
    * a **FLAC** stream (lossless compression).

    OGG files encoded with a *lossy* codec such as Vorbis, Opus, or Speex are
    **not supported**; attempting to read them will raise an exception because
    the underlying data layout (interleaved I/Q samples) can no longer be
    recovered reliably.

    Public API
    ----------
    The reader operates on **element indices**:

    ``reader.read(i)`` → returns the *i-th* IQ record as a
    ``torchsig.signals.Signal`` whose ``data`` attribute is a
    ``complex64`` array of shape ``(num_iq_samples,)``.  The associated
    metadata row is attached to ``Signal.metadata``.

    """

    def __init__(self, root: str | Path) -> None:
        # Initialise the base class (sets ``self.root`` and the CSV/JSON helpers)
        super().__init__(root)

        # --------------------------------------------------------------
        # Locate and sort all OGG files.
        # --------------------------------------------------------------
        self.ogg_files: list[Path] = sorted(self.root.glob("*.ogg"))
        if not self.ogg_files:
            raise FileNotFoundError(f"No .ogg files in {self.root}")

        # If the JSON did not provide the numbers, discover them from the
        # first OGG file (all files are assumed to have the same layout).
        if self.num_iq_samples == 0 or self.elements_per_file == 0:
            first_info = sf.info(self.ogg_files[0])
            total_frames = int(first_info.frames)  # total stereo frames in file
            # Guess a sensible split: we know the overall ``dataset_size``
            # (number of elements) from the JSON; we also know the number of files.
            # ``elements_per_file`` = ceil(dataset_size / num_files)
            if self.elements_per_file == 0:
                self.elements_per_file = int(np.ceil(self.dataset_size / len(self.ogg_files)))
            # Now deduce ``num_iq_samples``:
            self.num_iq_samples = total_frames // self.elements_per_file

        # --------------------------------------------------------------
        # Build the cumulative-start-index table that maps a **global element**
        # index → (file index, element-offset-inside-file).
        # --------------------------------------------------------------
        self.file_start_indices: list[int] = []  # cumulative count of elements
        cum = 0
        for _ogg_path in self.ogg_files:
            self.file_start_indices.append(cum)
            cum += self.elements_per_file
        # ``self.dataset_size`` (inherited from MetadataReader) should already
        # equal the total number of elements, but we recompute a sanity-check:
        self.total_elements = cum  # total number of **elements** in the dataset

        # Sanity-check: make sure the CSV size matches what we think we have.
        if self.dataset_size != self.total_elements:
            raise ValueError(f"Metadata reports {self.dataset_size} elements, but OGG files contain {self.total_elements}.")

    def read(self, idx: int) -> Signal:
        """Return the full IQ record for the *idx*-th element.

        Parameters
        ----------
        idx : int
            Zero-based **element** index (0 ≤ idx < self.dataset_size).

        Returns
        -------
        Signal
            ``Signal.data`` is a ``complex64`` array of shape
            ``(num_iq_samples,)``.  ``metadata`` comes from the CSV row
            that corresponds to the element.
        """
        # --------------------------------------------------------------
        # Bounds check.
        # --------------------------------------------------------------
        if idx < 0 or idx >= self.dataset_size:
            raise IndexError(f"index {idx} out of range (size={self.dataset_size})")

        # --------------------------------------------------------------
        # Locate the file that holds this element.
        # --------------------------------------------------------------
        file_idx = bisect.bisect_right(self.file_start_indices, idx) - 1
        element_offset_in_file = idx - self.file_start_indices[file_idx]

        # --------------------------------------------------------------
        # Read the **entire** file (all elements it stores).
        # --------------------------------------------------------------
        ogg_path = self.ogg_files[file_idx]
        total_frames_in_file = self.elements_per_file * self.num_iq_samples
        pcm, _ = sf.read(
            ogg_path,
            start=0,  # read from the beginning of the file
            frames=total_frames_in_file,
            dtype="float32",
            always_2d=False,
        )
        # Reshape to (elements_per_file, num_iq_samples, 2)
        pcm = pcm.reshape(self.elements_per_file, self.num_iq_samples, 2)

        # --------------------------------------------------------------
        # Extract the element we actually asked for.
        # --------------------------------------------------------------
        stereo = pcm[element_offset_in_file]  # shape (num_iq_samples, 2)

        # --------------------------------------------------------------
        # Convert to a complex64 vector.
        # --------------------------------------------------------------
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
