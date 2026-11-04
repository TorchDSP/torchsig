"""File handler for stereo WAV IQ datasets."""

from .audio_reader import AudioReader

__all__ = ["WAVReader"]


class WAVReader(AudioReader):
    """Read fixed-length or manifest-indexed stereo WAV IQ records."""

    suffix = ".wav"
    file_attribute = "wav_files"
    missing_files_message = "No .wav files found in {root}"
    mismatch_message = "Metadata contains {size} elements, but WAV layout implies {total}."
