"""File handler for stereo OGG-container IQ datasets."""

from .audio_reader import AudioReader

__all__ = ["OGGReader"]


class OGGReader(AudioReader):
    """Read stereo OGG IQ records while rejecting lossy OGG codecs."""

    suffix = ".ogg"
    reject_lossy = True
    file_attribute = "ogg_files"
    missing_files_message = "No .ogg files in {root}"
    mismatch_message = "Metadata reports {size} elements, but OGG files contain {total}."
