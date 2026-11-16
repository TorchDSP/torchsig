"""Shared contract tests for concrete audio readers."""

import pytest

from torchsig.utils.file_handlers.audio_reader import AudioReader
from torchsig.utils.file_handlers.ogg import OGGReader
from torchsig.utils.file_handlers.wav import WAVReader


@pytest.mark.parametrize("reader_type", [WAVReader, OGGReader])
def test_concrete_audio_readers_are_policy_only_subclasses(reader_type) -> None:
    assert issubclass(reader_type, AudioReader)
    assert "__init__" not in reader_type.__dict__
    assert "read" not in reader_type.__dict__
