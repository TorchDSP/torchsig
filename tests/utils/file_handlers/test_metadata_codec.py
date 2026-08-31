"""Tests for the shared TorchSig JSON metadata codec."""

import numpy as np
import pytest

from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.metadata_codec import (
    decode_metadata,
    encode_metadata,
)


def test_metadata_codec_encoding_is_deterministic() -> None:
    first = HierarchicalMetadataObject(metadata={"z": 1, "a": "value"})
    second = HierarchicalMetadataObject(metadata={"a": "value", "z": 1})

    assert encode_metadata(first) == encode_metadata(second)
    assert encode_metadata(first) == ('{"__torchsig_type__":"dict","items":{"a":"value","z":1}}')


def test_metadata_codec_decodes_utf8_bytes() -> None:
    metadata = HierarchicalMetadataObject(
        metadata={
            "array": np.array([[1, 2], [3, 4]], dtype=np.int16),
            "scalar": np.float32(1.5),
            "tuple": ("value", 2),
            "bytes": b"\x00\xff",
            "complex": 1 + 2j,
        }
    )

    decoded = decode_metadata(encode_metadata(metadata).encode("utf-8"))

    np.testing.assert_array_equal(decoded.pop("array"), metadata["array"])
    assert decoded == {
        "scalar": np.float32(1.5),
        "tuple": ("value", 2),
        "bytes": b"\x00\xff",
        "complex": 1 + 2j,
    }


def test_metadata_codec_preserves_reserved_marker_dictionary() -> None:
    value = {
        "__torchsig_type__": "complex",
        "real": "ordinary metadata",
    }
    metadata = HierarchicalMetadataObject(metadata={"value": value})

    assert decode_metadata(encode_metadata(metadata))["value"] == value


def test_metadata_codec_rejects_unsupported_value() -> None:
    metadata = HierarchicalMetadataObject(metadata={"value": object()})

    with pytest.raises(TypeError, match="Unsupported TorchSig metadata"):
        encode_metadata(metadata)


def test_metadata_codec_rejects_non_string_dictionary_key() -> None:
    metadata = HierarchicalMetadataObject(metadata={"value": {1: "bad"}})

    with pytest.raises(TypeError, match="keys must be strings"):
        encode_metadata(metadata)
