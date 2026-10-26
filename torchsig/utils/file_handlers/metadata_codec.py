"""Version 1 TorchSig JSON metadata codec."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torchsig.utils.abstractions import HierarchicalMetadataObject

__all__ = ["decode_metadata", "encode_metadata"]


def _pack_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        packed = {
            "__torchsig_type__": "ndarray",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    elif isinstance(value, np.generic):
        packed = _pack_value(np.asarray(value)) | {"scalar": True}
    elif isinstance(value, tuple):
        packed = {
            "__torchsig_type__": "tuple",
            "items": [_pack_value(item) for item in value],
        }
    elif isinstance(value, list):
        packed = [_pack_value(item) for item in value]
    elif isinstance(value, dict):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError(f"TorchSig metadata dictionary keys must be strings; got {type(non_string_keys[0]).__name__}")
        packed = {
            "__torchsig_type__": "dict",
            "items": {key: _pack_value(item) for key, item in value.items()},
        }
    elif isinstance(value, bytes):
        packed = {
            "__torchsig_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    elif isinstance(value, complex):
        packed = {
            "__torchsig_type__": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    elif value is None or isinstance(value, (str, int, float, bool)):
        packed = value
    else:
        raise TypeError(f"Unsupported TorchSig metadata type: {type(value).__name__}")
    return packed


def _unpack_value(value: Any) -> Any:
    if isinstance(value, list):
        unpacked = [_unpack_value(item) for item in value]
    elif not isinstance(value, dict):
        unpacked = value
    elif (value_type := value.get("__torchsig_type__")) == "ndarray":
        array = np.load(
            BytesIO(base64.b64decode(value["data"])),
            allow_pickle=False,
        )
        unpacked = array[()] if value.get("scalar", False) else array
    elif value_type == "tuple":
        unpacked = tuple(_unpack_value(item) for item in value["items"])
    elif value_type == "bytes":
        unpacked = base64.b64decode(value["data"])
    elif value_type == "complex":
        unpacked = complex(value["real"], value["imag"])
    elif value_type == "dict":
        unpacked = {key: _unpack_value(item) for key, item in value["items"].items()}
    else:
        unpacked = {key: _unpack_value(item) for key, item in value.items()}
    return unpacked


def encode_metadata(obj: HierarchicalMetadataObject) -> str:
    """Encode an object's local metadata as deterministic TorchSig JSON."""
    # HierarchicalMetadataObject is not iterable; its keys() API is required.
    metadata = {key: obj[key] for key in obj.keys()}  # noqa: SIM118
    return json.dumps(
        _pack_value(metadata),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=True,
    )


def decode_metadata(value: str | bytes) -> dict[str, Any]:
    """Decode TorchSig JSON metadata from text or UTF-8 bytes."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _unpack_value(json.loads(value))
