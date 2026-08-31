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


def _pack_value(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, np.ndarray):
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {
            "__torchsig_type__": "ndarray",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _pack_value(np.asarray(value)) | {"scalar": True}
    if isinstance(value, tuple):
        return {
            "__torchsig_type__": "tuple",
            "items": [_pack_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_pack_value(item) for item in value]
    if isinstance(value, dict):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError(f"TorchSig metadata dictionary keys must be strings; got {type(non_string_keys[0]).__name__}")
        return {
            "__torchsig_type__": "dict",
            "items": {key: _pack_value(item) for key, item in value.items()},
        }
    if isinstance(value, bytes):
        return {
            "__torchsig_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, complex):
        return {
            "__torchsig_type__": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported TorchSig metadata type: {type(value).__name__}")


def _unpack_value(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, list):
        return [_unpack_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get("__torchsig_type__")
    if value_type == "ndarray":
        array = np.load(
            BytesIO(base64.b64decode(value["data"])),
            allow_pickle=False,
        )
        return array[()] if value.get("scalar", False) else array
    if value_type == "tuple":
        return tuple(_unpack_value(item) for item in value["items"])
    if value_type == "bytes":
        return base64.b64decode(value["data"])
    if value_type == "complex":
        return complex(value["real"], value["imag"])
    if value_type == "dict":
        return {key: _unpack_value(item) for key, item in value["items"].items()}
    return {key: _unpack_value(item) for key, item in value.items()}


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
