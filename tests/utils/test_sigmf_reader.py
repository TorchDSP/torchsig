"""Tests for SigMFReader, SigMFFileHandler, and StaticTorchSigDataset integration."""

# Built-In
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import pytest

# TorchSig
from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.file_handlers.sigmf import SigMFFileHandler, SigMFReader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sigmf_pair(
    tmp_path: Path,
    signals: List[np.ndarray],
    sample_rate: float,
    datatype: str,
    filename_stem: str,
    captures: Optional[List[Dict[str, Any]]] = None,
    extra_annotations: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write a .sigmf-data binary file and .sigmf-meta JSON sidecar in *tmp_path*.

    Args:
        tmp_path: Directory in which to create the files.
        signals: List of np.complex64 arrays (one per annotation).
        sample_rate: Global sample rate written to the meta file.
        datatype: SigMF datatype string (e.g. ``"cf32_le"``).
        filename_stem: Base name without extension (e.g. ``"capture"``).
        captures: Capture segment list.  Defaults to a single capture at
            sample 0 with ``core:frequency = 2.4e9``.
        extra_annotations: Per-annotation extra key/value dicts merged
            into each annotation entry.

    Returns:
        Path: The ``.sigmf-data`` file path.
    """
    if captures is None:
        captures = [{"core:sample_start": 0, "core:frequency": 2.4e9}]
    if extra_annotations is None:
        extra_annotations = [{} for _ in signals]

    # Build raw binary according to datatype
    raw_parts: List[bytes] = []
    for sig in signals:
        sig = np.asarray(sig, dtype=np.complex64)
        if datatype == "cf32_le":
            interleaved = np.empty(len(sig) * 2, dtype="<f4")
            interleaved[0::2] = sig.real
            interleaved[1::2] = sig.imag
            raw_parts.append(interleaved.tobytes())
        elif datatype == "ci16_le":
            scaled = np.round(np.stack([sig.real, sig.imag], axis=1).ravel() * 32767.0)
            raw_parts.append(scaled.astype("<i2").tobytes())
        elif datatype == "ci16_be":
            scaled = np.round(np.stack([sig.real, sig.imag], axis=1).ravel() * 32767.0)
            raw_parts.append(scaled.astype(">i2").tobytes())
        elif datatype == "ci8":
            scaled = np.round(np.stack([sig.real, sig.imag], axis=1).ravel() * 127.0)
            raw_parts.append(scaled.astype("i1").tobytes())
        elif datatype == "cu8":
            vals = np.stack([sig.real, sig.imag], axis=1).ravel()
            raw_parts.append(
                np.clip(np.round(vals * 127.0 + 128.0), 0, 255).astype("u1").tobytes()
            )
        else:
            raise ValueError(f"_make_sigmf_pair: unsupported datatype {datatype!r}")

    data_file = tmp_path / f"{filename_stem}.sigmf-data"
    data_file.write_bytes(b"".join(raw_parts))

    # Build annotations
    annotations = []
    sample_cursor = 0
    for i, sig in enumerate(signals):
        ann: Dict[str, Any] = {
            "core:sample_start": sample_cursor,
            "core:sample_count": len(sig),
            "core:label": "bpsk",
        }
        ann.update(extra_annotations[i])
        annotations.append(ann)
        sample_cursor += len(sig)

    meta = {
        "global": {
            "core:datatype": datatype,
            "core:sample_rate": sample_rate,
            "core:version": "0.0.2",
        },
        "captures": captures,
        "annotations": annotations,
    }
    meta_file = tmp_path / f"{filename_stem}.sigmf-meta"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    return data_file


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------

class TestSigMFDtypeHandling:
    """Tests for all supported SigMF interleaved-complex datatypes."""

    def _make_signals(self) -> List[np.ndarray]:
        rng = np.random.default_rng(42)
        return [
            (rng.uniform(-0.5, 0.5, 64) + 1j * rng.uniform(-0.5, 0.5, 64)).astype(np.complex64)
            for _ in range(3)
        ]

    def test_cf32_le_roundtrip(self, tmp_path: Path) -> None:
        """cf32_le data reads back with near-exact precision."""
        signals = self._make_signals()
        _make_sigmf_pair(tmp_path, signals, 1e6, "cf32_le", "cap")
        reader = SigMFReader(root=str(tmp_path))
        for i, expected in enumerate(signals):
            got = reader.read(i)
            assert np.allclose(got.data, expected, atol=1e-6), (
                f"Signal {i} mismatch for cf32_le"
            )

    def test_ci16_le_roundtrip(self, tmp_path: Path) -> None:
        """ci16_le data divides by 32768 and returns complex64."""
        rng = np.random.default_rng(7)
        sig = (rng.uniform(-0.5, 0.5, 32) + 1j * rng.uniform(-0.5, 0.5, 32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "ci16_le", "cap")
        reader = SigMFReader(root=str(tmp_path))
        got = reader.read(0)
        assert got.data.dtype == np.complex64
        # Tolerate the rounding that occurs when scaling to int16 and back
        assert np.allclose(got.data, sig, atol=2.0 / 32768.0)

    def test_ci16_be_roundtrip(self, tmp_path: Path) -> None:
        """ci16_be big-endian data reads back correctly."""
        rng = np.random.default_rng(11)
        sig = (rng.uniform(-0.5, 0.5, 32) + 1j * rng.uniform(-0.5, 0.5, 32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "ci16_be", "cap")
        reader = SigMFReader(root=str(tmp_path))
        got = reader.read(0)
        assert got.data.dtype == np.complex64
        assert np.allclose(got.data, sig, atol=2.0 / 32768.0)

    def test_ci8_roundtrip(self, tmp_path: Path) -> None:
        """ci8 data divides by 128 and returns complex64."""
        rng = np.random.default_rng(13)
        sig = (rng.uniform(-0.5, 0.5, 32) + 1j * rng.uniform(-0.5, 0.5, 32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "ci8", "cap")
        reader = SigMFReader(root=str(tmp_path))
        got = reader.read(0)
        assert got.data.dtype == np.complex64
        assert np.allclose(got.data, sig, atol=2.0 / 128.0)

    def test_cu8_roundtrip(self, tmp_path: Path) -> None:
        """cu8 removes the 128-bias, divides by 128, and returns complex64."""
        rng = np.random.default_rng(17)
        # Keep values well within ±0.5 to avoid uint8 clipping
        sig = (rng.uniform(-0.4, 0.4, 32) + 1j * rng.uniform(-0.4, 0.4, 32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "cu8", "cap")
        reader = SigMFReader(root=str(tmp_path))
        got = reader.read(0)
        assert got.data.dtype == np.complex64
        assert np.allclose(got.data, sig, atol=2.0 / 128.0)

    def test_unsupported_dtype_raises(self, tmp_path: Path) -> None:
        """An unsupported datatype string raises ValueError at construction time."""
        # Manually write a meta file with an unsupported datatype
        data_file = tmp_path / "cap.sigmf-data"
        data_file.write_bytes(b"\x00" * 16)
        meta = {
            "global": {
                "core:datatype": "rf32_le",
                "core:sample_rate": 1e6,
                "core:version": "0.0.2",
            },
            "captures": [{"core:sample_start": 0}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": 4, "core:label": "x"}],
        }
        (tmp_path / "cap.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="rf32_le"):
            SigMFReader(root=str(tmp_path))


# ---------------------------------------------------------------------------
# Index behaviour
# ---------------------------------------------------------------------------

class TestSigMFReaderIndex:
    """Tests for annotation indexing, multi-file concatenation, and error paths."""

    def test_len_equals_annotation_count(self, tmp_path: Path) -> None:
        """Three annotations produce len == 3."""
        rng = np.random.default_rng(0)
        signals = [
            (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(np.complex64)
            for _ in range(3)
        ]
        _make_sigmf_pair(tmp_path, signals, 1e6, "cf32_le", "cap")
        reader = SigMFReader(root=str(tmp_path))
        assert len(reader) == 3

    def test_read_correct_samples(self, tmp_path: Path) -> None:
        """Annotation[1] returns the second signal, not the first (byte offset check)."""
        rng = np.random.default_rng(1)
        sig0 = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        sig1 = (rng.standard_normal(48) + 1j * rng.standard_normal(48)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig0, sig1], 1e6, "cf32_le", "cap")
        reader = SigMFReader(root=str(tmp_path))
        got = reader.read(1)
        assert np.allclose(got.data, sig1, atol=1e-6)

    def test_multiple_files_concatenated(self, tmp_path: Path) -> None:
        """Annotations from two .sigmf pairs are concatenated; total len is their sum."""
        rng = np.random.default_rng(2)
        signals_a = [(rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(np.complex64)]
        signals_b = [
            (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(np.complex64),
            (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(np.complex64),
        ]
        _make_sigmf_pair(tmp_path, signals_a, 1e6, "cf32_le", "fileA")
        _make_sigmf_pair(tmp_path, signals_b, 1e6, "cf32_le", "fileB")
        reader = SigMFReader(root=str(tmp_path))
        assert len(reader) == 3

    def test_sorted_file_order(self, tmp_path: Path) -> None:
        """Files are indexed in sorted (lexicographic) filename order."""
        sig_a = np.ones(16, dtype=np.complex64) * 0.1
        sig_z = np.ones(16, dtype=np.complex64) * 0.9
        # Create z_ first to ensure we're not relying on filesystem order
        _make_sigmf_pair(tmp_path, [sig_z], 1e6, "cf32_le", "z_capture")
        _make_sigmf_pair(tmp_path, [sig_a], 1e6, "cf32_le", "a_capture")
        reader = SigMFReader(root=str(tmp_path))
        # Index 0 must come from a_capture (sorted first)
        got = reader.read(0)
        assert np.allclose(got.data, sig_a, atol=1e-6)

    def test_no_sigmf_files_raises(self, tmp_path: Path) -> None:
        """Empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SigMFReader(root=str(tmp_path))

    def test_sample_count_zero_raises(self, tmp_path: Path) -> None:
        """An annotation with sample_count=0 raises ValueError on read()."""
        data_file = tmp_path / "cap.sigmf-data"
        data_file.write_bytes(b"\x00" * 32)
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 1e6,
                "core:version": "0.0.2",
            },
            "captures": [{"core:sample_start": 0}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": 0, "core:label": "bpsk"}],
        }
        (tmp_path / "cap.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
        reader = SigMFReader(root=str(tmp_path))
        with pytest.raises(ValueError, match="sample_count=0"):
            reader.read(0)


# ---------------------------------------------------------------------------
# Metadata mapping
# ---------------------------------------------------------------------------

class TestSigMFReaderMetadata:
    """Tests for Signal metadata fields populated from SigMF global and annotation fields."""

    def _one_signal(self, tmp_path: Path, extra: Optional[Dict] = None) -> SigMFReader:
        """Create a single-annotation reader with label 'BPSK'."""
        rng = np.random.default_rng(99)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(
            tmp_path,
            [sig],
            2.4e6,
            "cf32_le",
            "cap",
            extra_annotations=[extra] if extra else None,
        )
        return SigMFReader(root=str(tmp_path))

    def test_sample_rate_in_signal(self, tmp_path: Path) -> None:
        """Global core:sample_rate is stored in Signal.sample_rate."""
        rng = np.random.default_rng(20)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 3.2e6, "cf32_le", "cap")
        reader = SigMFReader(root=str(tmp_path))
        signal = reader.read(0)
        assert signal["sample_rate"] == pytest.approx(3.2e6)

    def test_class_name_from_label(self, tmp_path: Path) -> None:
        """core:label is lower-cased and stored as class_name."""
        reader = self._one_signal(tmp_path)
        signal = reader.read(0)
        assert signal["class_name"] == "bpsk"

    def test_class_index_assigned(self, tmp_path: Path) -> None:
        """Sorted unique labels receive deterministic integer indices."""
        rng_gen = np.random.default_rng(30)
        sigs = [(rng_gen.standard_normal(16) + 1j * rng_gen.standard_normal(16)).astype(np.complex64) for _ in range(2)]
        # Two different labels: "qpsk" (index 1) and "bpsk" (index 0) when sorted
        meta = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6, "core:version": "0.0.2"},
            "captures": [{"core:sample_start": 0}],
            "annotations": [
                {"core:sample_start": 0,  "core:sample_count": 16, "core:label": "qpsk"},
                {"core:sample_start": 16, "core:sample_count": 16, "core:label": "bpsk"},
            ],
        }
        interleaved = np.empty(len(sigs) * 32, dtype="<f4")
        for i, s in enumerate(sigs):
            interleaved[i * 32: i * 32 + 32: 2] = s.real
            interleaved[i * 32 + 1: i * 32 + 32: 2] = s.imag
        (tmp_path / "cap.sigmf-data").write_bytes(
            np.stack([s.view(np.float32) for s in sigs]).ravel().astype("<f4").tobytes()
        )
        # Write proper interleaved data for two signals
        all_bytes = b""
        for sig in sigs:
            interleaved = np.empty(len(sig) * 2, dtype="<f4")
            interleaved[0::2] = sig.real
            interleaved[1::2] = sig.imag
            all_bytes += interleaved.tobytes()
        (tmp_path / "cap.sigmf-data").write_bytes(all_bytes)
        (tmp_path / "cap.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
        reader = SigMFReader(root=str(tmp_path))
        qpsk_signal = reader.read(0)
        bpsk_signal = reader.read(1)
        # Sorted: ["bpsk", "qpsk"] → bpsk=0, qpsk=1
        assert bpsk_signal["class_index"] == 0  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        assert qpsk_signal["class_index"] == 1

    def test_unknown_label_normalized(self, tmp_path: Path) -> None:
        """Missing core:label produces class_name 'unknown'."""
        data_file = tmp_path / "cap.sigmf-data"
        rng = np.random.default_rng(40)
        sig = (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(np.complex64)
        interleaved = np.empty(32, dtype="<f4")
        interleaved[0::2] = sig.real
        interleaved[1::2] = sig.imag
        data_file.write_bytes(interleaved.tobytes())
        meta = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6, "core:version": "0.0.2"},
            "captures": [{"core:sample_start": 0}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": 16}],
        }
        (tmp_path / "cap.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
        reader = SigMFReader(root=str(tmp_path))
        signal = reader.read(0)
        assert signal["class_name"] == "unknown"

    def test_center_freq_from_captures(self, tmp_path: Path) -> None:
        """core:frequency from captures is stored as center_freq in Signal."""
        rng = np.random.default_rng(50)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(
            tmp_path, [sig], 1e6, "cf32_le", "cap",
            captures=[{"core:sample_start": 0, "core:frequency": 915e6}],
        )
        reader = SigMFReader(root=str(tmp_path))
        signal = reader.read(0)
        assert signal["center_freq"] == pytest.approx(915e6)

    def test_custom_fields_sanitized(self, tmp_path: Path) -> None:
        """Non-core annotation keys have ':' replaced with '_' in Signal metadata."""
        rng = np.random.default_rng(60)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(
            tmp_path, [sig], 1e6, "cf32_le", "cap",
            extra_annotations=[{"lora:sf": 7}],
        )
        reader = SigMFReader(root=str(tmp_path))
        signal = reader.read(0)
        assert signal["lora_sf"] == 7

    def test_meta_naming_convention_b(self, tmp_path: Path) -> None:
        """'foo.sigmf-data.sigmf-meta' sidecar naming convention is discovered."""
        rng = np.random.default_rng(70)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        interleaved = np.empty(64, dtype="<f4")
        interleaved[0::2] = sig.real
        interleaved[1::2] = sig.imag
        (tmp_path / "capture.sigmf-data").write_bytes(interleaved.tobytes())
        meta = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6, "core:version": "0.0.2"},
            "captures": [{"core:sample_start": 0}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": 32, "core:label": "bpsk"}],
        }
        # Convention B: foo.sigmf-data.sigmf-meta
        (tmp_path / "capture.sigmf-data.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
        reader = SigMFReader(root=str(tmp_path))
        assert len(reader) == 1
        signal = reader.read(0)
        assert signal["class_name"] == "bpsk"

    def test_no_sigmf_library_imported(self, tmp_path: Path) -> None:
        """SigMFReader must not import the 'sigmf' third-party library (zero-dependency contract)."""
        # Confirm the optional 'sigmf' package is not pulled in at test-collection time
        assert "sigmf" not in sys.modules, (
            "'sigmf' was already in sys.modules before this test ran; "
            "SigMFReader's zero-dependency contract may be broken."
        )
        rng = np.random.default_rng(75)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "cf32_le", "cap")
        reader = SigMFReader(root=str(tmp_path))
        reader.read(0)
        assert "sigmf" not in sys.modules, (
            "Constructing or reading from SigMFReader caused the 'sigmf' library to be imported; "
            "the implementation must remain zero-dependency."
        )


# ---------------------------------------------------------------------------
# StaticTorchSigDataset integration
# ---------------------------------------------------------------------------

class TestSigMFStaticDatasetIntegration:
    """Integration tests for StaticTorchSigDataset with SigMFReader as backend."""

    def _create_dataset(self, tmp_path: Path) -> StaticTorchSigDataset:
        """Create a StaticTorchSigDataset backed by a single SigMF pair."""
        rng = np.random.default_rng(80)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "cf32_le", "cap")
        return StaticTorchSigDataset(
            root=str(tmp_path),
            file_handler_class=SigMFReader,
            target_labels=["class_name"],
        )

    def test_static_dataset_getitem(self, tmp_path: Path) -> None:
        """StaticTorchSigDataset[0] returns a (np.ndarray, label) tuple."""
        dataset = self._create_dataset(tmp_path)
        item = dataset[0]
        assert isinstance(item, tuple)
        data, label = item
        assert isinstance(data, np.ndarray)
        assert label == "bpsk"

    def test_static_dataset_len(self, tmp_path: Path) -> None:
        """len(StaticTorchSigDataset) matches the underlying reader length."""
        dataset = self._create_dataset(tmp_path)
        reader = SigMFReader(root=str(tmp_path))
        assert len(dataset) == len(reader)

    def test_transforms_applied(self, tmp_path: Path) -> None:
        """ComplexTo2D transform produces a (2, N) shaped output array."""
        rng = np.random.default_rng(85)
        sig = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "cf32_le", "cap")
        dataset = StaticTorchSigDataset(
            root=str(tmp_path),
            file_handler_class=SigMFReader,
            target_labels=["class_name"],
            transforms=[ComplexTo2D()],
        )
        data, label = dataset[0]
        assert data.shape[0] == 2, (
            f"Expected 2 channels after ComplexTo2D, got shape {data.shape}"
        )
        assert label == "bpsk"


# ---------------------------------------------------------------------------
# SigMFFileHandler
# ---------------------------------------------------------------------------

class TestSigMFFileHandler:
    """Tests for the SigMFFileHandler factory."""

    def test_create_handler_reader_mode(self, tmp_path: Path) -> None:
        """create_handler('r', root) returns a SigMFReader."""
        rng = np.random.default_rng(90)
        sig = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex64)
        _make_sigmf_pair(tmp_path, [sig], 1e6, "cf32_le", "cap")
        handler = SigMFFileHandler.create_handler("r", str(tmp_path))
        assert isinstance(handler, SigMFReader)

    def test_create_handler_write_raises(self, tmp_path: Path) -> None:
        """create_handler('w', root) raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SigMFFileHandler.create_handler("w", str(tmp_path))
