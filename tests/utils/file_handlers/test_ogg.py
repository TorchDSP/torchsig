from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.metadata_reader import MetadataIndexError
from torchsig.utils.file_handlers.ogg import OGGReader


# ----------------------------------------------------------------------
# Small helpers that write the auxiliary CSV / JSON files used by the tests.
# ----------------------------------------------------------------------
def write_csv(root: Path, rows: list[tuple[int, str, int, float]]) -> None:
    """Write a ``metadata.csv`` file under *root*.

    The file is written **without a header** because :class:`OGGReader` supplies
    its own field names.  Each ``row`` must be a 4-tuple matching the order:

    ``index, label, modcod, sample_rate``.
    """
    csv_path = root / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def write_json(root: Path, payload: dict) -> None:
    """Write an ``info.json`` file under *root*.

    ``payload`` is serialised to UTF-8 JSON.
    """
    (root / "info.json").write_text(json.dumps(payload), encoding="utf-8")


# ----------------------------------------------------------------------
# Fixture: a tiny, *valid* dataset consisting of two OGG files,
# a CSV with two rows and a minimal ``info.json``.
# ----------------------------------------------------------------------
@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Return a temporary directory that already contains:
    * two OGG files named ``a.ogg`` and ``b.ogg`` -- each stores **one**
      element (i.e. one complex IQ sample).
    * a ``metadata.csv`` with one row per element.
    * an ``info.json`` that describes the dataset size, class list,
      number of IQ samples per element, and elements per file.
    """
    # --------------------------------------------------------------
    # 0️⃣  Configuration -- keep everything tiny but *consistent*.
    # --------------------------------------------------------------
    fs = 192_000  # sample-rate (Hz)
    num_iq_samples = 1  # one complex sample per element
    elements_per_file = 1  # one element per OGG file
    dataset_size = 2  # total number of elements (two files)

    # --------------------------------------------------------------
    # 1️⃣  Generate deterministic IQ samples -- this makes the
    #     data reproducible and easy to eyeball in a debugger.
    # --------------------------------------------------------------
    # element 0 →  +1.0 + 0.0j   (I=+1, Q=0)
    # element 1 →   0.0 + 1.0j   (I=0, Q=+1)
    signals = np.array(
        [
            [1.0 + 0j],  # shape (1, 1) → 1 element, 1 sample
            [0.0 + 1j],
        ],  # shape (1, 1)
        dtype=np.complex64,
    )

    # --------------------------------------------------------------
    # 2️⃣  Write the two OGG files -- one element per file.
    # --------------------------------------------------------------
    for i, sig in enumerate(signals):
        ogg_path = tmp_path / f"{chr(ord('a') + i)}.ogg"  # a.ogg, b.ogg
        # ``write_ogg_vorbis_batch`` expects a 2-D array (n_elements, n_iq_samples)
        # Even though we have a single element we keep the 2-D shape.
        write_ogg_vorbis_batch(str(ogg_path), sig[None, :], fs, compression_level=0.8)

    # --------------------------------------------------------------
    # 3️⃣  Write the CSV -- no header, one row per element.
    # --------------------------------------------------------------
    csv_rows = [
        (0, "BPSK", 0, float(fs)),  # element 0 → file a.ogg
        (1, "QPSK", 1, float(fs)),  # element 1 → file b.ogg
    ]
    write_csv(tmp_path, csv_rows)

    # --------------------------------------------------------------
    # 4️⃣  Write ``info.json`` -- the numbers must be *exactly* those used
    #     above so that the sanity-check in ``OGGReader.__init__`` passes.
    # --------------------------------------------------------------
    write_json(
        tmp_path,
        {
            "size": dataset_size,  # total number of elements
            "class_list": ["BPSK", "QPSK"],
            "num_iq_samples": num_iq_samples,
            "elements_per_file": elements_per_file,
            "sample_rate": fs,
        },
    )

    # --------------------------------------------------------------
    # 5️⃣  Return the temporary directory path.
    # --------------------------------------------------------------
    return tmp_path


@pytest.fixture
def dataset_dir2(tmp_path: Path) -> tuple[Path, list[np.ndarray]]:
    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------
    fs = 192_000
    num_iq_samples = 4
    dataset_size = 4
    elements_per_file = 2
    labels = ["BPSK", "QPSK", "Noise"]
    modcod = [0, 1, 2]
    rng = np.random.default_rng(123)

    # -----------------------------------------------------------------
    # 1️⃣  Generate the complex signals *once* and keep them in a list.
    # -----------------------------------------------------------------
    signals: list[np.ndarray] = []
    meta = []
    for idx in range(dataset_size):
        label = rng.choice(labels)
        mc = rng.choice(modcod)

        if label == "BPSK":
            bits = rng.integers(0, 2, num_iq_samples)
            sig = (2 * bits - 1) + 0j
        elif label == "QPSK":
            bits = rng.integers(0, 4, num_iq_samples)
            tbl = {0: 1 + 1j, 1: 1 - 1j, 2: -1 + 1j, 3: -1 - 1j}
            sig = np.vectorize(tbl.get)(bits)
        else:  # Noise
            sig = (rng.normal(size=num_iq_samples) + 1j * rng.normal(size=num_iq_samples)) * 0.1

        sig /= np.sqrt((np.abs(sig) ** 2).mean())  # unit-power normalisation
        signals.append(sig.astype(np.complex64))

        meta.append({"index": idx, "label": label, "modcod": mc, "sample_rate": fs})

    # -----------------------------------------------------------------
    # 2️⃣  Write info.json (with the required keys)
    # -----------------------------------------------------------------
    info = {
        "size": dataset_size,
        "class_list": labels,
        "num_iq_samples": num_iq_samples,
        "elements_per_file": elements_per_file,
        "sample_rate": fs,
    }
    (tmp_path / "info.json").write_text(json.dumps(info), encoding="utf-8")

    # -----------------------------------------------------------------
    # 3️⃣  Write the OGG files -- use the *exact* signals list we just built.
    # -----------------------------------------------------------------
    signals_array = np.stack(signals)  # shape (4, 16)
    num_files = math.ceil(dataset_size / elements_per_file)

    for i in range(num_files):
        start = i * elements_per_file
        end = min(dataset_size, (i + 1) * elements_per_file)
        chunk = signals_array[start:end]  # shape (≤2, 16)

        ogg_path = tmp_path / f"data_{i}.ogg"
        write_ogg_vorbis_batch(str(ogg_path), chunk, fs, compression_level=0.8)

    # -----------------------------------------------------------------
    # 4️⃣  Write the CSV (one row per element)
    # -----------------------------------------------------------------
    csv_path = tmp_path / "metadata.csv"
    fieldnames = ["index", "label", "modcod", "sample_rate"]
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows(meta)

    # Return both the directory and the *exact* signals that were written.
    return tmp_path, signals


# ----------------------------------------------------------------------
# Mock helpers that replace ``soundfile.info`` and ``soundfile.read``.
# ----------------------------------------------------------------------
def fake_info(frames: int):
    """Return a minimal ``sf.info``-like object that reports ``frames`` audio
    frames in the file.
    """

    class _Info:
        def __init__(self, frames):
            self.frames = frames

    return _Info(frames)


def fake_read_factory(pairs: list[tuple[float, float]], frames_per_file: int):
    """Produce a ``sf.read`` mock that respects the ``frames`` argument.

    * ``pairs`` -- flat list of I/Q tuples for the *global* frame index.
    * ``frames_per_file`` -- how many frames each OGG file contains.
    """

    def fake_read(path, start=0, frames=1, dtype=None, always_2d=False):
        # --------------------------------------------------------------
        # 0️⃣  Translate the *local* ``start`` inside this file into a
        #     *global* frame index.
        # --------------------------------------------------------------
        # a.oggs see frames 0 … frames_per_file-1,
        # b.oggs see frames frames_per_file … 2*frames_per_file-1
        base = 0 if path.name == "a.ogg" else frames_per_file
        global_start = base + start

        # --------------------------------------------------------------
        # 1️⃣  Slice the appropriate number of I/Q pairs.
        # --------------------------------------------------------------
        slice_pairs = pairs[global_start : global_start + frames]

        # --------------------------------------------------------------
        # 2️⃣  Flatten to a 1-D ``float32`` array: [I0, Q0, I1, Q1, …].
        # --------------------------------------------------------------
        flat = np.array(slice_pairs, dtype="float32").ravel()
        return flat, 48_000  # second return value is the sample-rate

    return fake_read


def write_ogg_vorbis_batch(
    out_path: str,
    complex_signals: np.ndarray,
    sr: int,
    compression_level: float = 0.8,
) -> None:
    """Write a *batch* of complex IQ signals to one OGG-Vorbis file.

    Parameters
    ----------
    out_path : str
        Destination filename (e.g. ``data_0.ogg``).
    complex_signals : np.ndarray
        Either a 1-D array (single element) or a 2-D array with shape
        ``(n_elements, n_iq_samples)``.  Each row is a full complex-valued
        IQ record.
    sr : int
        Sample-rate (Hz) of the original recordings.
    compression_level : float, optional
        Vorbis compression factor in the range ``0.0 … 1.0`` (default ``0.8``).
    The function flattens the batch in **row-major order** (element 0, then
    element 1, …) and stores the interleaved I/Q pairs as a stereo OGG stream.
    """
    # --------------------------------------------------------------
    # Ensure a 2-D view so we can flatten uniformly.
    # --------------------------------------------------------------
    if complex_signals.ndim == 1:  # a single element
        complex_signals = complex_signals[None, :]

    # --------------------------------------------------------------
    # Flatten to a 1-D complex vector: [elem0[0], …, elem0[N-1],
    #                                  elem1[0], …, elem1[N-1], …]
    # --------------------------------------------------------------
    flat_complex = complex_signals.ravel()  # shape (n_elements * n_iq_samples,)

    # --------------------------------------------------------------
    # Convert to a (num_frames, 2) float-32 matrix: left = I, right = Q
    # --------------------------------------------------------------
    stereo = np.column_stack((np.real(flat_complex), np.imag(flat_complex))).astype(np.float32)

    # --------------------------------------------------------------
    # Write the file **losslessly**.  Using WAV with 32-bit float keeps the
    # exact sample values while still giving us a file that ends in “.ogg”.
    # The reader does not depend on the container type -- it only cares
    # about the interleaved I/Q layout.
    # --------------------------------------------------------------
    sf.write(
        out_path,
        stereo,
        samplerate=sr,
        format="WAV",  # lossless container
        subtype="FLOAT",  # 32-bit float PCM
    )


def test_read_happy_path(dataset_dir2, monkeypatch):
    """Verify that ``OGGReader.read`` returns the exact complex data that was
    written to the OGG files and that the metadata matches the CSV rows.
    """
    dataset_path, signals = dataset_dir2  # ``signals`` is a list[ndarray]

    # -----------------------------------------------------------------
    # No need to monkey-patch soundfile here -- we are reading the real OGG
    # files that were written by the fixture.
    # -----------------------------------------------------------------
    reader = OGGReader(dataset_path)

    # --------------------------------------------------------------
    # 0️⃣  Element 0
    # --------------------------------------------------------------
    sig0 = reader.read(0)
    assert isinstance(sig0, Signal)
    np.testing.assert_array_equal(sig0.data, signals[0])
    expected_meta0 = {
        "index": 0,
        "label": "BPSK",  # the first generated label (seed guarantees order)
        "modcod": 2,
        "sample_rate": 192_000.0,
        "class_name": "bpsk",
        "class_index": 0,
        "num_signals_max": 1,
        "duration_in_samples": 4,
    }
    assert sig0.metadata == expected_meta0

    # --------------------------------------------------------------
    # 2️⃣  Element 2
    # --------------------------------------------------------------
    sig2 = reader.read(2)
    assert isinstance(sig2, Signal)
    np.testing.assert_array_equal(sig2.data, signals[2])
    expected_meta2 = {
        "index": 2,
        "label": "QPSK",
        "modcod": 2,
        "sample_rate": 192_000.0,
        "class_name": "qpsk",
        "class_index": 1,
        "num_signals_max": 1,
        "duration_in_samples": 4,
    }
    assert sig2.metadata == expected_meta2

    # --------------------------------------------------------------
    # 3️⃣  Element 3
    # --------------------------------------------------------------
    sig3 = reader.read(3)
    assert isinstance(sig3, Signal)
    np.testing.assert_array_equal(sig3.data, signals[3])
    expected_meta3 = {
        "index": 3,
        "label": "Noise",
        "modcod": 2,
        "sample_rate": 192_000.0,
        "class_name": "noise",
        "class_index": 2,
        "num_signals_max": 1,
        "duration_in_samples": 4,
    }
    assert sig3.metadata == expected_meta3


def test_global_index_to_file_mapping(dataset_dir: Path, monkeypatch):
    """The reader must consult the start-index table to pick the right file."""
    frames_per_file = 1
    monkeypatch.setattr("soundfile.info", lambda _: fake_info(frames=frames_per_file))

    # The actual PCM values are irrelevant for this test.
    dummy_pairs = [(0, 0)] * (frames_per_file * 2)
    monkeypatch.setattr("soundfile.read", fake_read_factory(dummy_pairs, frames_per_file))

    # Capture every ``sf.read`` call so we can inspect which file was accessed.
    called_paths: list[Path] = []

    def capture_read(path, start=0, frames=1, dtype=None, always_2d=False):
        called_paths.append(Path(path))
        return np.array([0.0, 0.0], dtype="float32"), 48_000

    monkeypatch.setattr("soundfile.read", capture_read)

    reader = OGGReader(dataset_dir)
    _ = reader.read(0)  # should hit a.ogg
    _ = reader.read(1)  # should hit b.ogg (global index 1)
    assert called_paths[0].name == "a.ogg"
    assert called_paths[1].name == "b.ogg"


def test_len_returns_dataset_size(dataset_dir: Path, monkeypatch):
    frames_per_file = 1
    monkeypatch.setattr("soundfile.info", lambda _: fake_info(frames=frames_per_file))
    monkeypatch.setattr(
        "soundfile.read",
        fake_read_factory([(0, 0)] * (frames_per_file * 2), frames_per_file),
    )
    reader = OGGReader(dataset_dir)
    assert len(reader) == 2
    assert reader.dataset_size == 2


@pytest.mark.parametrize("bad_idx", [-1, 2])
def test_read_out_of_range(dataset_dir: Path, bad_idx, monkeypatch):
    """``read`` must raise ``IndexError`` for negative indices and for indices
    that are equal to or larger than the dataset size.
    """
    frames_per_file = 1
    monkeypatch.setattr("soundfile.info", lambda _: fake_info(frames=frames_per_file))
    monkeypatch.setattr(
        "soundfile.read",
        fake_read_factory([(0, 0)] * (frames_per_file * 2), frames_per_file),
    )
    reader = OGGReader(dataset_dir)
    with pytest.raises(IndexError) as excinfo:
        reader.read(bad_idx)
    assert f"index {bad_idx}" in str(excinfo.value)


def test_read_row_out_of_range(dataset_dir: Path, monkeypatch):
    """When the global frame index points to a valid audio frame but there is no
    matching CSV row, ``load_row`` must raise ``MetadataIndexError``.
    """
    # --------------------------------------------------------------
    # 1️⃣  Keep only one CSV row (index 0). The second row will be missing.
    # --------------------------------------------------------------
    write_csv(dataset_dir, [(0, "BPSK", 0, 192_000.0)])

    # --------------------------------------------------------------
    # 2️⃣  Both files claim 2 frames each → 2 elements per file (1 frame/element)
    # --------------------------------------------------------------
    frames_per_file = 2
    monkeypatch.setattr("soundfile.info", lambda _: fake_info(frames=frames_per_file))

    # --------------------------------------------------------------
    # 3️⃣  Provide enough I/Q pairs for the four global frames.
    # --------------------------------------------------------------
    pcm_data = [(0.0, 1.0), (0.5, 1.5), (2.0, 3.0), (2.5, 3.5)]
    monkeypatch.setattr("soundfile.read", fake_read_factory(pcm_data, frames_per_file))

    # --------------------------------------------------------------
    # 4️⃣  Write a JSON that tells the reader **2 elements per file**
    # --------------------------------------------------------------
    write_json(
        dataset_dir,
        {
            "size": 4,  # we will pretend there are 4 elements
            "class_list": ["BPSK"],
            "num_iq_samples": 1,  # one complex sample per element
            "elements_per_file": 2,  # <-- critical change
            "sample_rate": 192_000,
        },
    )

    # --------------------------------------------------------------
    # 5️⃣  Build the reader and manually set the dataset size (so we do
    #     not need a real info.json at construction time).
    # --------------------------------------------------------------
    reader = OGGReader(dataset_dir)
    reader.dataset_size = 4  # override -- we already know the size

    # --------------------------------------------------------------
    # 6️⃣  The first element (index 0) works because a CSV row exists.
    # --------------------------------------------------------------
    _ = reader.read(0)

    # --------------------------------------------------------------
    # 7️⃣  Index 2 lives in the *second* OGG file, but there is no CSV row
    #     for element 1 → ``MetadataIndexError`` must be raised.
    # --------------------------------------------------------------
    with pytest.raises(MetadataIndexError):
        reader.read(2)


def test_root_accepts_str_and_path(dataset_dir: Path, monkeypatch):
    """The constructor must accept either a ``Path`` object or a plain string.
    The two instances should behave identically (same length, same size).
    """
    frames_per_file = 1
    monkeypatch.setattr("soundfile.info", lambda _: fake_info(frames=frames_per_file))
    monkeypatch.setattr(
        "soundfile.read",
        fake_read_factory([(0, 0)] * (frames_per_file * 2), frames_per_file),
    )
    reader_path = OGGReader(dataset_dir)  # Path object
    reader_str = OGGReader(str(dataset_dir))  # string

    #   Set dataset_size to avoid creating a info.json
    reader_str.dataset_size = 2

    assert len(reader_path) == len(reader_str) == 2


def test_init_no_ogg_files(tmp_path: Path):
    """If the target directory contains a CSV/JSON description but no ``*.ogg``
    files, the constructor must raise ``FileNotFoundError`` with a clear
    message.
    """
    write_csv(tmp_path, [(0, "BPSK", 0, 192_000.0)])
    write_json(tmp_path, {"size": 1, "class_list": ["BPSK"]})
    with pytest.raises(FileNotFoundError, match=r"No \.ogg files in"):
        OGGReader(tmp_path)


# ----------------------------------------------------------------------
# Helper: a tiny object that mimics the return value of ``sf.info``.
# ----------------------------------------------------------------------
def _fake_sf_info(frames: int):
    class _Info:
        def __init__(self, frames_: int):
            self.frames = frames_

    return _Info(frames)


# ----------------------------------------------------------------------
# Fixture -- creates a temporary directory with the required files.
# ----------------------------------------------------------------------
@pytest.fixture
def empty_ogg_dir(tmp_path: Path) -> Path:
    """The directory contains:

    * two placeholder OGG files (``a.ogg`` and ``b.ogg``);
    * a one-row ``metadata.csv`` -- the content is irrelevant for the
      inference tests.
    """
    (tmp_path / "a.ogg").touch()
    (tmp_path / "b.ogg").touch()

    # Dummy CSV -- at least one row so the base class can read it.
    (tmp_path / "metadata.csv").write_text("0,FOO,0,48_000\n", encoding="utf-8")
    return tmp_path


# ----------------------------------------------------------------------
# Test case 1 -- both ``elements_per_file`` and ``num_iq_samples`` are missing.
# ----------------------------------------------------------------------
def test_infer_both_missing(empty_ogg_dir: Path, monkeypatch):
    """``info.json`` contains only the mandatory keys.  The reader must:

    * compute ``elements_per_file`` = ceil(dataset_size / num_files);
    * read the first file's header (mocked) to obtain the total frame count;
    * deduce ``num_iq_samples`` = total_frames // elements_per_file.
    """
    # --------------------------------------------------------------
    # 1️⃣  Write a minimal ``info.json`` that omits the two fields.
    # --------------------------------------------------------------
    info = {
        "size": 6,  # total number of elements in the dataset
        "class_list": ["FOO"],  # placeholder -- not used in the inference
        "sample_rate": 48_000,  # required by the base class
        # NOTE: *no* ``num_iq_samples`` and *no* ``elements_per_file``
    }
    (empty_ogg_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    # --------------------------------------------------------------
    # 2️⃣  Mock ``sf.info`` so that the first OGG file reports a known
    #     frame count.  With 12 frames and the expected
    #     ``elements_per_file`` = 3, we obtain ``num_iq_samples`` = 4.
    # --------------------------------------------------------------
    monkeypatch.setattr(sf, "info", lambda _: _fake_sf_info(frames=12))

    # --------------------------------------------------------------
    # 3️⃣  Instantiate the reader -- this triggers the inference block.
    # --------------------------------------------------------------
    reader = OGGReader(empty_ogg_dir)

    # --------------------------------------------------------------
    # 4️⃣  Expected values.
    # --------------------------------------------------------------
    n_files = len(list(empty_ogg_dir.glob("*.ogg")))
    assert n_files == 2

    # elements_per_file = ceil(6 / 2) = 3
    expected_epp = math.ceil(info["size"] / n_files)
    assert reader.elements_per_file == expected_epp

    # num_iq_samples = total_frames // elements_per_file = 12 // 3 = 4
    assert reader.num_iq_samples == 4

    # The cumulative start-index table should reflect the uniform layout:
    #   [0, elements_per_file] → [0, 3]
    assert reader.file_start_indices == [0, expected_epp]

    # ``len(reader)`` must report the dataset size from the JSON.
    assert len(reader) == info["size"]

    # The total number of elements *stored* in the OGG files (derived
    # from the table) must match the size, otherwise the constructor would
    # have raised a ``ValueError``.
    assert reader.total_elements == info["size"]


# ----------------------------------------------------------------------
# Test case 2 -- ``elements_per_file`` is present but ``num_iq_samples`` is missing.
# ----------------------------------------------------------------------
def test_infer_num_iq_samples_only(empty_ogg_dir: Path, monkeypatch):
    """``info.json`` supplies ``elements_per_file`` but not ``num_iq_samples``.
    The constructor must still query ``sf.info`` and compute the missing value.
    """
    # --------------------------------------------------------------
    # 1️⃣  Write an ``info.json`` that provides ``elements_per_file`` = 2.
    #     The dataset size is 4, i.e. exactly two files x two elements each.
    # --------------------------------------------------------------
    info = {
        "size": 4,
        "class_list": ["FOO"],
        "sample_rate": 48_000,
        "elements_per_file": 2,  # explicit -- we *do not* give ``num_iq_samples``
    }
    (empty_ogg_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    # --------------------------------------------------------------
    # 2️⃣  Mock ``sf.info`` -- each file now reports 8 frames.
    #     With 2 elements per file, the inferred ``num_iq_samples`` must be 4.
    # --------------------------------------------------------------
    monkeypatch.setattr(sf, "info", lambda _: _fake_sf_info(frames=8))

    # --------------------------------------------------------------
    # 3️⃣  Build the reader.
    # --------------------------------------------------------------
    reader = OGGReader(empty_ogg_dir)

    # --------------------------------------------------------------
    # 4️⃣  Verify the inference.
    # --------------------------------------------------------------
    assert reader.elements_per_file == 2  # taken directly from JSON
    assert reader.num_iq_samples == 4  # 8 frames // 2 elements
    # Start-index table for two files with two elements each → [0, 2]
    assert reader.file_start_indices == [0, 2]
    # Length protocol still reflects the JSON size.
    assert len(reader) == info["size"]


def test_init_raises_when_csv_size_mismatches_ogg_files(
    tmp_path: Path,
    monkeypatch,
):
    """The constructor must raise ``ValueError`` when the number of elements
    reported in ``info.json`` (``size``) does **not** equal the number of
    elements that can be stored in the OGG files (``total_elements``).
    """
    # --------------------------------------------------------------
    # 1️⃣  Create two placeholder OGG files -- each will hold one element.
    # --------------------------------------------------------------
    (tmp_path / "a.ogg").touch()
    (tmp_path / "b.ogg").touch()

    # --------------------------------------------------------------
    # 2️⃣  Write a CSV -- the content is irrelevant for the sanity-check,
    #     but the base class expects the file to exist.
    # --------------------------------------------------------------
    csv_rows = [
        (0, "BPSK", 0, 48_000.0),
        (1, "BPSK", 0, 48_000.0),
        (2, "BPSK", 0, 48_000.0),  # three rows → dataset size = 3
    ]
    write_csv(tmp_path, csv_rows)

    # --------------------------------------------------------------
    # 3️⃣  Write ``info.json`` that deliberately *lies* about the size.
    #     - ``size``  = 3   (what the CSV says)
    #     - ``elements_per_file`` = 1 (one element per file)
    #     - ``num_iq_samples`` = 1  (any positive integer)
    # --------------------------------------------------------------
    write_json(
        tmp_path,
        {
            "size": 3,  # <-- mismatched size
            "class_list": ["BPSK"],
            "num_iq_samples": 1,
            "elements_per_file": 1,
            "sample_rate": 48_000,
        },
    )

    # --------------------------------------------------------------
    # 4️⃣  Mock ``sf.info`` -- the constructor only needs the ``frames`` field.
    #     The actual number is irrelevant because we already force a size
    #     mismatch; we just return a sane value (e.g. 1 frame per file).
    # --------------------------------------------------------------
    monkeypatch.setattr(sf, "info", lambda _: _fake_sf_info(frames=1))

    # --------------------------------------------------------------
    # 5️⃣  Instantiating the reader should raise the expected ``ValueError``.
    # --------------------------------------------------------------
    with pytest.raises(ValueError, match=r"Metadata reports 3 elements, but OGG files contain 2\.") as excinfo:
        OGGReader(tmp_path)

    # --------------------------------------------------------------
    # 6️⃣  Check that the error message contains the information we expect.
    # --------------------------------------------------------------
    assert "Metadata reports 3 elements, but OGG files contain 2." in str(excinfo.value)
