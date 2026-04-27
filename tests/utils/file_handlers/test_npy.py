"""Unit tests for :class:`torchsig.utils.file_handlers.npy.NPYReader`."""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import builtins
import json
from pathlib import Path
import csv

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import numpy as np
import pytest

# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------
from torchsig.utils.file_handlers.npy import NPYReader

# Default sample‑rate used throughout the module (Hz)
DEFAULT_SR = 192_000


# ----------------------------------------------------------------------
# Helper – create a minimal, one‑sample dataset
# ----------------------------------------------------------------------
def _make_one_sample_dataset(root: Path) -> Path:
    """
    Create a single‑sample ``*.npy`` file and a trivial ``metadata.csv``.

    The CSV contains a single row; its contents are irrelevant for the
    ``info.json``‑related tests, but the file must exist for the
    :func:`torchsig.utils.file_handlers.metadata_loader.load_row` helper.

    Parameters
    ----------
    root:
        Directory in which the artefacts are written.

    Returns
    -------
    Path
        Path to the generated ``metadata.csv`` (useful for debugging).
    """
    # ------------------------------------------------------------------
    # 1️⃣  Write a one‑element NumPy array (complex64) – the actual value
    #     does not matter for the JSON‑error tests.
    # ------------------------------------------------------------------
    np.save(root / "data_0.npy", np.array([0 + 0j], dtype=np.complex64))

    # ------------------------------------------------------------------
    # 2️⃣  Write a CSV with a single row.  ``newline=''`` guarantees
    #     platform‑independent line endings (important on Windows).
    # ------------------------------------------------------------------
    csv_path = root / "metadata.csv"
    csv_path.write_text(
        f"0,BPSK,0,{DEFAULT_SR}\n", encoding="utf-8", newline=""
    )
    return csv_path


# ----------------------------------------------------------------------
# Helper – write an ``info.json`` file required by :class:`NPYReader`
# ----------------------------------------------------------------------
def write_info_json(root: Path, size: int) -> Path:
    """
    Write a minimal ``info.json`` containing only the ``size`` key.

    Parameters
    ----------
    root:
        Directory in which ``info.json`` is created.
    size:
        The advertised number of samples (this is what ``len(reader)`` will
        return).  It may deliberately differ from the real number of samples
        to test out-of-bounds handling.

    Returns
    -------
    Path
        The path to the created ``info.json`` file.
    """
    info_path = root / "info.json"
    info_path.write_text(json.dumps({"size": size}), encoding="utf-8")
    return info_path


# ----------------------------------------------------------------------
# Helper – creates a minimal dataset with a controllable CSV
# ----------------------------------------------------------------------
def create_dataset(
    tmp_dir: pathlib.Path,
    *,
    npy_shapes: list[int],
    csv_rows: list[tuple[int, str, int, int]],
    advertised_size: int | None = None,
) -> pathlib.Path:
    """
    Populate *tmp_dir* with:

    * ``*.npy`` files matching ``npy_shapes`` (list of per‑file sample counts).
    * A ``metadata.csv`` containing the supplied rows.
    * An ``info.json`` whose ``size`` field equals ``advertised_size`` if
      provided, otherwise the sum of ``npy_shapes``.

    Parameters
    ----------
    tmp_dir:
        Temporary directory provided by the ``tmp_path`` fixture.
    npy_shapes:
        Number of samples to store in each ``data_i.npy`` file.
    csv_rows:
        Iterable of rows that will be written to ``metadata.csv``.
        Each row is a tuple ``(index, label, modcod, sample_rate)``.
    advertised_size:
        Optional size to write into ``info.json``.  If ``None`` the size is the
        real total number of samples (sum of ``npy_shapes``).

    Returns
    -------
    pathlib.Path
        Path to the directory that now contains a complete dataset.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Write the .npy files – each file gets a unique imaginary offset
    # ------------------------------------------------------------------
    for i, length in enumerate(npy_shapes):
        data = np.arange(length, dtype=np.complex64) + (i * 10j)
        np.save(tmp_dir / f"data_{i}.npy", data)

    # ------------------------------------------------------------------
    # 2️⃣  Write the CSV – we honour the order given in *csv_rows*
    # ------------------------------------------------------------------
    csv_path = tmp_dir / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)

    # ------------------------------------------------------------------
    # 3️⃣  Write ``info.json`` – required by the NPYReader constructor
    # ------------------------------------------------------------------
    if advertised_size is None:
        advertised_size = sum(npy_shapes)

    info_path = tmp_dir / "info.json"
    info_path.write_text(json.dumps({"size": advertised_size}), encoding="utf-8")

    return tmp_dir


# ----------------------------------------------------------------------
# Fixture – build a temporary dataset with three *.npy files
# ----------------------------------------------------------------------
@pytest.fixture
def npy_dataset(tmp_path: Path):
    """
    Construct a temporary directory that mimics a real TorchSig dataset.

    The layout contains:

    * three ``*.npy`` files (2, 3, and 1 samples respectively),
    * a ``metadata.csv`` file with one row per global sample index,
    * an ``info.json`` file whose ``size`` field matches the total number of
      samples.

    Returns
    -------
    tuple[Path, list[Path], list[int], int]
        ``(root, npy_paths, shapes, total_samples)`` where:

        * ``root`` – the temporary directory,
        * ``npy_paths`` – a list of the three created ``*.npy`` files,
        * ``shapes`` – a list with the length of each file (``[2, 3, 1]``),
        * ``total_samples`` – the sum of ``shapes`` (``6``).
    """
    # ------------------------------------------------------------------
    # 1️⃣  Create three *.npy files with deterministic content.
    #     The imaginary part encodes a per‑file offset (10·i) so that each
    #     waveform can be distinguished unambiguously.
    # ------------------------------------------------------------------
    shapes = [2, 3, 1]               # number of samples per file
    total = sum(shapes)              # total number of samples across all files
    npy_paths: list[Path] = []

    for i, length in enumerate(shapes):
        arr = np.arange(length, dtype=np.complex64) + (i * 10j)
        file_path = tmp_path / f"data_{i}.npy"
        np.save(file_path, arr)
        npy_paths.append(file_path)

    # ------------------------------------------------------------------
    # 2️⃣  Write a ``metadata.csv`` that matches the global indexing.
    #     The CSV must contain the columns ``index,label,modcod,sample_rate``.
    # ------------------------------------------------------------------
    csv_path = tmp_path / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        f.writelines(
            [
                f"0,BPSK,0,{DEFAULT_SR}\n",
                f"0,BPSK,0,{DEFAULT_SR}\n",
                f"1,QPSK,1,{DEFAULT_SR}\n",
                f"1,QPSK,1,{DEFAULT_SR}\n",
                f"1,QPSK,1,{DEFAULT_SR}\n",
                f"2,Noise,2,{DEFAULT_SR}\n",
            ]
        )

    # ------------------------------------------------------------------
    # 3️⃣  Write the required ``info.json`` that advertises the dataset size.
    # ------------------------------------------------------------------
    write_info_json(tmp_path, size=total)

    return tmp_path, npy_paths, shapes, total


# ----------------------------------------------------------------------
# Test successful construction and internal bookkeeping
# ----------------------------------------------------------------------
def test_npy_reader_initialisation(npy_dataset):
    """
    Verify that the reader discovers the correct files, builds the cumulative
    start‑index table, and reports the size from ``info.json``.
    """
    root, npy_paths, shapes, total = npy_dataset
    reader = NPYReader(str(root))

    # ---- file discovery -------------------------------------------------
    assert len(reader.npy_files) == len(npy_paths)
    assert [
        p.name for p in reader.npy_files
    ] == sorted(p.name for p in npy_paths)

    # ---- cumulative start indices ---------------------------------------
    expected_starts = [0]
    for length in shapes[:-1]:
        expected_starts.append(expected_starts[-1] + length)
    assert reader.file_start_indices == expected_starts

    # ---- total number of samples ----------------------------------------
    assert reader.total_elements == total

    # ---- advertised size (info.json) ------------------------------------
    assert len(reader) == total
    assert reader.class_list == ["BPSK", "QPSK", "Noise"]


# ----------------------------------------------------------------------
# Test that a directory without *.npy files raises FileNotFoundError
# ----------------------------------------------------------------------
def test_npy_reader_no_files(tmp_path: Path):
    """
    ``NPYReader`` should fail fast when no NumPy files are present, even if
    ``metadata.csv`` and ``info.json`` exist.
    """
    write_info_json(tmp_path, size=0)
    (tmp_path / "metadata.csv").write_text(
        f"0,BPSK,0,{DEFAULT_SR}\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match=r"No \.npy files"):
        NPYReader(str(tmp_path))


# ----------------------------------------------------------------------
# Test read() – correct file selection & data extraction
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "global_idx,expected_file_idx,expected_value",
    [
        # file0 (2 samples) → indices 0, 1
        (0, 0, 0 + 0j),
        (1, 0, 1 + 0j),
        # file1 (3 samples) → indices 2, 3, 4
        (2, 1, 0 + 10j),
        (3, 1, 1 + 10j),
        (4, 1, 2 + 10j),
        # file2 (1 sample) → index 5
        (5, 2, 0 + 20j),
    ],
)
def test_read_returns_correct_data(
    npy_dataset,
    global_idx: int,
    expected_file_idx: int,
    expected_value: complex,
):
    """
    For each global index, ensure that

    * the correct waveform is returned,
    * the waveform is a 1‑D ``np.ndarray`` containing the expected complex
      value,
    * the metadata dictionary contains the expected ``class_index`` (which,
      for this test suite, matches the file index).
    """
    root, _, shapes, _ = npy_dataset
    reader = NPYReader(str(root))

    # ------------------------------------------------------------------
    # 1️⃣  Retrieve the Signal instance
    # ------------------------------------------------------------------
    signal = reader.read(global_idx)

    # ------------------------------------------------------------------
    # 2️⃣  Validate the waveform data
    # ------------------------------------------------------------------
    assert isinstance(signal.data, np.ndarray)
    np.testing.assert_allclose(signal.data, np.array([expected_value]))

    # ------------------------------------------------------------------
    # 3️⃣  Validate the metadata (class_index should equal the file index)
    # ------------------------------------------------------------------
    assert signal.metadata["class_index"] == expected_file_idx

    # ------------------------------------------------------------------
    # 4️⃣  (optional sanity check) Verify that the internal binary search
    #     landed on the expected file by recomputing the start offset.
    # ------------------------------------------------------------------
    cumulative_start = sum(shapes[:expected_file_idx])
    assert cumulative_start == reader.file_start_indices[expected_file_idx]


# ----------------------------------------------------------------------
# Test out‑of‑range handling
# ----------------------------------------------------------------------
def test_read_index_out_of_range(npy_dataset):
    """
    ``NPYReader.read`` must raise ``IndexError`` for any index outside the
    range ``[0, total_elements)``.
    """
    root, _, _, total = npy_dataset
    reader = NPYReader(str(root))

    with pytest.raises(IndexError, match="out of range"):
        reader.read(-1)               # negative index

    with pytest.raises(IndexError, match="out of range"):
        reader.read(total)            # exactly one past the last element


# ----------------------------------------------------------------------
# Test that a malformed ``info.json`` raises a clear ``ValueError``
# ----------------------------------------------------------------------
def test_info_json_malformed(tmp_path: Path):
    """
    ``info.json`` containing invalid JSON should cause ``NPYReader`` to raise
    ``ValueError`` with a helpful message.
    """
    # Minimal .npy file so the constructor reaches the JSON parsing stage
    np.save(tmp_path / "data_0.npy", np.arange(5, dtype=np.complex64))

    # Write a broken JSON file (syntax error)
    (tmp_path / "info.json").write_text("{ not a valid json }", encoding="utf-8")

    # Valid CSV required by ``metadata_loader``
    (tmp_path / "metadata.csv").write_text(
        f"0,BPSK,0,{DEFAULT_SR}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Error loading"):
        NPYReader(str(tmp_path))


# ----------------------------------------------------------------------
# Test that a missing ``info.json`` raises ``ValueError``
# ----------------------------------------------------------------------
def test_info_json_missing(tmp_path: Path):
    """
    When ``info.json`` is absent the reader should raise ``ValueError`` with a
    message that includes the full path to the missing file.
    """
    np.save(tmp_path / "data_0.npy", np.arange(3, dtype=np.complex64))
    (tmp_path / "metadata.csv").write_text(
        f"0,BPSK,0,{DEFAULT_SR}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Error loading"):
        NPYReader(str(tmp_path))


# ----------------------------------------------------------------------
# Missing ``info.json`` → ``ValueError`` with full path
# ----------------------------------------------------------------------
def test_missing_info_json_raises_value_error(tmp_path: Path):
    """
    Helper ``_make_one_sample_dataset`` creates a valid dataset **without**
    ``info.json``.  The reader must raise ``ValueError`` and the message must
    contain the absolute path to the missing file.
    """
    _make_one_sample_dataset(tmp_path)
    expected_msg = f"Error loading {tmp_path}/info.json"
    with pytest.raises(ValueError) as excinfo:
        NPYReader(str(tmp_path))
    assert expected_msg in str(excinfo.value)


# ----------------------------------------------------------------------
# Malformed JSON (syntax error) → same behaviour as above
# ----------------------------------------------------------------------
def test_malformed_info_json_raises_value_error(tmp_path: Path):
    """
    ``info.json`` is present but malformed; the exception handling should be
    identical to the “missing file” case.
    """
    _make_one_sample_dataset(tmp_path)
    (tmp_path / "info.json").write_text("{ not a valid json }", encoding="utf-8")
    expected_msg = f"Error loading {tmp_path}/info.json"
    with pytest.raises(ValueError) as excinfo:
        NPYReader(str(tmp_path))
    assert expected_msg in str(excinfo.value)


# ----------------------------------------------------------------------
# Permission error when opening ``info.json`` → ``ValueError``
# ----------------------------------------------------------------------
def test_unreadable_info_json_raises_value_error(tmp_path: Path, monkeypatch):
    """
    Simulate a permission failure when the reader tries to open ``info.json``.
    The original exception should be wrapped in a ``ValueError`` with a clear
    message.
    """
    _make_one_sample_dataset(tmp_path)
    (tmp_path / "info.json").write_text(json.dumps({"size": 1}), encoding="utf-8")

    real_open = builtins.open

    def _open(path, *args, **kwargs):
        if str(path).endswith("info.json"):
            raise PermissionError("simulated permission failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)

    expected_msg = f"Error loading {tmp_path}/info.json"
    with pytest.raises(ValueError) as excinfo:
        NPYReader(str(tmp_path))
    assert expected_msg in str(excinfo.value)


# ----------------------------------------------------------------------
# Generic unexpected exception (e.g., OSError) while reading ``info.json``
# ----------------------------------------------------------------------
def test_generic_exception_is_wrapped(tmp_path: Path, monkeypatch):
    """
    Any exception raised while opening ``info.json`` should be wrapped in a
    ``ValueError`` so that callers see a uniform error type.
    """
    _make_one_sample_dataset(tmp_path)
    (tmp_path / "info.json").write_text(json.dumps({"size": 1}), encoding="utf-8")

    real_open = builtins.open

    def _open(path, *args, **kwargs):
        if str(path).endswith("info.json"):
            raise OSError("simulated OS failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)

    expected_msg = f"Error loading {tmp_path}/info.json"
    with pytest.raises(ValueError) as excinfo:
        NPYReader(str(tmp_path))
    assert expected_msg in str(excinfo.value)


# ----------------------------------------------------------------------
# Verify that ``__len__`` reflects the size field from ``info.json``,
# not the *actual* number of samples (the class stores both).
# ----------------------------------------------------------------------
def test_len_uses_info_json_not_total_elements(tmp_path: Path, monkeypatch):
    """
    ``len(reader)`` must return the *advertised* size from ``info.json``, even
    when that size differs from the real number of samples stored in the
    ``*.npy`` files.
    """
    # Two files, each with 4 samples → total_elements = 8
    for i in range(2):
        np.save(
            tmp_path / f"data_{i}.npy",
            np.arange(4, dtype=np.complex64) + (i * 10),
        )

    # CSV with at least 8 rows (the loader only checks that it can be read)
    with (tmp_path / "metadata.csv").open("w", newline="") as f:
        for i in range(8):
            f.write(f"{i},BPSK,0,{DEFAULT_SR}\n")

    # Advertised size deliberately lies (5 != 8)
    write_info_json(tmp_path, size=5)

    reader = NPYReader(str(tmp_path))

    # ``__len__`` must reflect the JSON size
    assert len(reader) == 5

    # The internal count used for bounds checking stays correct
    assert reader.total_elements == 8

    # Access a valid index according to ``total_elements`` but beyond the
    # advertised size – the reader should still succeed.
    signal = reader.read(6)
    assert signal is not None

    # ------------------------------------------------------------------
    #  Re‑exercise the error‑path: monkey‑patch ``open`` to raise a
    #  PermissionError for ``info.json`` and verify that a new instance
    #  still raises the expected ``ValueError``.
    # ------------------------------------------------------------------
    real_open = builtins.open

    def _open(path, *args, **kwargs):
        if str(path).endswith("info.json"):
            raise PermissionError("simulated permission failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)

    expected_msg = f"Error loading {tmp_path}/info.json"
    with pytest.raises(ValueError) as excinfo:
        NPYReader(str(tmp_path))
    assert expected_msg in str(excinfo.value)


# ----------------------------------------------------------------------
# Force an exception *inside* the second ``json.load`` call (the one in the
# ``try`` block of ``NPYReader.__init__``) and verify that it is wrapped.
# ----------------------------------------------------------------------
def test_second_json_load_exception_is_caught(tmp_path: Path, monkeypatch):
    """
    ``NPYReader`` reads ``info.json`` twice:

    1.  In ``_load_json_metadata`` (used for internal bookkeeping).
    2.  In the ``try`` block that extracts ``size``.

    The test forces the **second** ``json.load`` to raise, ensuring that the
    ``except Exception`` clause is exercised directly.
    """
    _make_one_sample_dataset(tmp_path)
    (tmp_path / "info.json").write_text(
        json.dumps({"size": 1}), encoding="utf-8"
    )

    # ``json.load`` will succeed the first time then raise a RuntimeError.
    call_counter = {"n": 0}
    real_load = json.load

    def _fake_load(fp, *a, **kw):
        if call_counter["n"] == 0:
            call_counter["n"] += 1
            return real_load(fp, *a, **kw)
        raise RuntimeError("simulated json.load failure")

    monkeypatch.setattr(json, "load", _fake_load)

    with pytest.raises(ValueError):
        NPYReader(str(tmp_path))

# ----------------------------------------------------------------------
# CSV index out of bounds → IndexError
# ----------------------------------------------------------------------
def test_read_raises_index_error_when_csv_is_too_short(tmp_path: pathlib.Path):
    """
    Build a dataset where ``info.json`` advertises 5 samples but the CSV
    contains only three rows.  Accessing a global index that points beyond the
    CSV should raise ``IndexError`` with the message
    ``"Metadata idx <n> is out of bounds"``.
    """
    dataset = create_dataset(
        tmp_path,
        npy_shapes=[2, 3],
        csv_rows=[                         # only three rows (indices 0‑2)
            (0, "BPSK", 0, 192_000),
            (1, "QPSK", 1, 192_000),
            (2, "Noise", 2, 192_000),
        ],
    )

    reader = NPYReader(str(dataset))

    # Global index 4 is valid for the waveform data (total = 5) but the CSV
    # has no row 4 → we expect an IndexError from the CSV‑lookup block.
    with pytest.raises(IndexError, match=r"Metadata idx 4 is out of bounds"):
        reader.read(4)