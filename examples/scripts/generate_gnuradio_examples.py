#!/usr/bin/env python3

"""A small, reusable Python class that generates a TorchSig-compatible
stereo-IQ WAV or SigMF dataset (BPSK, QPSK, 8-PSK) and the required side-car files.

Usage (in a Jupyter notebook) ------------------------------------------------
    from examples.scripts.gnuradio_iq_dataset import IQDatasetGenerator

    # Use all defaults
    gen = IQDatasetGenerator(root="training")
    gen.generate()                 # creates the whole WAV data set
    gen.plot_example(mod="BPSK", snr=-5)   # quick visual sanity-check

    # Or generate SigMF instead of WAV
    gen = IQDatasetGenerator(root="training_sigmf", output_format="sigmf")
    gen.generate()

You can also customise every parameter:

    custom_mods = {
        "BPSK": np.array([1+0j, -1+0j]),
        "QPSK": np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2),
        "8PSK": np.exp(1j*np.arange(8)*2*np.pi/8),
        "16QAM": (np.arange(-3, 4, 2)[:,None] + 1j*np.arange(-3, 4, 2)).flatten() / np.sqrt(10)
    }

    gen = IQDatasetGenerator(
        root="my_dataset",
        modulations=custom_mods,
        snr_db=[-10, 0, 10],
        duration_s=0.5,
        base_rate=2_000_000,
        audio_rate=96_000,
        seed=12345,
        chunk_size=2                     # 2 bits per symbol for QPSK, 16QAM, …
    )
    gen.generate()
------------------------------------------------------------------------
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import csv
import hashlib
import importlib
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)

METADATA_FIELDS = (
    "index",
    "label",
    "modcod",
    "sample_rate",
    "snr_db",
    "seed",
    "file_path",
    "start_frame",
    "num_frames",
    "expected_sample_rate",
    "channel_count",
)


@dataclass(frozen=True)
class RecordSpec:
    """Deterministic description of one generated IQ record."""

    index: int
    mod_name: str
    snr_db: float
    seed: int
    relative_path: str


@dataclass(frozen=True)
class RecordResult:
    """Metadata returned to the parent after one record is ready."""

    spec: RecordSpec
    num_frames: int


def _derive_record_seed(base_seed: int, mod_name: str, snr_db: float) -> int:
    """Derive a process-stable GNU Radio seed for one record.

    The contract hashes compact JSON containing the integer base seed, the
    modulation name, and the hexadecimal representation of the SNR float with
    SHA-256. The first four digest bytes are interpreted as a big-endian
    integer and limited to the positive signed 32-bit range.
    """
    key = json.dumps(
        [int(base_seed), mod_name, float(snr_db).hex()],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


# ----------------------------------------------------------------------
# Helper -- tiny enum mimicking the original script's naming
# ----------------------------------------------------------------------
def _default_modulations() -> dict[str, np.ndarray]:
    """Return the three constellations used in the original script."""
    return {
        "BPSK": np.array([1 + 0j, -1 + 0j]),  # 2-PSK
        "QPSK": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),  # 4-PSK
        "8PSK": np.exp(1j * np.arange(8) * 2 * np.pi / 8),  # 8-PSK
    }


# ----------------------------------------------------------------------
class IQDatasetGenerator:
    """Generate a TorchSig-compatible stereo-IQ WAV data-set.

    Parameters
    ----------
    root : str | Path
        Base directory where the ``training/`` hierarchy will be created.
    modulations : dict (optional)
        Mapping ``mod_name → np.ndarray`` of complex constellation points.
        If omitted the three default constellations are used.
    modcod_map : dict (optional)
        Mapping ``mod_name → integer`` that will be stored in the CSV
        column ``modcod``.  By default the order of ``modulations.keys()``
        is enumerated starting at zero.
    snr_db : list[int] (optional)
        List of SNR values (in dB) that should be generated for *each*
        modulation.  Default = ``[-5, 0, 5, 10, 15, 20]``.
    duration_s : float (optional)
        Length of each WAV file in seconds.  Default = ``1.0``.
    base_rate : int (optional)
        Simulation sample-rate *before* the rational resampler.
        Default = ``1_000_000`` (1 MS/s).
    audio_rate : int (optional)
        Sample-rate of the final WAV files.  Default = ``48_000``.
    seed : int (optional)
        Random seed used to generate the bit-stream (ensures reproducibility).
    chunk_size : int (optional)
        Number of bits per symbol for the chosen constellations.
        BPSK → 1, QPSK → 2, 8PSK → 3, …  (default = 1 as in the original script).
    output_format : {"wav", "sigmf"}, optional
        Output file format. ``"wav"`` is the default to preserve backwards
        compatibility. Use ``"sigmf"`` to write ``*.sigmf-data`` and
        ``*.sigmf-meta`` files instead.
    num_workers : int, optional
        Number of record-generation processes. The default of ``1`` preserves
        historical in-process execution.
    log_level : int | str, optional
        Logging level for generator progress messages.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        root: str | Path,
        modulations: dict[str, np.ndarray] | None = None,
        modcod_map: dict[str, int] | None = None,
        snr_db: list[int] | None = None,
        duration_s: float = 1.0,
        base_rate: int = 1_000_000,
        audio_rate: int = 48_000,
        seed: int = 20230612,
        chunk_size: int = 1,
        output_format: str = "wav",
        num_workers: int = 1,
        log_level: int | str = logging.INFO,
    ):
        self.root = Path(root).expanduser().resolve()
        self.output_format = output_format.lower()

        if self.output_format not in {"wav", "sigmf"}:
            raise ValueError("output_format must be either 'wav' or 'sigmf'")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")
        if isinstance(log_level, str):
            normalized_log_level = logging.getLevelNamesMapping().get(log_level.upper())
            if normalized_log_level is None:
                raise ValueError(f"unknown log level: {log_level}")
            log_level = normalized_log_level
        elif isinstance(log_level, bool) or not isinstance(log_level, int):
            raise TypeError("log_level must be an integer or logging level name")
        self.modulations = modulations if modulations is not None else _default_modulations()
        self.mod_names = list(self.modulations.keys())

        # Build the integer mod-code map if the user didn't supply one
        if modcod_map is None:
            self.modcod_map = {name: i for i, name in enumerate(self.mod_names)}
        else:
            self.modcod_map = dict(modcod_map)

        self.snr_db = snr_db if snr_db is not None else [-5, 0, 5, 10, 15, 20]
        self.duration_s = float(duration_s)
        self.base_rate = int(base_rate)
        self.audio_rate = int(audio_rate)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.num_workers = num_workers
        self.log_level = log_level

        # Will be filled after ``generate()`` runs
        self.metadata_rows: list[list[str]] = []
        self.info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helper -- convert an SNR (dB) → noise voltage for the GNU-Radio channel model
    # ------------------------------------------------------------------
    @staticmethod
    def _noise_vol_from_snr(snr_db: float) -> float:
        """Noise voltage for unit-power constellations."""
        return np.sqrt(0.5 * 10 ** (-snr_db / 10.0))

    @staticmethod
    def _bits_per_symbol(constellation: np.ndarray) -> int:
        """Return the integer number of bits needed to address the constellation."""
        return int(np.ceil(np.log2(constellation.size)))

    # --------------------------------------------------------------
    # Core -- generate ONE file (now uses the correct bits per symbol)
    # --------------------------------------------------------------
    def generate_one(
        self,
        mod_name: str,
        snr_db: float,
        out_path: str,
        record_seed: int | None = None,
    ) -> None:
        """Generate ONE stereo (I-left, Q-right) WAV file **as float32**.
        The number of bits per symbol is derived from the constellation size,
        so QPSK → 2 bits/symbol, 8PSK → 3 bits/symbol, etc.
        """
        blocks = importlib.import_module("gnuradio.blocks")
        channels = importlib.import_module("gnuradio.channels")
        digital = importlib.import_module("gnuradio.digital")
        gr_filter = importlib.import_module("gnuradio.filter")
        gr = importlib.import_module("gnuradio.gr")

        tb = gr.top_block()

        # ------------------------------------------------------------------
        # 0) Determine how many bits each symbol carries for this modulation
        # ------------------------------------------------------------------
        constel = self.modulations[mod_name]  # np.ndarray of complex points
        bits_per_symbol = int(np.ceil(np.log2(constel.size)))  # 1 for BPSK, 2 for QPSK, 3 for 8PSK …

        # ------------------------------------------------------------------
        # 1) deterministic random bits -- make the seed depend on (mod, snr)
        # ------------------------------------------------------------------
        # Using a distinct seed for each (mod, snr) pair gives a different
        # bit-stream while still being reproducible if the same arguments are
        # used again.
        seed_for_this_call = record_seed
        if seed_for_this_call is None:
            seed_for_this_call = _derive_record_seed(self.seed, mod_name, snr_db)
        rng = np.random.default_rng(seed_for_this_call)

        # Number of *symbols* we need to fill the requested audio duration:
        symbols_needed = int(self.audio_rate * self.duration_s)
        n_bits = symbols_needed * bits_per_symbol

        bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
        src_bits = blocks.vector_source_b(bits.tolist(), repeat=False)

        # ------------------------------------------------------------------
        # 2) pack bits → integer symbols (using the correct chunk size)
        # ------------------------------------------------------------------
        packer = blocks.pack_k_bits_bb(bits_per_symbol)

        # ------------------------------------------------------------------
        # 3) map integer symbols → complex IQ
        # ------------------------------------------------------------------
        mapper = digital.chunks_to_symbols_bc(constel.tolist(), 1)

        # ------------------------------------------------------------------
        # 4) optional AWGN
        # ------------------------------------------------------------------
        chan = channels.channel_model(
            noise_voltage=self._noise_vol_from_snr(snr_db),
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=seed_for_this_call,
        )

        # ------------------------------------------------------------------
        # 5) Resample 1 MS/s → audio_rate (reduce fraction to avoid GCD warning)
        # ------------------------------------------------------------------
        interp = self.audio_rate
        decim = self.base_rate
        g = gcd(interp, decim)
        interp //= g
        decim //= g
        resamp = gr_filter.rational_resampler_ccc(
            interpolation=interp,
            decimation=decim,
            taps=[1.0],
        )

        # ------------------------------------------------------------------
        # 6) split complex → two float streams (I, Q)
        # ------------------------------------------------------------------
        c2float = blocks.complex_to_float()

        # ------------------------------------------------------------------
        # 7) collect the float streams (vector_sink_f)
        # ------------------------------------------------------------------
        collector_i = blocks.vector_sink_f()
        collector_q = blocks.vector_sink_f()

        # ------------------------------------------------------------------
        # 8) Trim to exactly the number of audio samples we need
        # ------------------------------------------------------------------
        expected_len = int(self.audio_rate * self.duration_s)  # samples per channel
        head_i = blocks.head(gr.sizeof_float, expected_len)
        head_q = blocks.head(gr.sizeof_float, expected_len)

        # ------------------------------------------------------------------
        # Wiring
        # ------------------------------------------------------------------
        tb.connect(src_bits, packer)
        tb.connect(packer, mapper)
        tb.connect(mapper, chan)
        tb.connect(chan, resamp)
        tb.connect(resamp, c2float)

        # I-path
        tb.connect((c2float, 0), head_i)
        tb.connect(head_i, collector_i)

        # Q-path
        tb.connect((c2float, 1), head_q)
        tb.connect(head_q, collector_q)

        # ------------------------------------------------------------------
        # Run the flow-graph (blocking)
        # ------------------------------------------------------------------
        tb.run()

        # ------------------------------------------------------------------
        # Retrieve the samples; pad if we ended up a couple of samples short.
        # ------------------------------------------------------------------
        i_samples = np.array(collector_i.data(), dtype=np.float32)
        q_samples = np.array(collector_q.data(), dtype=np.float32)

        if i_samples.shape[0] < expected_len:
            pad_len = expected_len - i_samples.shape[0]
            i_samples = np.pad(i_samples, (0, pad_len), mode="constant")
            q_samples = np.pad(q_samples, (0, pad_len), mode="constant")

        self._write_iq_recording(
            out_path=Path(out_path),
            i_samples=i_samples,
            q_samples=q_samples,
            mod_name=mod_name,
            snr_db=snr_db,
            record_seed=seed_for_this_call,
        )

    # ------------------------------------------------------------------
    # Helper -- write either WAV or SigMF output
    # ------------------------------------------------------------------
    def _write_iq_recording(
        self,
        out_path: Path,
        i_samples: np.ndarray,
        q_samples: np.ndarray,
        mod_name: str,
        snr_db: float,
        record_seed: int | None = None,
    ) -> None:
        """Write generated IQ samples in the configured output format."""
        if self.output_format == "wav":
            iq_interleaved = np.column_stack((i_samples, q_samples))
            wavfile.write(
                out_path,
                self.audio_rate,
                iq_interleaved.astype(np.float32),
            )
            return

        if self.output_format == "sigmf":
            data_path = out_path
            meta_path = data_path.with_suffix(".sigmf-meta")

            complex_iq = (i_samples.astype(np.float32) + 1j * q_samples.astype(np.float32)).astype(np.complex64)
            complex_iq.tofile(data_path)

            sigmf_meta = {
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": self.audio_rate,
                    "core:version": "1.0.0",
                    "core:description": ("GNU Radio generated TorchSig-compatible IQ recording"),
                    "torchsig:label": mod_name,
                    "torchsig:modcod": self.modcod_map[mod_name],
                    "torchsig:snr_db": snr_db,
                    "torchsig:seed": self.seed if record_seed is None else record_seed,
                },
                "captures": [
                    {
                        "core:sample_start": 0,
                        "core:frequency": 0,
                    }
                ],
                "annotations": [
                    {
                        "core:sample_start": 0,
                        "core:sample_count": int(complex_iq.shape[0]),
                        "core:label": mod_name,
                    }
                ],
            }

            with meta_path.open("w") as f:
                json.dump(sigmf_meta, f, indent=4)
            return

        raise ValueError(f"Unsupported output format: {self.output_format}")

    # ------------------------------------------------------------------
    # Generate the complete data set (folder hierarchy, CSV, JSON)
    # ------------------------------------------------------------------
    def build_record_specs(self) -> list[RecordSpec]:
        """Return record specifications in stable dataset order."""
        suffix = "wav" if self.output_format == "wav" else "sigmf-data"
        specs = []
        for index, (mod_name, snr) in enumerate((mod_name, snr) for mod_name in self.mod_names for snr in self.snr_db):
            filename = f"{mod_name}_{snr:+03.0f}dB_seed{self.seed}.{suffix}"
            specs.append(
                RecordSpec(
                    index=index,
                    mod_name=mod_name,
                    snr_db=float(snr),
                    seed=_derive_record_seed(self.seed, mod_name, snr),
                    relative_path=(Path(mod_name) / filename).as_posix(),
                )
            )
        return specs

    def _metadata_row(self, result: RecordResult) -> list[str]:
        """Convert a worker result to the versioned sidecar schema."""
        spec = result.spec
        return [
            str(spec.index),
            spec.mod_name,
            str(self.modcod_map[spec.mod_name]),
            str(self.audio_rate),
            str(spec.snr_db),
            str(spec.seed),
            spec.relative_path,
            "0",
            str(result.num_frames),
            str(self.audio_rate),
            "2",
        ]

    def generate(self) -> None:
        """Create the full dataset under ``self.root``.
        After the call ``self.metadata_rows`` and ``self.info`` are populated.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        specs = self.build_record_specs()
        if self.num_workers == 1:
            results = [_execute_record(self, spec) for spec in specs]
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(_execute_record, [self] * len(specs), specs))

        self.metadata_rows = [self._metadata_row(result) for result in results]
        self.info = {
            "num_iq_samples": int(self.audio_rate * self.duration_s),
            "elements_per_file": 1,
            "size": len(self.metadata_rows),
            "class_list": self.mod_names,
            "sample_rate": self.audio_rate,
            "output_format": self.output_format,
            "datatype": "cf32_le" if self.output_format == "sigmf" else "float32_stereo",
            "sidecar_schema_version": "1.0",
        }

        csv_path = self.root / "metadata.csv"
        csv_temp = _temporary_path(csv_path)
        json_path = self.root / "info.json"
        json_temp = _temporary_path(json_path)
        try:
            with csv_temp.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(METADATA_FIELDS)
                writer.writerows(self.metadata_rows)
            with json_temp.open("w", encoding="utf-8") as json_file:
                json.dump(self.info, json_file, indent=4)
                json_file.write("\n")
            csv_temp.replace(csv_path)
            json_temp.replace(json_path)
        except BaseException:
            csv_temp.unlink(missing_ok=True)
            json_temp.unlink(missing_ok=True)
            raise

        logger.log(self.log_level, "Dataset generation complete: %s", self.root)
        logger.log(self.log_level, "Metadata: %s", csv_path)
        logger.log(self.log_level, "Dataset information: %s", json_path)

    # ------------------------------------------------------------------
    # Plot an example file (convenient for notebooks)
    # ------------------------------------------------------------------
    def plot_example(
        self,
        mod: str,
        snr: float,
        index: int | None = None,
        n_plot_ms: float = 2.0,
    ) -> None:
        """Load a generated file and plot the short-time I trace *and* the
        full IQ constellation.

        Parameters
        ----------
        mod : str
            Modulation name (must exist in the dataset).
        snr : float
            SNR value (must match one of the generated files).
        index : int | None, optional
            If you know the global element index you want to plot, pass it
            here and the function will ignore ``mod``/``snr``.  Handy for
            debugging after you have already called ``generate()``.
        n_plot_ms : float, optional
            Length of the time-trace window (in milliseconds).  Default = 2_ms.
        """
        if index is not None:
            # Locate the CSV row that matches the global index
            if not (0 <= index < len(self.metadata_rows)):
                raise IndexError(f"index {index} out of range.")
            row = self.metadata_rows[index]
            # row layout: index, label, modcod, sample_rate, snr, seed
            _, mod, _, _, snr_str, _ = row
            snr = float(snr_str)

        suffix = "wav" if self.output_format == "wav" else "sigmf-data"
        out_path = self.root / mod / f"{mod}_{snr:+03.0f}dB_seed{self.seed}.{suffix}"
        if not out_path.is_file():
            raise FileNotFoundError(f"Requested file not found: {out_path}")

        sr, i, q = _load_iq_file(out_path, sample_rate=self.audio_rate)

        # ---- 1) short I-trace -------------------------------------------------
        t = np.arange(len(i)) / sr
        n_plot = int((n_plot_ms / 1000.0) * sr)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(t[:n_plot], i[:n_plot], lw=1)
        plt.title(f"I-channel (first {n_plot_ms:.1f} ms)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(visible=True)

        # ---- 2) IQ constellation --------------------------------------------
        plt.subplot(1, 2, 2)
        plt.scatter(i, q, s=6, alpha=0.6, edgecolors="k", label="Received")
        plt.title("IQ Constellation")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.axis("equal")
        plt.grid(visible=True)

        # overlay the ideal constellation for the requested modulation
        ideal = self._ideal_constellation(mod)
        if ideal is not None:
            plt.scatter(ideal.real, ideal.imag, marker="x", s=80, c="red", linewidths=2, label=f"Ideal {mod}")
            plt.legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal helper -- return the ideal constellation for a modulation
    # ------------------------------------------------------------------
    @staticmethod
    def _ideal_constellation(name: str) -> np.ndarray | None:
        """Return the perfect (noise-free) constellation for the three basics."""
        ideals = {
            "BPSK": np.array([1 + 0j, -1 + 0j]),
            "QPSK": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),
            "8PSK": np.exp(1j * np.arange(8) * 2 * np.pi / 8),
        }
        return ideals.get(name)


def _temporary_path(final_path: Path) -> Path:
    """Create an adjacent temporary path while preserving the final suffix."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.stem}.",
        suffix=final_path.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _execute_record(generator: IQDatasetGenerator, spec: RecordSpec) -> RecordResult:
    """Generate and atomically publish one record in the current process."""
    out_path = generator.root / spec.relative_path
    meta_path = out_path.with_suffix(".sigmf-meta")
    complete = out_path.is_file() and (generator.output_format != "sigmf" or meta_path.is_file())
    if complete:
        logger.log(generator.log_level, "File already exists: %s", out_path)
        return RecordResult(spec=spec, num_frames=int(generator.audio_rate * generator.duration_s))

    logger.log(generator.log_level, "Generating %s", out_path)
    temp_path = _temporary_path(out_path)
    temp_meta_path = temp_path.with_suffix(".sigmf-meta")
    published_paths = []
    try:
        generator.generate_one(
            spec.mod_name,
            spec.snr_db,
            str(temp_path),
            record_seed=spec.seed,
        )
        if generator.output_format == "sigmf":
            temp_meta_path.replace(meta_path)
            published_paths.append(meta_path)
        temp_path.replace(out_path)
        published_paths.append(out_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        temp_meta_path.unlink(missing_ok=True)
        for published_path in published_paths:
            published_path.unlink(missing_ok=True)
        raise

    return RecordResult(spec=spec, num_frames=int(generator.audio_rate * generator.duration_s))


def _load_iq_file(out_path: str | Path, sample_rate: int | None = None) -> tuple[int, np.ndarray, np.ndarray]:
    """Load either stereo-IQ WAV or cf32_le SigMF IQ samples."""
    out_path = Path(out_path)

    if out_path.suffix == ".wav":
        sr, data = wavfile.read(str(out_path))

        if data.ndim != 2 or data.shape[1] != 2:  # noqa: PLR2004
            raise ValueError("Expected a stereo (2-channel) wav containing I & Q.")

        if data.dtype == np.int16:
            i = data[:, 0].astype(np.float32) / 32767.0
            q = data[:, 1].astype(np.float32) / 32767.0
        elif data.dtype == np.float32:
            i = data[:, 0]
            q = data[:, 1]
        else:
            raise TypeError(f"Unsupported WAV sample type: {data.dtype}")

        return int(sr), i, q

    if out_path.name.endswith(".sigmf-data"):
        meta_path = out_path.with_suffix(".sigmf-meta")
        sr = sample_rate

        if meta_path.is_file():
            with meta_path.open() as f:
                meta = json.load(f)
            datatype = meta.get("global", {}).get("core:datatype", "cf32_le")
            sr = int(meta.get("global", {}).get("core:sample_rate", sr or 0))
        else:
            datatype = "cf32_le"

        if datatype != "cf32_le":
            raise TypeError(f"Unsupported SigMF datatype for plotting: {datatype}")

        if sr is None or sr == 0:
            raise ValueError("sample_rate is required when plotting SigMF without metadata.")

        iq = np.fromfile(out_path, dtype=np.complex64)
        return int(sr), iq.real.astype(np.float32), iq.imag.astype(np.float32)

    raise TypeError(f"Unsupported IQ file type: {out_path}")


def plot_iq(out_path: str) -> None:
    """Show a short I-trace and the full IQ constellation.

    Works for
      * 16-bit PCM WAV files (int16, values in [-32768, 32767])
      * 32-bit float WAV files (float32, values already in [-1, 1])
      * SigMF ``cf32_le`` recordings
    """
    if not Path(out_path).is_file():
        raise FileNotFoundError(f"File not found: {out_path}")

    sr, i, q = _load_iq_file(out_path)

    # --------------------------------------------------------------
    # 1) short I-trace (first ~2 ms)
    # --------------------------------------------------------------
    t = np.arange(len(i)) / sr
    n_plot = int(0.002 * sr)  # ~2 ms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t[:n_plot], i[:n_plot], lw=1)
    plt.title("I-channel (first 2 ms)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(visible=True)

    # --------------------------------------------------------------
    # 2) IQ constellation (all samples)
    # --------------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.scatter(i, q, s=6, alpha=0.6, edgecolors="k", label="Received symbols")
    plt.title("IQ Constellation")
    plt.xlabel("I (real)")
    plt.ylabel("Q (imag)")
    plt.axis("equal")
    plt.grid(visible=True)

    # --------------------------------------------------------------
    # 3) overlay the ideal constellation points (if we can guess)
    # --------------------------------------------------------------
    mod_guess = None
    for k in _default_modulations():
        if f"/{k}/" in out_path.replace("\\", "/"):
            mod_guess = k
            break
    if mod_guess:
        ideal = _default_modulations()[mod_guess]
        plt.scatter(ideal.real, ideal.imag, marker="x", s=80, c="red", linewidths=2, label=f"Ideal {mod_guess}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# If you run this file directly (e.g. ``python gnuradio_iq_dataset.py``)
# it will generate the data set with the default parameters -- handy for
# quick testing without a notebook.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    generator = IQDatasetGenerator(root="training")
    generator.generate()
    # Show a quick plot of the first BPSK, -5 dB file
    generator.plot_example(mod="BPSK", snr=-5)
