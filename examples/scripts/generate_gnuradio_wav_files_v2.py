#!/usr/bin/env python3
"""GNU Radio script to generate IQ data as:
  - Stereo 32-bit float WAV files (native GNU Radio output, elements_per_file per file)
  - metadata.csv (one row per IQ record)
  - info.json (global metadata)

Each element in the dataset has the modulation type specified in its metadata row.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from gnuradio import blocks, gr


def get_rng(seed: int | None) -> np.random.Generator:
    """Return a reproducible RNG; if seed is None use OS entropy."""
    return np.random.default_rng(seed)


def generate_random_modulated_samples(
    num_samples: int,
    modulation: str,
    rng: np.random.Generator,
    *,
    snr_db: float | None = None,
    phase_offset_rad: float = 0.0,
    amp_scale: float = 1.0,
) -> np.ndarray:
    """Build the *raw* constellation, then optionally:
        • rotate the whole vector by a single random phase (`phase_offset_rad`);
        • scale the amplitude;
        • add AWGN (if an SNR is supplied);
        • apply occasional Rayleigh fading.
    No interpolation or carrier-frequency rotation is performed, so the
    points stay discrete.
    """
    # ------------------------------------------------------------------
    # Raw constellation
    # ------------------------------------------------------------------
    if modulation == "NOISE":
        # Low-power Gaussian noise
        sig = (rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples)) * 0.1
    elif modulation == "BPSK":
        bits = rng.integers(0, 2, size=num_samples)
        sig  = (2 * bits - 1).astype(np.complex64) + 0j          # ±1 on the real axis
    elif modulation == "QPSK":
        bits = rng.integers(0, 4, size=num_samples)
        tbl  = {0: 1 + 1j, 1: 1 - 1j, 2: -1 + 1j, 3: -1 - 1j}
        sig  = np.vectorize(tbl.get, otypes=[np.complex64])(bits)
        sig  = sig / np.sqrt(2)                                 # unit-power normalisation
    elif modulation == "8PSK":
        bits   = rng.integers(0, 8, size=num_samples)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)     # 0°,45°, … 315°
        sig    = np.exp(1j * angles[bits]).astype(np.complex64)
    elif modulation == "16QAM":
        bits = rng.integers(0, 16, size=num_samples)
        const_i = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        const_q = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        i = const_i[bits // 4]
        q = const_q[bits % 4]
        sig = (i + 1j * q).astype(np.complex64)
    else:
        # Fallback -- constant tone (should never be used)
        sig = np.ones(num_samples, dtype=np.complex64)

    # ------------------------------------------------------------------
    # Random post-processing (static phase, scaling, noise, fading)
    # ------------------------------------------------------------------
    # a) Static phase rotation (single complex constant)
    if phase_offset_rad != 0.0:
        sig = sig * np.exp(1j * phase_offset_rad)

    # b) Global amplitude scaling
    sig = sig * amp_scale

    # c) Add AWGN for the requested SNR
    if snr_db is not None and not np.isnan(snr_db):
        signal_power = np.mean(np.abs(sig) ** 2)
        snr_linear   = 10 ** (snr_db / 10)
        noise_power  = signal_power / snr_linear
        noise = (rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples))
        noise = noise * np.sqrt(noise_power / 2)
        sig = sig + noise

    # d) Optional Rayleigh fading (kept from the original “more random” script)
    if modulation in ("16QAM", "NOISE") and rng.random() < 0.2:
        fading = rng.rayleigh(scale=1.0, size=num_samples)
        sig = sig * fading.astype(np.complex64)

    return sig.astype(np.complex64)


def generate_metadata(
    root: Path,
    total_elements: int,
    sample_rate: float,
    class_list: list[str],
    num_iq_samples: int,
    elements_per_file: int,
    num_files: int,
    rng: np.random.Generator,
) -> None:
    """Writes info.json and metadata.csv."""
    info = {
        "size": total_elements,
        "num_iq_samples": num_iq_samples,
        "elements_per_file": elements_per_file,
        "num_files": num_files,
        "sample_rate": sample_rate,
        "class_list": class_list,
    }
    (root / "info.json").write_text(json.dumps(info, indent=2))

    header = [
        "index",
        "label",
        "modcod",
        "sample_rate_hz",
        "snr_db",          # empty string → no noise added
        "phase_offset_rad",
        "amp_scale",
    ]

    with (root / "metadata.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)                     # write header once

        for i in range(total_elements):
            label    = rng.choice(class_list)
            modcod   = class_list.index(label)

            # Random generation parameters (same as later used in the generator)
            snr_db          = rng.uniform(5, 30) if label != "NOISE" else np.nan
            phase_offset    = rng.uniform(-np.pi, np.pi)
            amp_scale       = rng.uniform(0.2, 1.0)

            writer.writerow(
                [
                    i,
                    label,
                    modcod,
                    int(sample_rate),
                    f"{snr_db:.2f}" if not np.isnan(snr_db) else "",
                    f"{phase_offset:.4f}",
                    f"{amp_scale:.3f}",
                ]
            )


class IQGenerator:
    """Core generator -- creates *num_files* wav files, each holding
    *elements_per_file* IQ records, each with *num_iq_samples* complex samples.
    Randomness (class, SNR, freq-offset …) is drawn *per record*.
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: float,
        num_iq_samples: int,
        elements_per_file: int,
        num_files: int,
        modulation: str,
        rng: np.random.Generator,
    ) -> None:
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.num_iq_samples = num_iq_samples
        self.elements_per_file = elements_per_file
        self.num_files = num_files
        self.modulation = modulation.upper()
        self.rng = rng

        os.makedirs(output_dir, exist_ok=True)

        class_list = ["BPSK", "QPSK", "8PSK", "16QAM", "NOISE"] if modulation == "ALL" else [modulation]

        total_elements = num_files * elements_per_file

        generate_metadata(
            output_dir,
            total_elements,
            sample_rate,
            class_list,
            num_iq_samples,
            elements_per_file,
            num_files,
            rng,
        )

        for file_idx in range(num_files):
            # --------------------------------------------------------------
            #   1) Assemble *elements_per_file* records into one huge vector
            # --------------------------------------------------------------
            i_buffer = np.empty(elements_per_file * num_iq_samples, dtype=np.float32)
            q_buffer = np.empty_like(i_buffer)

            # We'll also fill the *metadata* rows for THIS file later on.
            file_rows = []

            for elem_idx in range(elements_per_file):
                global_idx = file_idx * elements_per_file + elem_idx

                # ----------------------------------------------------------
                #   Pick a random class for *this* element (ignoring round-robin)
                # ----------------------------------------------------------
                label = rng.choice(class_list)

                # Random generation parameters -- stored later in the CSV
                snr_db = (
                    rng.uniform(5, 30) if label != "NOISE" else np.nan
                )
                amp_scale = rng.uniform(0.2, 1.0)

                # ----------------------------------------------------------
                #   2) Generate the complex waveform
                # ----------------------------------------------------------
                complex_vec = generate_random_modulated_samples(
                    num_iq_samples,
                    label,
                    rng,
                    snr_db=snr_db,
                    amp_scale=amp_scale,
                )

                # ----------------------------------------------------------
                #   3) Split into I/Q float vectors (interleaved in memory)
                # ----------------------------------------------------------
                start = elem_idx * num_iq_samples
                stop = start + num_iq_samples
                i_buffer[start:stop] = complex_vec.real.astype(np.float32)
                q_buffer[start:stop] = complex_vec.imag.astype(np.float32)

                # ----------------------------------------------------------
                #   4) Remember the per-record row for the CSV (optional)
                # ----------------------------------------------------------
                file_rows.append(
                    [
                        global_idx,
                        label,
                        class_list.index(label),
                        int(sample_rate),
                        f"{snr_db:.2f}" if not np.isnan(snr_db) else "",
                        f"{amp_scale:.3f}",
                    ]
                )

            # --------------------------------------------------------------
            #   5) Write *this* file's records into a *stereo* wav sink
            # --------------------------------------------------------------
            wav_path = output_dir / f"signal_{file_idx:04d}.wav"

            src_i = blocks.vector_source_f(i_buffer, False)  # left channel (I)
            src_q = blocks.vector_source_f(q_buffer, False)  # right channel (Q)

            wav_sink = blocks.wavfile_sink(
                str(wav_path),
                2,                       # stereo (I-Q)
                int(sample_rate),
                blocks.FORMAT_WAV,
                blocks.FORMAT_FLOAT,
            )

            tb = gr.top_block(f"WAV Generator File {file_idx}")
            tb.connect(src_i, (wav_sink, 0))
            tb.connect(src_q, (wav_sink, 1))

            tb.start()
            tb.wait()
            del tb

            # --------------------------------------------------------------
            #   6) Append per-file rows to the *global* CSV (lazy-append)
            # --------------------------------------------------------------
            meta_path = output_dir / "metadata.csv"
            with meta_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(file_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV dataset with random IQ samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (creates .wav files + metadata.csv + info.json)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1e6,
        help="Sample rate (Hz) for IQ data",
    )
    parser.add_argument(
        "--num-iq-samples",
        type=int,
        default=1024,
        help="Number of IQ samples PER element (complex samples)",
    )
    parser.add_argument(
        "--elements-per-file",
        type=int,
        default=1000,
        help="Number of elements (rows) per WAV file",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Total number of WAV files to generate",
    )
    parser.add_argument(
        "--modulation",
        type=str,
        default="BPSK",
        choices=["BPSK", "QPSK", "8PSK", "16QAM", "NOISE", "ALL"],
        help="Modulation type (or 'ALL' for a mixture)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.num_files <= 0 or args.elements_per_file <= 0 or args.num_iq_samples <= 0:
        parser.error("All counts must be positive integers")

    rng = get_rng(args.seed)

    print("\n=== Generating random IQ dataset ===")
    print(f"  Output directory : {args.output_dir}")
    print(f"  Total elements   : {args.num_files * args.elements_per_file}")
    print(f"  Samples/element  : {args.num_iq_samples}")
    print(f"  Elements/file    : {args.elements_per_file}")
    print(f"  Sample rate      : {args.sample_rate:.0f} Hz")
    print(f"  Modulation mode  : {args.modulation}")
    if args.seed is not None:
        print(f"  RNG seed          : {args.seed}")
    print("=====================================\n")

    IQGenerator(
        Path(args.output_dir),
        args.sample_rate,
        args.num_iq_samples,
        args.elements_per_file,
        args.num_files,
        args.modulation,
        rng,
    )

    print("\n✅  Done! WAV dataset generated in", args.output_dir)


if __name__ == "__main__":
    main()
