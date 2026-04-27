#!/usr/bin/env python3
"""Benchmark TorchSig HDF5Writer compression propagation, walltime, and file size."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import time
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5 import HDF5Writer


def make_float32_signals(count: int = 10000, length: int = 1024, seed: int = 0) -> list[Signal]:
    """Structured float32 signals with sinusoid + low-rank pattern + noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)

    freqs = rng.uniform(5, 80, size=count).astype(np.float32)
    phases = rng.uniform(0, 2 * np.pi, size=count).astype(np.float32)
    amps = rng.uniform(0.2, 1.0, size=count).astype(np.float32)

    base = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * t + phases[:, None])
    trend = (0.1 * np.sin(2 * np.pi * 3 * t) + 0.05 * np.cos(2 * np.pi * 11 * t)).astype(np.float32)

    bank = np.stack([np.sin(2 * np.pi * k * t).astype(np.float32) for k in (7, 13, 29, 43)])
    pattern = bank[rng.integers(0, len(bank), size=count)]

    noise = rng.normal(0.0, 0.03, size=(count, length)).astype(np.float32)
    matrix = (base + trend + 0.15 * pattern + noise).astype(np.float32)

    return [Signal(data=row) for row in matrix]


def make_int16_signals(count: int = 1000, length: int = 1024, seed: int = 1) -> list[Signal]:
    """Structured int16 signals with sparse bursts and tiny quantization noise."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((count, length), dtype=np.int16)

    for i in range(count):
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(0, length - 128))
            dur = int(rng.integers(32, 128))
            value = int(rng.choice([512, 1024, 2048, -512, -1024, -2048]))

            burst = np.full(dur, value, dtype=np.int16)
            if dur > 8:
                burst[::8] = value // 2
            matrix[i, start : start + dur] = burst

        matrix[i] += rng.integers(-4, 5, size=length, dtype=np.int16)

    return [Signal(data=row) for row in matrix]


def dcpl_filters(dset: h5py.Dataset) -> str:
    """Return a compact string describing the dataset creation filter pipeline."""
    plist = dset.id.get_create_plist()
    filters = []

    for i in range(plist.get_nfilters()):
        filt_id, _flags, values, name = plist.get_filter(i)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        filters.append(f"{name}:{filt_id}:{tuple(int(v) for v in values)}")

    return "|".join(filters) if filters else ""


def inspect_data_group(
    h5_path: Path,
    requested_compression: str,
    requested_level: int | None,
    requested_shuffle: bool,
    requested_fletcher32: bool,
) -> dict[str, object]:
    """Inspect /data/* to determine whether requested options propagated."""
    with h5py.File(h5_path, "r") as handle:
        data_group = handle["data"]
        keys = sorted(data_group.keys(), key=int)
        if not keys:
            raise RuntimeError("No datasets found under /data")

        ref = data_group[keys[0]]
        ref_state = (
            ref.compression,
            ref.compression_opts,
            ref.shuffle,
            ref.fletcher32,
            ref.chunks,
            dcpl_filters(ref),
        )

        consistent = True
        for key in keys[1:]:
            dset = data_group[key]
            state = (
                dset.compression,
                dset.compression_opts,
                dset.shuffle,
                dset.fletcher32,
                dset.chunks,
                dcpl_filters(dset),
            )
            if state != ref_state:
                consistent = False
                break

        expected_level = None if requested_compression == "lzf" else requested_level
        propagated = (
            ref.compression == requested_compression
            and ref.compression_opts == expected_level
            and ref.shuffle == requested_shuffle
            and ref.fletcher32 == requested_fletcher32
            and ref.chunks is not None
            and consistent
        )

        return {
            "actual_compression": ref.compression,
            "actual_compression_opts": ref.compression_opts,
            "actual_shuffle": ref.shuffle,
            "actual_fletcher32": ref.fletcher32,
            "actual_chunks": tuple(ref.chunks) if ref.chunks is not None else None,
            "actual_filter_pipeline": dcpl_filters(ref),
            "consistent_across_data_group": consistent,
            "propagated": propagated,
        }


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean and sample std; std=0 for a single value."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    outdir: Path,
    dataset_name: str,
    dataset_shape: str,
    dtype_name: str,
    signals: list[Signal],
    compression: str,
    compression_opts: int | None,
    repeat: int,
    shuffle: bool,
    fletcher32: bool,
    max_batches_in_memory: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run one dataset/compression case N times."""
    raw_rows: list[dict[str, object]] = []

    for rep in range(1, repeat + 1):
        run_dir = outdir / "runs" / f"{dataset_name}__{compression}_{compression_opts if compression_opts is not None else 'none'}__rep{rep}"
        if run_dir.exists():
            shutil.rmtree(run_dir)

        writer = HDF5Writer(
            root=str(run_dir),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            fletcher32=fletcher32,
            max_batches_in_memory=max_batches_in_memory,
        )

        h5_path = run_dir / "data.h5"
        row: dict[str, object] = {
            "dataset_name": dataset_name,
            "dataset_shape": dataset_shape,
            "dtype": dtype_name,
            "requested_compression": compression,
            "requested_level": compression_opts,
            "repeat_index": rep,
            "requested_shuffle": shuffle,
            "requested_fletcher32": fletcher32,
            "max_batches_in_memory": max_batches_in_memory,
        }

        try:
            t0 = time.perf_counter()
            writer.setup()
            writer.write(0, signals)
            writer.teardown()  # include flush + close in walltime
            walltime_s = time.perf_counter() - t0

            row["walltime_s"] = walltime_s
            row["file_size_bytes"] = h5_path.stat().st_size
            row.update(
                inspect_data_group(
                    h5_path=h5_path,
                    requested_compression=compression,
                    requested_level=compression_opts,
                    requested_shuffle=shuffle,
                    requested_fletcher32=fletcher32,
                )
            )
            row["error"] = ""
        except Exception as exc:  # record failures instead of hiding them
            row["walltime_s"] = ""
            row["file_size_bytes"] = ""
            row["actual_compression"] = ""
            row["actual_compression_opts"] = ""
            row["actual_shuffle"] = ""
            row["actual_fletcher32"] = ""
            row["actual_chunks"] = ""
            row["actual_filter_pipeline"] = ""
            row["consistent_across_data_group"] = ""
            row["propagated"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            try:
                writer.teardown()
            except Exception:
                pass

        raw_rows.append(row)

    ok_rows = [r for r in raw_rows if not r["error"]]
    walltimes = [float(r["walltime_s"]) for r in ok_rows]
    sizes = [float(r["file_size_bytes"]) for r in ok_rows]
    wall_mean, wall_std = mean_std(walltimes)
    size_mean, size_std = mean_std(sizes)

    summary = {
        "dataset_name": dataset_name,
        "dataset_shape": dataset_shape,
        "dtype": dtype_name,
        "requested_compression": compression,
        "requested_level": compression_opts,
        "requested_shuffle": shuffle,
        "requested_fletcher32": fletcher32,
        "repeat": repeat,
        "successful_runs": len(ok_rows),
        "walltime_mean_s": wall_mean,
        "walltime_std_s": wall_std,
        "file_size_mean_bytes": size_mean,
        "file_size_std_bytes": size_std,
        "propagated_all_runs": all(bool(r["propagated"]) for r in ok_rows) if ok_rows else False,
        "actual_compression_first": ok_rows[0]["actual_compression"] if ok_rows else "",
        "actual_filter_pipeline_first": ok_rows[0]["actual_filter_pipeline"] if ok_rows else "",
        "errors": " | ".join(str(r["error"]) for r in raw_rows if r["error"]),
    }
    return raw_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("torchsig_hdf5_bench"))
    parser.add_argument("--repeat", type=int, default=5, help="Number of repetitions per case.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-batches-in-memory",
        type=int,
        default=1,
        help="Use 1 for deterministic single-batch timing.",
    )

    parser.set_defaults(shuffle=True, fletcher32=True)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--fletcher32", dest="fletcher32", action="store_true")
    parser.add_argument("--no-fletcher32", dest="fletcher32", action="store_false")

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("float32_10000x1024", "10000x1024", "float32", make_float32_signals(seed=args.seed)),
        ("int16_1000x1024", "1000x1024", "int16", make_int16_signals(seed=args.seed + 1)),
    ]
    compression_cases = [
        ("gzip", 1),
        ("gzip", 6),
        ("gzip", 9),
        ("lzf", None),
    ]

    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for dataset_name, dataset_shape, dtype_name, signals in tqdm(datasets, desc="Processing datasets"):
        for compression, level in tqdm(compression_cases, desc="Processing compression cases"):
            raw_case_rows, summary = run_case(
                outdir=args.outdir,
                dataset_name=dataset_name,
                dataset_shape=dataset_shape,
                dtype_name=dtype_name,
                signals=signals,
                compression=compression,
                compression_opts=level,
                repeat=args.repeat,
                shuffle=args.shuffle,
                fletcher32=args.fletcher32,
                max_batches_in_memory=args.max_batches_in_memory,
            )
            raw_rows.extend(raw_case_rows)
            summary_rows.append(summary)

    raw_csv = args.outdir / "raw_results.csv"
    summary_csv = args.outdir / "summary_results.csv"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)

    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")
    print()

    for row in summary_rows:
        print(
            f"{row['dataset_name']:>16} | "
            f"{row['requested_compression']:>4} {str(row['requested_level']):>4} | "
            f"walltime {row['walltime_mean_s']:.6f} ± {row['walltime_std_s']:.6f} s | "
            f"size {row['file_size_mean_bytes']:.0f} ± {row['file_size_std_bytes']:.0f} B | "
            f"propagated={row['propagated_all_runs']} | "
            f"actual={row['actual_compression_first']!r}"
        )


if __name__ == "__main__":
    main()
