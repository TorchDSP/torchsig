"""Tests for the standalone Stage-2 materialization benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from torchsig.datasets import StructuredHDF5Dataset, materialize_structured_dataset
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers import HomogeneousHDF5Writer

_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "benchmark_stage2_materialization.py"
_SPEC = importlib.util.spec_from_file_location("stage2_benchmark", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
stage2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stage2
_SPEC.loader.exec_module(stage2)


def _write_source(root: Path, count: int = 10) -> None:
    signals = [Signal(data=np.exp(2j * np.pi * np.arange(256) * (index + 1) / 256).astype(np.complex64), class_index=index % 4) for index in range(count)]
    with HomogeneousHDF5Writer(root) as writer:
        writer.write(0, signals)


def test_fallback_pipeline_static_reader_and_materialized_equivalence(tmp_path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "structured"
    _write_source(source_root)
    online = stage2.OnlineStage2Dataset(source_root, None, 8)

    sample = online[0]
    assert set(sample) == {"features", "labels"}
    assert sample["features"]["iq"].shape == (2, 256)
    assert sample["features"]["spectral"].shape == (128,)
    assert sample["features"]["amplitude_histogram"].dtype == np.float32
    assert sample["labels"]["class_index"].dtype == np.int64

    materialized = materialize_structured_dataset(online, output_root, batch_size=4, progress=False)
    materialized.close()
    stored = StructuredHDF5Dataset(output_root)
    try:
        for index in (0, 3, 7):
            stage2._assert_equivalent(online[index], stored[index])
        stage2._assert_model_equivalent([online[0], online[1]], [stored[0], stored[1]], None, 8, 4)
    finally:
        stored.close()


def test_required_fallback_workload_schemas(tmp_path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root)

    for workload in ("iq", "spectrogram", "detection"):
        online = stage2.OnlineStage2Dataset(source_root, None, 2, workload)
        sample = online[0]
        if workload == "iq":
            assert sample["features"].shape == (2, 256)
            assert sample["labels"].shape == ()
        elif workload == "spectrogram":
            assert sample["features"].shape == (2, 64, 64)
        else:
            assert sample["labels"]["boxes"].shape == (4, 4)
            assert sample["labels"]["classes"].shape == (4,)
            assert sample["labels"]["valid"].dtype == np.bool_
        stored = materialize_structured_dataset(
            online,
            tmp_path / f"structured-{workload}",
            batch_size=2,
            progress=False,
        )
        try:
            stage2._assert_equivalent(online[0], stored[0])
        finally:
            stored.close()
            online.source.reader.teardown()


def test_storage_estimate_and_split_indices_are_deterministic() -> None:
    sample = {"x": np.zeros((2, 4), dtype=np.float32), "y": np.int64(1)}
    assert stage2.estimate_storage_bytes(sample, samples=10, copies=2) == (32 + 8) * 20
    assert stage2._split_indices(100, 17) == stage2._split_indices(100, 17)
    splits = stage2._split_indices(100, 17)
    assert tuple(map(len, splits.values())) == (80, 10, 10)


def test_aggregation_break_even_and_recommendation() -> None:
    results = []
    for repetition, online_rate, materialized_rate, ceiling_rate in ((0, 100.0, 190.0, 220.0), (1, 110.0, 200.0, 230.0), (2, 105.0, 195.0, 225.0)):
        common = {
            "access": "shuffled",
            "workers": 4,
            "repetition": repetition,
            "dataloader_samples_per_second": 500.0,
            "startup_seconds": 0.1,
            "file_size": 1024**3,
        }
        results.extend(
            [
                {
                    **common,
                    "pipeline": "online",
                    "compression": "n/a",
                    "chunk_samples": 0,
                    "layout": "full",
                    "samples_per_second": online_rate,
                    "epoch_seconds": 1000 / online_rate,
                    "materialization_seconds": 0.0,
                },
                {
                    **common,
                    "pipeline": "materialized",
                    "compression": "none",
                    "chunk_samples": 1,
                    "layout": "full",
                    "samples_per_second": materialized_rate,
                    "epoch_seconds": 1000 / materialized_rate,
                    "materialization_seconds": 10.0,
                },
                {
                    **common,
                    "pipeline": "device_only",
                    "compression": "memory",
                    "chunk_samples": 0,
                    "layout": "full",
                    "samples_per_second": ceiling_rate,
                    "epoch_seconds": 1000 / ceiling_rate,
                    "materialization_seconds": 0.0,
                },
            ]
        )

    aggregates = stage2.aggregate_results(results)
    materialized = next(row for row in aggregates if row["pipeline"] == "materialized")
    assert materialized["speedup_vs_online"] > 1.8
    assert materialized["break_even_epochs"] is not None
    recommendation = stage2.recommend(aggregates)
    assert "target (1.50x shuffled speedup): met" in recommendation[0]
    assert "compression=none" in recommendation[1]
    assert any("device-only ceiling" in line for line in recommendation)
    assert "stability (<=5% standard deviation): yes" in recommendation[-2]
    assert recommendation[-1] == "Physical split separation was not measured."


def test_single_repetition_does_not_claim_stability() -> None:
    result = {
        "pipeline": "materialized",
        "access": "shuffled",
        "workers": 0,
        "compression": "none",
        "chunk_samples": 1,
        "layout": "full",
        "samples_per_second": 100.0,
        "dataloader_samples_per_second": 120.0,
        "startup_seconds": 0.1,
        "epoch_seconds": 1.0,
        "materialization_seconds": 2.0,
        "file_size": 1024,
    }
    recommendation = stage2.recommend(stage2.aggregate_results([result]))
    assert "not established" in recommendation[-2]


def test_environment_metadata_is_reviewable() -> None:
    environment = stage2._environment_metadata()
    assert environment["python"]
    assert environment["platform"]
    assert environment["logical_cpu_count"] > 0


def test_cli_smoke(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / "outputs"
    results = tmp_path / "results.json"

    def fake_run_fresh(case, _scratch):
        return {
            **stage2.asdict(case),
            "device": "cpu",
            "startup_seconds": 0.001,
            "measured_steps": 1,
            "elapsed_seconds": 0.01,
            "seconds_per_step": 0.01,
            "steps_per_second": 100.0,
            "samples_per_second": 200.0,
            "dataloader_samples_per_second": 300.0,
            "process_cpu_seconds": 0.01,
            "process_cpu_utilization_percent": 100.0,
            "peak_host_memory_bytes": 1024,
            "peak_gpu_memory_bytes": 0,
            "warm_filesystem_cache": True,
            "epoch_seconds": 0.02,
        }

    monkeypatch.setattr(stage2, "_run_fresh", fake_run_fresh)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--output-dir",
            str(output_root),
            "--samples",
            "6",
            "--synthetic-iq-length",
            "64",
            "--batch-size",
            "2",
            "--worker-counts",
            "0",
            "--chunk-sizes",
            "1",
            "--compressions",
            "none",
            "--warmup-steps",
            "0",
            "--benchmark-steps",
            "1",
            "--repetitions",
            "1",
            "--split-layouts",
            "full",
            "--results",
            str(results),
            "--yes",
        ],
    )

    stage2.main()

    payload = json.loads(results.read_text())
    assert len(payload["measurements"]) == 5
    assert payload["failures"] == []
    assert payload["aggregates"]
    assert results.with_suffix(".csv").exists()
