import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.scripts import generate_gnuradio_examples as generator_module
from examples.scripts.generate_gnuradio_examples import (
    METADATA_FIELDS,
    IQDatasetGenerator,
    _derive_record_seed,
    _execute_record,
)


def _generator(root: Path, *, num_workers: int = 1, log_level: int | str = logging.INFO):
    return IQDatasetGenerator(
        root=root,
        modulations={"BPSK": generator_module.np.array([1 + 0j, -1 + 0j])},
        snr_db=[-5, 5],
        duration_s=0.001,
        audio_rate=1_000,
        num_workers=num_workers,
        log_level=log_level,
    )


def _create_completed_records(generator: IQDatasetGenerator) -> None:
    for spec in generator.build_record_specs():
        path = generator.root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"record")


def test_record_seed_is_stable_across_hash_seeds():
    command = "from examples.scripts.generate_gnuradio_examples import _derive_record_seed; print(_derive_record_seed(123, 'BPSK', -5))"
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.check_output(  # noqa: S603
                [sys.executable, "-c", command],
                cwd=Path(__file__).parents[1],
                env=environment,
                text=True,
            ).strip()
        )

    assert outputs[0] == outputs[1]
    assert int(outputs[0]) == 1546504995
    assert int(outputs[0]) == _derive_record_seed(123, "BPSK", -5)


def test_record_specs_are_stable_ordered_and_independently_seeded(tmp_path):
    first = _generator(tmp_path / "first").build_record_specs()
    second = _generator(tmp_path / "second").build_record_specs()

    assert first == second
    assert [spec.index for spec in first] == [0, 1]
    assert [spec.snr_db for spec in first] == [-5.0, 5.0]
    assert len({spec.seed for spec in first}) == len(first)


@pytest.mark.parametrize("num_workers", [0, -1, True, 1.5])
def test_invalid_worker_counts_are_rejected(tmp_path, num_workers):
    with pytest.raises(ValueError, match="positive integer"):
        IQDatasetGenerator(root=tmp_path, num_workers=num_workers)


def test_default_worker_count_preserves_serial_execution(tmp_path):
    assert IQDatasetGenerator(root=tmp_path).num_workers == 1


def test_named_log_level_is_normalized(tmp_path):
    assert IQDatasetGenerator(root=tmp_path, log_level="warning").log_level == logging.WARNING


def test_invalid_log_level_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown log level"):
        IQDatasetGenerator(root=tmp_path, log_level="verbose")


def test_serial_and_process_pool_sidecars_are_identical(tmp_path):
    serial = _generator(tmp_path / "serial")
    parallel = _generator(tmp_path / "parallel", num_workers=2)
    _create_completed_records(serial)
    _create_completed_records(parallel)

    serial.generate()
    parallel.generate()

    assert (serial.root / "metadata.csv").read_text() == (parallel.root / "metadata.csv").read_text()
    assert (serial.root / "info.json").read_text() == (parallel.root / "info.json").read_text()
    assert serial.metadata_rows == parallel.metadata_rows

    with (parallel.root / "metadata.csv").open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == METADATA_FIELDS
    assert rows[0]["file_path"].startswith("BPSK/")
    assert rows[0]["start_frame"] == "0"
    assert rows[0]["num_frames"] == "1"
    assert rows[0]["expected_sample_rate"] == "1000"
    assert rows[0]["channel_count"] == "2"

    info = json.loads((parallel.root / "info.json").read_text())
    assert info["size"] == 2
    assert info["sidecar_schema_version"] == "1.0"


def test_failed_record_removes_temporary_output_and_skips_sidecars(tmp_path, monkeypatch):
    generator = _generator(tmp_path)

    def fail_after_partial_write(mod_name, snr_db, out_path, record_seed=None):
        Path(out_path).write_bytes(b"partial")
        raise RuntimeError("generation failed")

    monkeypatch.setattr(generator, "generate_one", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="generation failed"):
        generator.generate()

    assert not (tmp_path / "metadata.csv").exists()
    assert not (tmp_path / "info.json").exists()
    assert not list(tmp_path.rglob("*.wav"))


def test_record_publication_is_atomic_and_uses_explicit_seed(tmp_path, monkeypatch):
    generator = _generator(tmp_path)
    spec = generator.build_record_specs()[0]
    calls = []

    def write_record(mod_name, snr_db, out_path, record_seed=None):
        calls.append((mod_name, snr_db, record_seed))
        Path(out_path).write_bytes(b"complete")

    monkeypatch.setattr(generator, "generate_one", write_record)
    result = _execute_record(generator, spec)

    assert calls == [(spec.mod_name, spec.snr_db, spec.seed)]
    assert result.spec == spec
    assert (tmp_path / spec.relative_path).read_bytes() == b"complete"
    assert not list((tmp_path / spec.mod_name).glob(".*"))


def test_generation_uses_module_logging(tmp_path, caplog):
    generator = _generator(tmp_path, log_level=logging.WARNING)
    _create_completed_records(generator)

    with caplog.at_level(logging.WARNING, logger=generator_module.__name__):
        generator.generate()

    assert "Dataset generation complete" in caplog.text
    assert "Metadata:" in caplog.text
