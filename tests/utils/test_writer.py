"""Unit Tests for writer utilities."""

import numpy as np
import yaml
import pytest

from torchsig.utils.writer import DatasetCreator
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults, default_dataset
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset


@pytest.mark.parametrize(
    "dataset_length, expect_error, multithreading",
    [
        (None, True, False),    # iterable dataset length cannot be inferred safely
        (10, False, False),     # non-multiple-of-batch_size check (batch_size=4)
        (7, False, True),       # multithreading test 
    ],
)
def test_DatasetCreator(tmp_path, dataset_length, expect_error, multithreading):
    # test that DatasetCreator can write dataset and metadata files, and 
    # that it restores caller-visible state after writing

    seed = 1234567890
    batch_size = 4

    ds = default_dataset(num_signals_max=3, num_signals_min=0)
    dl = WorkerSeedingDataLoader(ds, seed=seed, batch_size=batch_size)
    orig_target_labels = getattr(ds, "target_labels", None)
    orig_collate_fn = dl.collate_fn

    if expect_error:
        with pytest.raises(Exception):
            dc = DatasetCreator(
                dataloader=dl,
                dataset_length=dataset_length,
                root=tmp_path,
                overwrite=True,
                multithreading=multithreading,
            )
            dc.create()
        return
    else:
        dc = DatasetCreator(
            dataloader=dl,
            dataset_length=dataset_length,
            root=tmp_path,
            overwrite=True,
            multithreading=multithreading,
        )
        dc.create()

    dataset_info_path = tmp_path / "dataset_info.yaml"
    writer_info_path = tmp_path / "writer_info.yaml"
    assert dataset_info_path.exists()
    assert writer_info_path.exists()

    ds_yaml = yaml.safe_load(dataset_info_path.read_text()) or {}
    wr_yaml = yaml.safe_load(writer_info_path.read_text()) or {}

    assert ds_yaml["dataset_length"] == dataset_length
    assert ds_yaml["seed"] == seed
    assert "target_labels" in ds_yaml
    assert "dataset_metadata" in ds_yaml

    assert wr_yaml["complete"] is True
    assert wr_yaml["items_written"] == dataset_length

    # Invariants: writer should restore caller-visible state
    assert getattr(ds, "target_labels", None) == orig_target_labels
    assert dl.collate_fn == orig_collate_fn


def test_DatasetCreator_overwrite_false_skips_when_matching(tmp_path):
    # test that DatasetCreator with overwrite=False does not raise an error when dataset already exists and 
    # matches expected length, and that it does not modify existing files
    seed = 42
    ds = default_dataset(num_signals_max=1, num_signals_min=1)
    dl = WorkerSeedingDataLoader(ds, seed=seed, batch_size=2)

    # First generation
    DatasetCreator(dataloader=dl, dataset_length=6, root=tmp_path, overwrite=True, multithreading=False).create()

    # Sentinel file should remain untouched
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("do not delete")

    # Second run with overwrite=False should skip regeneration (no exception)
    DatasetCreator(dataloader=dl, dataset_length=6, root=tmp_path, overwrite=False, multithreading=False).create()
    assert sentinel.exists()
    assert sentinel.read_text() == "do not delete"


def test_DatasetCreator_overwrite_false_errors_if_incomplete(tmp_path):
    # Simulate partial/incomplete dataset on disk
    (tmp_path / "data.h5").write_text("marker")  # existence_probe should treat dataset as existing
    (tmp_path / "writer_info.yaml").write_text("complete: false\n")
    (tmp_path / "dataset_info.yaml").write_text(yaml.safe_dump({"dataset_length": 6, "seed": 1, "target_labels": None, "dataset_metadata": {}}))

    ds = default_dataset(num_signals_max=1, num_signals_min=1)
    dl = WorkerSeedingDataLoader(ds, seed=1, batch_size=2)

    dc = DatasetCreator(dataloader=dl, dataset_length=6, root=tmp_path, overwrite=False, multithreading=False)
    with pytest.raises(RuntimeError, match=r"partially exists"):
        dc.create()


@pytest.mark.parametrize(
    "dataset_length, multithreading",
    [
        (16, False),     # tiny dataset without multithreading
        (17, True),      # tiny dataset with multithreading
    ],
)
def test_DatasetCreator_tqdm_output(tmp_path, capsys, dataset_length, multithreading):
    # test that DatasetCreator tqdm output is produced and contains expected description and completion percentage

    seed = 12345
    batch_size = 4

    ds = default_dataset(num_signals_max=4, num_signals_min=0)
    dl = WorkerSeedingDataLoader(ds, seed=seed, batch_size=batch_size)
    dc = DatasetCreator(
        dataloader=dl,
        dataset_length=dataset_length,
        root=tmp_path,
        overwrite=True,
        multithreading=multithreading,
        tqdm_desc="Pytest TQDM Test",
    )
    dc.create()

    # tqdm outputs to stderr by default
    captured = capsys.readouterr()
    assert "Pytest TQDM Test" in captured.err
    assert "100%" in captured.err

@pytest.mark.full
@pytest.mark.parametrize(
    "dataset_length, multithreading",
    [
        (17, False),     # tiny dataset without multithreading
        (23, True),     # tiny dataset with multithreading
    ],
)
def test_writer_reproducibility(tmp_path, dataset_length, multithreading):
    # tests that two independent runs of DatasetCreator with the same seed and dataset length produce identical datasets 
    # on disk (YAML metadata and actual data files), and that the seed is correctly recorded in the YAML metadata
    seed = 123456
    batch_size = 2

    root0 = tmp_path / "run0"
    root1 = tmp_path / "run1"

    ds0 = default_dataset(num_signals_max=3, num_signals_min=1)
    dl0 = WorkerSeedingDataLoader(ds0, seed=seed, batch_size=batch_size)
    dl0.seed(seed)
    DatasetCreator(dataloader=dl0, dataset_length=dataset_length, root=root0, overwrite=True, multithreading=multithreading).create()

    ds1 = default_dataset(num_signals_max=3, num_signals_min=1)
    dl1 = WorkerSeedingDataLoader(ds1, seed=seed, batch_size=batch_size)
    dl1.seed(seed)
    DatasetCreator(dataloader=dl1, dataset_length=dataset_length, root=root1, overwrite=True, multithreading=multithreading).create()

    # YAML equivalence (stable keys)
    y0 = yaml.safe_load((root0 / "dataset_info.yaml").read_text()) or {}
    y1 = yaml.safe_load((root1 / "dataset_info.yaml").read_text()) or {}

    assert y0["seed"] == y1["seed"] == seed
    assert y0["dataset_length"] == y1["dataset_length"] == dataset_length

    # Disk data equivalence
    sds0 = StaticTorchSigDataset(root=str(root0), target_labels=["class_index"])
    sds1 = StaticTorchSigDataset(root=str(root1), target_labels=["class_index"])

    assert len(sds0) == len(sds1) == dataset_length

    for i in range(dataset_length):
        x0, m0 = sds0[i]
        x1, m1 = sds1[i]
        assert m0 == m1
        assert np.allclose(x0, x1, rtol=1e-6)

@pytest.mark.full
def test_writer_memory_growth(tmp_path):
    """Measure how RAM usage grows while writing datasets of increasing size.

    Writes three datasets (50 / 150 / 300 samples) using reduced 4096-sample
    IQ buffers and checks two orthogonal properties after each write:

    Structural invariant
        The total number of children registered in the Seedable hierarchy
        (dataset → generators → distributions → …) must be identical before
        and after writing.  Any increase would mean generated Signal objects
        are leaking into ``Seedable.children`` lists and will never be freed.

    RAM invariant
        The process RSS (resident set size) after ``DatasetCreator.create()``
        finishes must not grow faster than O(1) per sample.  Specifically the
        post-write RSS increase per sample must not be larger for the 300-
        sample run than for the 50-sample run, indicating that IQ arrays are
        being freed once writing is complete rather than accumulating.

    The per-run numbers are printed so they appear in the pytest log when
    running with ``-s`` and are useful for manual inspection.
    """
    import gc
    import tracemalloc
    from torchsig.signals.signal_types import Signal

    SEED = 27182
    BATCH_SIZE = 4
    # 4096-sample IQ buffers: 4096 × 8 bytes/complex64 ≈ 32 KiB per signal.
    # This keeps each run under ~10 s while still exercising the full write path.
    SMALL_META = {
        **TorchSigDefaults().default_dataset_metadata,
        "num_iq_samples_dataset": 4096,
        "signal_duration_in_samples_min": int(4096 * 0.8),
        "signal_duration_in_samples_max": 4096,
        "num_signals_max": 3,
        "num_signals_min": 1,
    }

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def current_rss_bytes() -> int:
        """Current RSS in bytes, read from /proc/self/status (Linux)."""
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024   # kB → bytes
        raise RuntimeError("VmRSS not found in /proc/self/status")

    def count_seedable_children(node, _seen: set | None = None) -> int:
        """Return total registered children in the Seedable subtree rooted at *node*."""
        if _seen is None:
            _seen = set()
        nid = id(node)
        if nid in _seen:
            return 0
        _seen.add(nid)
        total = len(node.children)
        for child in node.children:
            total += count_seedable_children(child, _seen)
        return total

    def has_signal_in_children(node, _seen: set | None = None) -> bool:
        """Return True if any Signal object is registered anywhere in the Seedable subtree."""
        if _seen is None:
            _seen = set()
        nid = id(node)
        if nid in _seen:
            return False
        _seen.add(nid)
        for child in node.children:
            if isinstance(child, Signal):
                return True
            if has_signal_in_children(child, _seen):
                return True
        return False

    # -------------------------------------------------------------------
    # Run writes and collect metrics
    # -------------------------------------------------------------------

    sizes = [50, 150, 300]
    metrics: dict[int, dict] = {}

    for size in sizes:
        root = tmp_path / f"ds_{size}"
        gc.collect()

        ds = TorchSigIterableDataset(metadata=SMALL_META)
        dl = WorkerSeedingDataLoader(ds, seed=SEED, batch_size=BATCH_SIZE)
        dl.seed(SEED)

        children_before = count_seedable_children(ds)

        tracemalloc.start()
        rss_before = current_rss_bytes()

        DatasetCreator(
            dataloader=dl,
            dataset_length=size,
            root=root,
            overwrite=True,
            multithreading=False,
        ).create()

        gc.collect()
        rss_after = current_rss_bytes()
        _, tracemalloc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        children_after = count_seedable_children(ds)
        signal_leaked = has_signal_in_children(ds)

        rss_increase = rss_after - rss_before
        metrics[size] = {
            "children_before": children_before,
            "children_after": children_after,
            "signal_leaked": signal_leaked,
            "rss_increase_bytes": rss_increase,
            "tracemalloc_peak_bytes": tracemalloc_peak,
        }

        print(
            "size=%3d: children %d → %d  |  RSS Δ=%+.0f KiB  |  tracemalloc peak=%.0f KiB",
            size, children_before, children_after,
            rss_increase / 1024, tracemalloc_peak / 1024,
        )

    # -------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------

    for size in sizes:
        m = metrics[size]

        # Structural: Seedable hierarchy must not grow during writing
        assert m["children_before"] == m["children_after"], (
            f"size={size}: Seedable tree grew by "
            f"{m['children_after'] - m['children_before']} nodes during writing. "
            "Generated Signal objects are leaking into Seedable.children lists."
        )

        # Structural: no Signal instance should appear anywhere in the tree
        assert not m["signal_leaked"], (
            f"size={size}: A Signal object was found inside a Seedable.children list "
            "after writing. The Seedable hierarchy is retaining completed samples."
        )

    # RAM: post-write RSS increase per sample must not grow with dataset size.
    # Compare the 300-sample run against the 50-sample run: if memory leaked
    # linearly with N, the per-sample cost would be identical (constant slope)
    # but still reveal the leak via an unusually large constant. A non-leaking
    # implementation should show a per-sample cost that is stable or decreasing
    # (amortised overhead), so we require the 300-sample run's per-sample RSS
    # is no worse than 4× the 50-sample run's (generous headroom for OS noise).
    rss_per_sample = {n: max(0, metrics[n]["rss_increase_bytes"]) / n for n in sizes}
    print("RSS per sample (bytes): %s", rss_per_sample)

    ratio_300_to_50 = (rss_per_sample[300] + 1) / (rss_per_sample[50] + 1)
    assert ratio_300_to_50 < 4.0, (
        f"RSS per sample grew {ratio_300_to_50:.1f}× from 50 → 300 samples "
        f"(per-sample bytes: 50→{rss_per_sample[50]:.0f}, "
        f"300→{rss_per_sample[300]:.0f}). "
        "This suggests signal IQ arrays are not being freed after writing."
    )
