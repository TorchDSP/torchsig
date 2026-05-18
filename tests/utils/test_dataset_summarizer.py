"""Unit Tests for dataset_summarizer"""

from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator, default_collate_fn
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.dataset_summarizer import DatasetSummary, summarize_dataset

import os
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend so plots don't block CI
matplotlib.use("Agg")

SEED = 1234567890
DATASET_LENGTH = 10

data_dir = Path.joinpath(Path(__file__).parent, "summarizer_data")


def setup_module(module):
    """Create a small static dataset on disk for summarizer tests."""
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    md = TorchSigDefaults().default_dataset_metadata
    # Use small IQ size for speed
    md["num_iq_samples_dataset"] = 4096
    md["signal_duration_in_samples_min"] = int(4096 * 0.8)
    md["signal_duration_in_samples_max"] = 4096
    md["fft_size"] = 64
    md["fft_stride"] = 64

    ds = TorchSigIterableDataset(metadata=md, seed=SEED)
    dl = WorkerSeedingDataLoader(ds, collate_fn=default_collate_fn, batch_size=16)
    dl.seed(SEED)

    dc = DatasetCreator(
        dataloader=dl,
        root=str(data_dir),
        dataset_length=DATASET_LENGTH,
        overwrite=True,
    )
    dc.create()


def teardown_module(module):
    """Clean up test data."""
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)


def test_summary_from_root():
    """DatasetSummary constructed from a dataset root path."""
    summary = DatasetSummary(root=str(data_dir))

    assert summary.dataset_length == DATASET_LENGTH
    assert len(summary.class_counts) > 0
    assert sum(summary.class_counts.values()) > 0

    # All four continuous metrics should be histogrammed
    for key in ("center_freq", "bandwidth", "duration", "snr_db"):
        assert key in summary.histograms
        counts, edges = summary.histograms[key]
        assert len(counts) > 0
        assert len(edges) == len(counts) + 1
        assert counts.sum() > 0

    # num_signals_per_sample should match dataset length
    assert len(summary.num_signals_per_sample) == DATASET_LENGTH


def test_summary_from_dataset():
    """DatasetSummary.from_dataset() with an already-loaded dataset."""
    dataset = StaticTorchSigDataset(root=str(data_dir), target_labels=None)
    summary = DatasetSummary.from_dataset(dataset)

    assert summary.dataset_length == DATASET_LENGTH
    assert sum(summary.class_counts.values()) > 0
    for key in ("center_freq", "bandwidth", "duration", "snr_db"):
        assert key in summary.histograms


def test_summary_custom_bins():
    """Custom per-metric bin counts are respected."""
    custom_bins = {"center_freq": 10, "bandwidth": 20, "duration": 15, "snr_db": 5}
    summary = DatasetSummary(root=str(data_dir), n_bins=custom_bins)

    for key, expected_n in custom_bins.items():
        counts, edges = summary.histograms[key]
        assert len(counts) == expected_n
        assert len(edges) == expected_n + 1


def test_summary_int_bins():
    """A single int for n_bins applies to all metrics."""
    summary = DatasetSummary(root=str(data_dir), n_bins=25)

    for key in ("center_freq", "bandwidth", "duration", "snr_db"):
        counts, _ = summary.histograms[key]
        assert len(counts) == 25


def test_plot_all_metrics():
    """plot() with default metrics produces a figure."""
    summary = DatasetSummary(root=str(data_dir))
    fig, axes = summary.plot()

    assert fig is not None
    assert len(axes) >= 5  # class + 4 continuous metrics
    plt.close(fig)


def test_plot_subset():
    """plot() with a specific subset of metrics."""
    summary = DatasetSummary(root=str(data_dir))
    fig, axes = summary.plot(metrics=["class", "snr_db"])

    assert len(axes) == 2
    plt.close(fig)


def test_plot_save(tmp_path):
    """plot() can save to a file."""
    summary = DatasetSummary(root=str(data_dir))
    save_path = str(tmp_path / "summary.png")
    fig, _ = summary.plot(save_path=save_path)

    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)


def test_plot_unknown_metric():
    """plot() raises ValueError for an unknown metric."""
    summary = DatasetSummary(root=str(data_dir))
    with pytest.raises(ValueError, match="Unknown metric"):
        summary.plot(metrics=["nonexistent"])


def test_summarize_dataset_convenience():
    """The summarize_dataset() convenience function returns a summary."""
    summary = summarize_dataset(root=str(data_dir))

    assert isinstance(summary, DatasetSummary)
    assert summary.dataset_length == DATASET_LENGTH
    assert sum(summary.class_counts.values()) > 0


def test_class_counts_match_total_signals():
    """Total class occurrences should equal sum of signals across all samples."""
    summary = DatasetSummary(root=str(data_dir))

    total_class_occurrences = sum(summary.class_counts.values())
    total_signals = int(summary.num_signals_per_sample.sum())

    assert total_class_occurrences == total_signals


def test_histogram_counts_match_total_signals():
    """Each continuous histogram's total count should equal total signals."""
    summary = DatasetSummary(root=str(data_dir))
    total_signals = int(summary.num_signals_per_sample.sum())

    for key in ("center_freq", "bandwidth", "duration", "snr_db"):
        counts, _ = summary.histograms[key]
        assert int(counts.sum()) == total_signals, f"{key} histogram count mismatch"