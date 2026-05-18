"""Summarize static TorchSig datasets written to disk.

Usage:
    summary = DatasetSummary("/path/to/dataset")
    summary.plot()
    summary.plot(metrics=["class", "snr_db"], save_path="summary.png")
"""

from __future__ import annotations

from collections import Counter
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from torchsig.datasets.datasets import StaticTorchSigDataset

DEF_N_BINS = 50

# Plot configuration: metric_key -> (title, xlabel)
_PLOT_CONFIG = {
    "class": ("Class Distribution", "Signal Modulation Class"),
    "center_freq": ("Center Frequency Distribution", "Center Frequency (Hz)"),
    "bandwidth": ("Bandwidth Distribution", "Bandwidth (Hz)"),
    "duration": ("Duration Distribution", "Duration (Samples)"),
    "snr_db": ("SNR Distribution", "Signal to Noise Ratio (dB)"),
    "num_signals": ("Signals per Sample Distribution", "Number of Signals"),
}


class DatasetSummary:
    """Summary statistics for a ``StaticTorchSigDataset``.

    Reads every sample once, extracts per-signal metadata from
    ``Signal.component_signals``, and histograms the results with numpy.

    Attributes:
        dataset_length: Number of samples in the dataset.
        class_counts: Counter mapping class_name -> occurrences.
        histograms: Dict of metric_key -> (counts, bin_edges) from np.histogram.
        num_signals_per_sample: Array of signal counts per sample.
    """

    def __init__(
        self,
        root: str,
        n_bins: int | dict[str, int] = DEF_N_BINS,
    ) -> None:
        """Summarize a static dataset on disk.

        Args:
            root: Path to the dataset directory (containing ``data.h5``).
            n_bins: Number of histogram bins. Either a single int applied to all
                metrics, or a dict mapping metric names to bin counts.
                Defaults to 50.
        """
        dataset = StaticTorchSigDataset(root=root, target_labels=None)
        self._build(dataset, n_bins)

    @classmethod
    def from_dataset(
        cls,
        dataset: StaticTorchSigDataset,
        n_bins: int | dict[str, int] = DEF_N_BINS,
    ) -> DatasetSummary:
        """Create a summary from an already-loaded dataset.

        The dataset must return ``Signal`` objects (i.e. ``target_labels=None``).

        Args:
            dataset: A loaded ``StaticTorchSigDataset``.
            n_bins: Number of histogram bins (int or per-metric dict).

        Returns:
            A populated ``DatasetSummary``.
        """
        instance = cls.__new__(cls)
        instance._build(dataset, n_bins)
        return instance

    def _build(
        self,
        dataset: StaticTorchSigDataset,
        n_bins: int | dict[str, int],
    ) -> None:
        """Single-pass collection and histogramming."""
        if isinstance(n_bins, int):
            n_bins = dict.fromkeys(("center_freq", "bandwidth", "duration", "snr_db"), n_bins)

        self.dataset_length = len(dataset)
        if self.dataset_length == 0:
            raise ValueError("Cannot summarize an empty dataset.")

        # Collect raw values in a single pass
        class_names: list[str] = []
        center_freqs: list[float] = []
        bandwidths: list[float] = []
        durations: list[float] = []
        snrs: list[float] = []
        n_signals: list[int] = []

        for i in tqdm(range(self.dataset_length)):
            signal = dataset[i]
            components = signal.component_signals or [signal]
            n_signals.append(len(components))
            for comp in components:
                class_names.append(comp.class_name)
                center_freqs.append(comp.center_freq)
                bandwidths.append(comp.bandwidth)
                durations.append(comp.duration_in_samples)
                snrs.append(comp.snr_db)

        # Class counts (preserved as an ordered Counter)
        self.class_counts = Counter(class_names)

        # Histogram continuous metrics
        self.histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw = {
            "center_freq": center_freqs,
            "bandwidth": bandwidths,
            "duration": durations,
            "snr_db": snrs,
        }
        for key, values in raw.items():
            arr = np.asarray(values)
            bins = n_bins.get(key, DEF_N_BINS)
            self.histograms[key] = np.histogram(arr, bins=bins)

        # Signals-per-sample histogram (integer bins)
        self.num_signals_per_sample = np.asarray(n_signals)
        if self.num_signals_per_sample.min() != self.num_signals_per_sample.max():
            lo, hi = int(self.num_signals_per_sample.min()), int(self.num_signals_per_sample.max())
            edges = np.arange(lo, hi + 2) - 0.5  # center bins on integers
            self.histograms["num_signals"] = np.histogram(self.num_signals_per_sample, bins=edges)

    def plot(
        self,
        metrics: list[str] | None = None,
        max_cols: int = 2,
        width_per_plot: int = 15,
        height_per_plot: int = 10,
        round_labels: int = 2,
        save_path: str | None = None,
    ):
        """Plot summary histograms.

        Args:
            metrics: Which metrics to plot (keys of ``_PLOT_CONFIG``).
                Defaults to all available metrics.
            max_cols: Maximum subplot columns per row.
            width_per_plot: Width in inches per subplot.
            height_per_plot: Height in inches per subplot.
            round_labels: Decimal places for bin-edge labels.
            save_path: If provided, save the figure to this path.

        Returns:
            The matplotlib ``(fig, axes)`` tuple.
        """
        if metrics is None:
            metrics = ["class", *self.histograms]

        n = len(metrics)
        cols = min(n, max_cols)
        rows = ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * width_per_plot, rows * height_per_plot))
        fig.set_layout_engine("constrained")

        # Normalize axes to a flat list
        axes = [axes] if n == 1 else np.asarray(axes).flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            title, xlabel = _PLOT_CONFIG.get(metric, (metric, metric))

            if metric == "class":
                names = list(self.class_counts.keys())
                counts = list(self.class_counts.values())
                ax.bar(range(len(names)), counts)
                ax.set_xticks(range(len(names)), names, rotation=90)
            elif metric in self.histograms:
                counts, edges = self.histograms[metric]
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.bar(range(len(counts)), counts)
                # Show a subset of tick labels to avoid crowding
                step = max(1, len(centers) // 10)
                tick_pos = list(range(0, len(centers), step))
                tick_labels = [round(centers[j], round_labels) for j in tick_pos]
                ax.set_xticks(tick_pos, tick_labels, rotation=90)
            else:
                raise ValueError(f"Unknown metric: {metric!r}")

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Occurrences")

        # Remove unused subplots
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        if save_path:
            plt.savefig(save_path)

        plt.show()
        return fig, axes[:n]


def summarize_dataset(
    root: str,
    n_bins: int | dict[str, int] = DEF_N_BINS,
    save_path: str | None = None,
) -> DatasetSummary:
    """Convenience function: summarize and plot a static dataset.

    Args:
        root: Path to the dataset directory.
        n_bins: Histogram bin count (int or per-metric dict).
        save_path: Optional path to save the figure.

    Returns:
        The ``DatasetSummary`` instance.
    """
    summary = DatasetSummary(root, n_bins=n_bins)
    fig, _ = summary.plot(save_path=save_path)
    plt.close(fig)
    return summary
