import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Any

# Signal Metadata type checking

signal_metadata_floats = ["center_freq", "bandwidth", "snr_db", "sample_rate", "start", "stop", "duration", "upper_freq", "lower_freq", "oversampling_rate"]
signal_metadata_ints = ["start_in_samples", "duration_in_samples", "class_index", "dataset_length", "stop_in_samples"]
signal_metadata_strs = ["class_name"]


def save_sample(savepath: str, data: np.ndarray, targets: List[dict], sample_rate: float):
    # plot sample
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1)
    xmin = 0
    xmax = 1
    ymin = -sample_rate / 2
    ymax = sample_rate / 2
    pos = ax.imshow(data, extent=[xmin, xmax, ymin, ymax], aspect="auto", cmap="Wistia", vmin=-5)
    fig.colorbar(pos, ax=ax)

    title = "labels: "

    for t in targets:

        ax.plot([t["start"], t["start"]], [t["lower_freq"], t["upper_freq"]], "b", alpha=0.5)
        ax.plot([t["stop"], t["stop"]], [t["lower_freq"], t["upper_freq"]], "b", alpha=0.5)
        ax.plot([t["start"], t["stop"]], [t["lower_freq"], t["lower_freq"]], "b", alpha=0.5)
        ax.plot([t["start"], t["stop"]], [t["upper_freq"], t["upper_freq"]], "b", alpha=0.5)
        textDisplay = str(t["class_name"]) + ", SNR = " + str(t["snr_db"]) + " dB"
        ax.text(t["start"], t["lower_freq"], textDisplay, bbox=dict(facecolor="w", alpha=0.5, linewidth=0))
        ax.set_xlim([0, 1])
        ax.set_ylim([-sample_rate / 2, sample_rate / 2])
        title = f"{title}{t['class_name']} "

    fig.suptitle(title, fontsize=16)
    plt.savefig(savepath)
