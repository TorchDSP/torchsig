# # Example 01 - Modulations Dataset
# This notebook steps through an example of how to use `torchsig` to instantiate a `SignalDataset` containing 53 unique modulations. The notebook then plots the signals using `Visualizers` for both the IQ and Spectrogram representations of the dataset. The end of the notebook then shows how the instantiated dataset can be saved to an LMDB static dataset for standalone research, experimentation, and/or analysis.

# ----
# ### Import Libraries
# First, import all the necessary public libraries as well as a few classes from the `torchsig` toolkit.

from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.dataset import SignalDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
import numpy as np
import pickle
import lmdb

# ----
# ### Instantiate Modulations Dataset
# Next, instantiate the `ModulationsDataset` by passing in the desired classes, a boolean specifying whether to use the class name or index as the label, the desired level of signal impairments/augmentations, the number of IQ samples per example, and the total number of samples. Note that the total number of samples will be divided evenly among the class list (for example, `num_samples=5300` will result in 100x samples of each of the 53 modulation classes). Also note that the classes input parameter can be omitted if all classes are desired.
#
# If all classes are included at `level=0` (clean signals), all signals will occupy roughly half of the returned signal bandwidth except for the FSK and MSK modulations. These two subfamilies do not contain any pulse shaping, and as such are returned at roughly 1/8th occupied bandwidth for the main lobe. At the higher impairment levels, there is a randomized low pass filter applied at the 8x oversampled rate to suppress the sidelobes prior to downsampling to roughly the same half bandwidth target as the remaining signals.
#
# Within the OFDM family, there are 12 subclasses pertaining to the number of subcarriers present within the OFDM signal. These subcarriers are the powers of 2 from 64 to 2048 as well as the LTE specifications values of 72, 180, 300, 600, 900, and 1200. The DC subcarrier is randomly on or off throughout all subcarrier counts. The subcarrier modulations are divided into two categories: 1) randomly select a single modulation from the list: `bpsk, qpsk, 16qam, 64qam, 256qam, and 1024qam` and modulate all subcarriers with the random selection; and 2) randomly select a modulation from the same list for each subcarrier independently. The subcarrier modulations are not included in any of the labels for future classification tasks. In addition to these randomizations, the cyclic prefix ratio is also randomly selected between discrete values of 1/8 and 1/4, and it is also not included in the labels at this time. As a final randomization with the OFDM signals, two distinct sidelobe suppression techniques are evenly sampled from to smooth the discontinuities at the symbol boundaries: 1) apply a window, and 2) apply a low pass filter.

classes = [
    "ook",
    "bpsk",
    "4pam",
    "4ask",
    "qpsk",
    "8pam",
    "8ask",
    "8psk",
    "16qam",
    "16pam",
    "16ask",
    "16psk",
    "32qam",
    "32qam_cross",
    "32pam",
    "32ask",
    "32psk",
    "64qam",
    "64pam",
    "64ask",
    "64psk",
    "128qam_cross",
    "256qam",
    "512qam_cross",
    "1024qam",
    "2fsk",
    "2gfsk",
    "2msk",
    "2gmsk",
    "4fsk",
    "4gfsk",
    "4msk",
    "4gmsk",
    "8fsk",
    "8gfsk",
    "8msk",
    "8gmsk",
    "16fsk",
    "16gfsk",
    "16msk",
    "16gmsk",
    "ofdm-64",
    "ofdm-72",
    "ofdm-128",
    "ofdm-180",
    "ofdm-256",
    "ofdm-300",
    "ofdm-512",
    "ofdm-600",
    "ofdm-900",
    "ofdm-1024",
    "ofdm-1200",
    "ofdm-2048",
]
num_classes = len(classes)
level = 0
include_snr = True

# Seed the dataset instantiation for reproduceability
pl.seed_everything(1234567891)

dataset = ModulationsDataset(
    classes=classes,
    use_class_idx=False,
    level=level,
    num_iq_samples=4096,
    num_samples=int(num_classes * 100),
    include_snr=include_snr,
)

idx = 600
if include_snr:
    data, (modulation, snr) = dataset[idx]
else:
    data, modulation = dataset[idx]

print("Dataset length: {}".format(len(dataset)))
print("Number of classes: {}".format(num_classes))
print("Data shape: {}".format(data.shape))
print("Example modulation: {}".format(modulation))
if include_snr:
    print("SNR: {}".format(snr))


# ----
# ### Plot Subset to Verify
# The `IQVisualizer` and the `SpectrogramVisualizer` can be passed a `Dataloader` and plot visualizations of the dataset. The `batch_size` of the `DataLoader` determines how many examples to plot for each iteration over the visualizer. Note that the dataset itself can be indexed and plotted sequentially using any familiar python plotting tools as an alternative plotting method to using the `torchsig` `Visualizer` as shown below.

if include_snr:
    # For plotting, omit the SNR values
    class DataWrapper(SignalDataset):
        def __init__(self, dataset):
            self.dataset = dataset
            super().__init__(dataset)

        def __getitem__(self, idx):
            x, (y, z) = self.dataset[idx]
            return x, y

        def __len__(self) -> int:
            return len(self.dataset)

    plot_dataset = DataWrapper(dataset)
else:
    plot_dataset = dataset

data_loader = DataLoader(dataset=plot_dataset, batch_size=16, shuffle=True)

visualizer = IQVisualizer(
    data_loader=data_loader,
    visualize_transform=None,
)

for figure in iter(visualizer):
    figure.set_size_inches(14, 9)
    plt.savefig("examples/figures/01_iq_data.png")
    break


visualizer = SpectrogramVisualizer(
    data_loader=data_loader,
    nfft=1024,
    visualize_transform=None,
)

for figure in iter(visualizer):
    figure.set_size_inches(14, 9)
    plt.savefig("examples/figures/01_spectrogram.png")
    break


# ----
# ### Save Data to LMDB
# As a final exercise for this example notebook, the dataset can be saved to an LMDB static dataset for offline use. Note this is similar to how the static Sig53 dataset is generated and saved to serve as a static performance evaluation dataset.

env = lmdb.open("examples/dataset", max_dbs=3 if include_snr else 2, map_size=int(1e12))

iq_sample_db = env.open_db("iq_samples".encode())
modulations_db = env.open_db("modulations".encode())
if include_snr:
    snrs_db = env.open_db("snrs".encode())


class_dict = dict(zip(classes, range(len(classes))))

for i in tqdm(range(len(dataset))):
    # Retrieve sample
    if include_snr:
        data, (modulation, snr) = dataset[i]
    else:
        data, modulation = dataset[i]

    # Convert data from complex128 to complex64
    data_c64 = data.astype(np.complex64)

    # Save IQ data to database
    with env.begin(write=True, db=iq_sample_db) as txn:
        txn.put(str(i).encode(), pickle.dumps(data_c64))
    # Save modulation to database as class index
    with env.begin(write=True, db=modulations_db) as txn:
        txn.put(str(i).encode(), str(class_dict[modulation]).encode())
    if include_snr:
        # Save SNRs to database
        with env.begin(write=True, db=snrs_db) as txn:
            txn.put(str(i).encode(), str(snr).encode())
