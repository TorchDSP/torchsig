# Example 00 - The Official Sig53 Dataset
# This notebook walks through an example of how the official Sig53 dataset can be instantiated and analyzed.

# ----
# ### Import Libraries
# First, import all the necessary public libraries as well as a few classes from the `torchsig` toolkit.

from torchsig.utils.writer import DatasetLoader, DatasetCreator, LMDBDatasetWriter
from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets.sig53 import Sig53
from torchsig.utils.dataset import SignalDataset
from torchsig.datasets import conf
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os

# ----
# ### Instantiate Sig53 Dataset
# To instantiate the Sig53 dataset, several parameters are given to the imported `Sig53` class. These paramters are:
# - `root` ~ A string to specify the root directory of where to instantiate and/or read an existing Sig53 dataset
# - `train` ~ A boolean to specify if the Sig53 dataset should be the training (True) or validation (False) sets
# - `impaired` ~ A boolean to specify if the Sig53 dataset should be the clean version or the impaired version
# - `transform` ~ Optionally, pass in any data transforms here if the dataset will be used in an ML training pipeline
# - `target_transform` ~ Optionally, pass in any target transforms here if the dataset will be used in an ML training pipeline
#
# A combination of the `train` and the `impaired` booleans determines which of the four (4) distinct Sig53 datasets will be instantiated:
# - `train=True` & `impaired=False` = Clean training set of 1.06M examples
# - `train=True` & `impaired=True` = Impaired training set of 5.3M examples
# - `train=False` & `impaired=False` = Clean validation set of 106k examples
# - `train=False` & `impaired=True` = Impaired validation set of 106k examples
#
# The final option of the impaired validation set is the dataset to be used when reporting any results with the official Sig53 dataset.
#
# Additional optional parameters of potential interest are:
# - `regenerate` ~ A boolean specifying if the dataset should be regenerated even if an existing dataset is detected (Default: False)
# - `eb_no` ~ A boolean specifying if the SNR should be defined as Eb/No if True (making higher order modulations more powerful) or as Es/No if False (Defualt: False)
# - `use_signal_data` ~ A boolean specifying if the data and target information should be converted to `SignalData` objects as they are read in (Default: False)

# Specify script options
figure_dir = "examples/figures"
if not os.path.isdir(figure_dir):
    os.mkdir(figure_dir)

cfg = conf.Sig53CleanTrainConfig

ds = ModulationsDataset(
    level=cfg.level,
    num_samples=1060,
    num_iq_samples=cfg.num_iq_samples,
    use_class_idx=cfg.use_class_idx,
    include_snr=cfg.include_snr,
    eb_no=cfg.eb_no,
)

loader = DatasetLoader(ds, seed=12345678)
writer = LMDBDatasetWriter(path="examples/sig53/sig53_clean_train")
creator = DatasetCreator(loader, writer)
creator.create()
sig53 = Sig53("examples/sig53", train=True, impaired=False)

# Retrieve a sample and print out information
idx = np.random.randint(len(sig53))
data, (label, snr) = sig53[idx]
print("Dataset length: {}".format(len(sig53)))
print("Data shape: {}".format(data.shape))
print("Label Index: {}".format(label))
print("Label Class: {}".format(Sig53.convert_idx_to_name(label)))
print("SNR: {}".format(snr))


# ----
# ### Plot Subset to Verify
# The `IQVisualizer` and the `SpectrogramVisualizer` can be passed a `Dataloader` and plot visualizations of the dataset. The `batch_size` of the `DataLoader` determines how many examples to plot for each iteration over the visualizer. Note that the dataset itself can be indexed and plotted sequentially using any familiar python plotting tools as an alternative plotting method to using the `torchsig` `Visualizer` as shown below.


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


plot_dataset = DataWrapper(sig53)

data_loader = DataLoader(dataset=plot_dataset, batch_size=16, shuffle=True)

# Transform the plotting titles from the class index to the name
def target_idx_to_name(tensor: np.ndarray) -> list:
    batch_size = tensor.shape[0]
    label = []
    for idx in range(batch_size):
        label.append(Sig53.convert_idx_to_name(int(tensor[idx])))
    return label


visualizer = IQVisualizer(
    data_loader=data_loader,
    visualize_transform=None,
    visualize_target_transform=target_idx_to_name,
)

for figure in iter(visualizer):
    figure.set_size_inches(14, 9)
    plt.savefig("examples/figures/00_iq_data.png")
    break


# Repeat but plot the spectrograms for a new random sampling of the data
visualizer = SpectrogramVisualizer(
    data_loader=data_loader,
    nfft=1024,
    visualize_transform=None,
    visualize_target_transform=target_idx_to_name,
)

for figure in iter(visualizer):
    figure.set_size_inches(14, 9)
    plt.savefig("examples/figures/00_spectrogram.png")
    break


# ----
# ### Analyze Dataset
# The dataset can also be analyzed at the macro level for details such as the distribution of classes and SNR values. This exercise is performed below to show the nearly uniform distribution across each.

# Loop through the dataset recording classes and SNRs
class_counter_dict = {
    class_name: 0 for class_name in list(Sig53._idx_to_name_dict.values())
}
all_snrs = []

for idx in tqdm(range(len(sig53))):
    data, (modulation, snr) = sig53[idx]
    class_counter_dict[Sig53.convert_idx_to_name(modulation)] += 1
    all_snrs.append(snr)


# Plot the distribution of classes
class_names = list(class_counter_dict.keys())
num_classes = list(class_counter_dict.values())

plt.figure(figsize=(9, 9))
plt.pie(num_classes, labels=class_names)
plt.title("Class Distribution Pie Chart")
plt.savefig("examples/figures/00_class_distribution_pie.png")

plt.figure(figsize=(11, 4))
plt.bar(class_names, num_classes)
plt.xticks(rotation=90)
plt.title("Class Distribution Bar Chart")
plt.xlabel("Modulation Class Name")
plt.ylabel("Counts")
plt.savefig("examples/figures/00_class_distribution_bar.png")


# Plot the distribution of SNR values
plt.figure(figsize=(11, 4))
plt.hist(x=all_snrs, bins=100)
plt.title("SNR Distribution")
plt.xlabel("SNR Bins (dB)")
plt.ylabel("Counts")
plt.savefig("examples/figures/00_snr_distribution_hist.png")
