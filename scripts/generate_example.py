from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.transforms import Spectrogram

import matplotlib.pyplot as plt


# define dataset metadata, can override defaults
dataset_metadata = TorchSigDefaults().default_dataset_metadata

# optionally, apply impairments
impairments = Impairments(level=0)
burst_impairments = impairments.signal_transforms
whole_signal_impairments = impairments.dataset_transforms

# create the dataset
dataset = TorchSigIterableDataset(
    metadata=dataset_metadata,
    transforms=[whole_signal_impairments, Spectrogram(fft_size=dataset_metadata["fft_size"])],
    component_transforms=[burst_impairments],
)
# create a dataloader (reproducible)
dataloader = WorkerSeedingDataLoader(dataset, batch_size=2)

# save the dataset to disk
dataset_creator = DatasetCreator(
    dataset_length=20,
    dataloader=dataloader,
    root="./sample_dataset",
    overwrite=True,
    multithreading=False,
)
dataset_creator.create()

# load the dataset in from disk
static_dataset = StaticTorchSigDataset(
    root="./sample_dataset",
    target_labels=[] # empty list -> dataset returns Signal.data only
)

# save or show the first 4 items of the dataset
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

axes[0, 0].imshow(static_dataset[0], cmap="Wistia", vmin=0)
axes[0, 0].set_title("Index 0")
axes[0, 1].imshow(static_dataset[1], cmap="Wistia", vmin=0)
axes[0, 1].set_title("Index 1")
axes[1, 0].imshow(static_dataset[2], cmap="Wistia", vmin=0)
axes[1, 0].set_title("Index 2")
axes[1, 1].imshow(static_dataset[3], cmap="Wistia", vmin=0)
axes[1, 1].set_title("Index 3")

fig.tight_layout()

fig.show()
fig.savefig("generate_example.png")
print("First 4 data saved to generate_example.png")
