<a align="center" href="https://torchsig.com">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/torchsig_logo_white_dodgerblue.png">
        <img src="docs/logo.png" width="500">
    </picture>
</a>

-----

[TorchSig](https://torchsig.com) is an open-source signal processing machine learning toolkit based on the PyTorch data handling pipeline. The user-friendly toolkit simplifies common digital signal processing operations, augmentations, and transformations when dealing with both real and complex-valued signals. TorchSig streamlines the integration process of these signals processing tools building on PyTorch, enabling faster and easier development and research for machine learning techniques applied to signals data, particularly within (but not limited to) the radio frequency domain.

# Getting Started

## Prerequisites
- Ubuntu &ge; 22.04
- Hard drive storage with &ge; 1 TB
- CPU with &ge; 4 cores
- GPU with &ge; 16 GB storage (recommended)
- Python &ge; 3.10

We highly reccomend Ubuntu or using a Docker container.

## Installation
Clone the `torchsig` repository and install using the following commands:
```
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install -e .
```
 
# Examples and Tutorials

TorchSig has a series of Jupyter notebooks in the `examples/` directory. View the README inside `examples/` to learn more.

# Usage

## Generating Datasets with Python
TorchSig uses a unified dataset architecture. Create datasets using the Python API:
```python
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
)

print(static_dataset[0])
```

# Docker
One option for running TorchSig is within Docker. Start by building the Docker container:

```bash
docker build -t torchsig -f docker/Dockerfile .
```

And then you can launch a Docker instance:
```bash
docker run -it torchsig
```
See `docker/README.md` to learn more.

# Development
To contribute to our library, please make sure to run the following:

```bash
# pytests all pass
pytest

# pylint score > 9/10
pylint --rcfile=.pylintrc torchsig

# not required
# but helpful for maintaining PEP 8 Style Guide
ruff check torchsig
```
Both need to pass in order to contribute to our Github.

# Key Features
TorchSig provides many useful tools to facilitate and accelerate research on signals processing machine learning technologies:
- **Unified Dataset Architecture**: TorchSig features a single, flexible dataset system that supports both signal classification (single signal) and signal detection (multiple signals) tasks through configuration.
- **Comprehensive Signal Library**: Support for 60+ signal types across all major modulation families (FSK, QAM, PSK, ASK, OFDM, Analog) with realistic impairments and channel effects.
- **Advanced Transform System**: Numerous signals processing transforms enable existing ML techniques to be employed on signals data, with unified impairment models supporting perfect, cabled, and wireless channel conditions.

## Core Classes
- **`Signal` and `SignalMetadataObject`**: Enable signal objects and metadata to be seamlessly handled and operated on throughout the TorchSig infrastructure.
- **`TorchSigIterableDataset`**: Unified dataset class that synthetically creates, augments, and transforms signals datasets. Behavior (classification vs detection) is determined by configuration parameters.
  - Can generate samples infinitely when `num_samples=None`, or finite datasets when `num_samples` is specified.
  - Dataset type determined by `num_signals_max`: 1 for classification, >1 for detection tasks.
- **`DatasetCreator`**: Writes a PyTorch `DataLoader` containing a `TorchSigIterableDataset` objects to disk with progress tracking and memory optimization.
- **`StaticTorchSigDataset`**: Loads previously generated datasets from disk back into memory.
  - Can access previously generated samples efficiently.
  - Supports both classification and detection datasets through unified interface.



# Documentation
Documentation can be found [online](https://torchsig.readthedocs.io/en/latest/) or built locally by following the instructions below.
```
cd docs
pip install -r docs-requirements.txt
make html
firefox build/html/index.html
```


# License
TorchSig is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. TorchSig has no connection to MIT, other than through the use of this license.

# Publications
| Title | Year  | Cite (APA) |
| ----- | ----  | ---------- |
| [TorchSig 2.0: Dataset Customization, New Transforms and Future Plans](https://events.gnuradio.org/event/26/contributions/752/) | 2025 | Oh, E., Mullins, J., Carrick, M., Vondal, M., Hoffman, J., Leonardo, F., Toliver, P., Miller, R. (2025, September). TorchSig 2.0: Dataset Customization, New Transforms and Future Plans. In Proceedings of the GNU Radio Conference (Vol. 10, No. 1). |
| [TorchSig: A GNU Radio Block and New Spectrogram Tools for Augmenting ML Training](https://events.gnuradio.org/event/24/contributions/628/) | 2024 | Vallance, P., Oh, E., Mullins, J., Gulati, M., Hoffman, J., & Carrick, M. (2024, September). TorchSig: A GNU Radio Block and New Spectrogram Tools for Augmenting ML Training. In Proceedings of the GNU Radio Conference (Vol. 9, No. 1). |
| [Large Scale Radio Frequency Wideband Signal Detection & Recognition](https://doi.org/10.48550/arXiv.2211.10335)| 2022 | Boegner, L., Vanhoy, G., Vallance, P., Gulati, M., Feitzinger, D., Comar, B., & Miller, R. D. (2022). Large Scale Radio Frequency Wideband Signal Detection & Recognition. arXiv preprint arXiv:2211.10335. |
| [Large Scale Radio Frequency Signal Classification](https://doi.org/10.48550/arXiv.2207.09918) | 2022 | Boegner, L., Gulati, M., Vanhoy, G., Vallance, P., Comar, B., Kokalj-Filipovic, S., ... & Miller, R. D. (2022). Large Scale Radio Frequency Signal Classification. arXiv preprint arXiv:2207.09918. |


# Citing TorchSig

Please cite TorchSig if you use it for your research or business.

```bibtext
@misc{torchsig,
  title={Large Scale Radio Frequency Signal Classification},
  author={Luke Boegner and Manbir Gulati and Garrett Vanhoy and Phillip Vallance and Bradley Comar and Silvija Kokalj-Filipovic and Craig Lennon and Robert D. Miller},
  year={2022},
  archivePrefix={arXiv},
  eprint={2207.09918},
  primaryClass={cs-LG},
  note={arXiv:2207.09918}
  url={https://arxiv.org/abs/2207.09918}
}
```
