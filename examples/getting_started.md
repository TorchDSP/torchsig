# TorchSig: Getting Started

This file gives an overview of the terms and capabilities of TorchSig. There is no code to run.

---

## TorchSig Signals
A signal has many definitions in the digital signal processing, radio frequency, and machine learning world. Below we define precisely what we define as certain signals.

```python
from torchsig.signals.signal_types import SignalMetadataObject, Signal
```

* `SignalMetadataObject`: Object that contains metadata for a Signal.
* `Signal`: A subclass of `SignalMetadataObject` that represents a signal. Contains IQ data, metadata, and optionally a list component `Signals`. The `BaseSignalGenerator` are responsible for creating a Signal object.

---

## TorchSig Transforms

```python
from torchsig.transforms.transforms import ...
from torchsig.transforms.functional import ...
from torchsig.transforms.metadata_transforms import ...
```

### Transforms
Provide callable transform operations on TorchSig data.
* Used with Signal object to modify data
* Broad, general application range that include signal processing, communications channel effects, and dataset manipulation
* Often configure and call TorchSig Functionals for lower-level operations by controlling how data is modified and defining parameter distributions

Transforms work at several levels of TorchSig data organization. When applied to Signal objects they transform these isolated signal bursts individually, often to model transmitter effects on signal I/Q data. When applied to Signal objects they transform the aggregate data within these samples, which were constructed with an arbitrary number of Signals. These sample-scale modifications often implement wideband channel or receiver effects, or perform dataset manipulations.

### Functionals
Core computational processes and algorithm implementations for transforming data.
* Functions that perform fundamental data operations, but do not specify how input parameters are distributed or selected
* Similiar to the organization of [Torchvision's transforms framework](https://pytorch.org/vision/0.9/transforms.html#functional-transforms)

### Dataset Impairments
Special collections of Transform sequences that emulate different types of channel environment effects.
* Level 0: Perfect environment, such as inside a computer simulation. Has no transforms.
* Level 1: Cabled environment, such as a benchtop experiment. Contains some transforms that moderately impair the signals.
* Level 2: Wireless environment. Contains many transforms that impair the signals greatly, such as models of radio frequency hardware and wireless channel effects.

Most of the provided default TorchSig dataset classes allow the user to specify the baseline impairment level, as well as specify any additional desired transforms in sequence. Note that Impairments are carefully applied within the SignalBuilders in a strict order to represent actual signal processing effects.

### Metadata Transforms
Similar to Transforms define above, but these transforms do not alter signal data, and only alters signal metadata. 
* The metadata transforms enable users to calculate custoom labels, targets, or other fields they would like from the dataset. 
* Metadata transforms interface with `Signal` objects.

In older versions of TorchSig, this would be target transforms. 

---

## TorchSig Datasets

```python
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.file_handlers import FileReader
```

In I/Q datasets the data is structured in arrays of signal I/Q samples with supporting metadata, as described in the signals section above. These datasets are usually synthetic I/Q data and metadata generated parametrically with a TorchSig workflow, but may also be externally sourced data imported into the TorchSig dataset framework.

These are some major object types for generating or working with a dataset.
* Dataset metadata: Metadata information necessary for generating or working with the dataset. Represented with a Python dictionary.
  * A default dataset metadata dictionary can be found in `TorchSigDefaults().default_dataset_metadata`
* `TorchSigIterableDataset`: Dataset that can infinitely generate signals in memory or optionally write finite datasets to disk with DatasetCreator. Is an instance of PyTorch's [IterableDataset](https://docs.pytorch.org/docs/stable/data.html#iterable-style-datasets).
* `StaticTorchSigDataset`: Dataset that reads in a pre-generated TorchSig dataset from disk. This assumes you have created a dataset using `DatasetCreator`.
  * Can also use this class to load in external data, provided the user has written a custom `FileReader` class,
* `DatasetCreator`: Takes in a PyTorch [DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), and writes the dataset to disk. The DataLoader should contain a `TorchSigIterableDataset`.

Refer to the provided example notebooks for illustrative workflows on generating datasets, saving and loading datasets, importing external data, and working with TorchSig I/Q datasets.



### Dataset Feature Selection Table

| I want to ... | Use |
| --------- | ----- |
| generate an infinite dataset | `TorchSigIterableDataset` |
| write a dataset to disk | `TorchSigIterableDataset`, `DatasetCreator` |
| load a dataset from disk | `StaticDataset` |
| import external data from disk as a static TorchSig dataset | `StaticDataset `(provide a file handler) |
| generate a finite dataset, where I can call previously generated TorchSig samples | `TorchSigIterableDataset`, `DatasetCreator`, `StaticDataset` |

---

This is the end of the file. If you are unsure where to start, see `examples/README.md` to view the list of notebooks or check out `create_dataset_example.ipynb`.