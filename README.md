<p align="center">
  <img src="docs/logo.png" alt="drawing" width="500"/>
</p>

-----


TorchSig is an open-source signal processing machine learning toolkit based on the PyTorch data handling pipeline. The user-friendly toolkit simplifies common digital signals processing operations, augmentations, and transformations when dealing with both real and complex-valued signals. TorchSig streamlines the integration process of these signals processing tools building on PyTorch, enabling faster and easier development and research for machine learning techniques applied to signals data, particularly within (but not limited to) the radio frequency domain. An example dataset based on many unique communication signal modulations is also included to accelerate the field of modulation recognition.

*TorchSig is currently in beta (v0.1.0)*

## Key Features
---
TorchSig provides many useful tools to facilitate and accelerate research on signals processing machine learning technologies:
- The `SignalData` class and its `SignalDescription` objects enable signals objects and meta data to be seamlessly handled and operated on throughout the TorchSig infrastructure.
- The `Sig53` Dataset is a state-of-the-art static modulations-based RF dataset meant to serve as the next baseline for RFML development & evaluation.
- The `ModulationsDataset` class synthetically creates, augments, and transforms the largest communications signals modulations dataset to date in a generic, flexible fashion.
- Numerous signals processing transforms enable existing ML techniques to be employed on the signals data, streamline domain-specific signals augmentations in signals processing machine learning experiments, and signals-specific data transformations to speed up the field of expert feature signals processing machine learning integration.


## Documentation
---
Documentation can be built locally by following the instructions below.
```
cd docs
pip install -r docs-requirements.txt
make html
firefox build/html/index.html
```

## Installation
---
Clone the `torchsig` repository and simply install using the following commands:
```
cd torchsig
pip install -r requirements.txt
pip install .
```

## License
---
TorchSig is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. TorchSig has no connection to MIT, other than through the use of this license.


## Citing TorchSig
---
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