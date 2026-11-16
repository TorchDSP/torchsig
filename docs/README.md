# TorchSig Documentation
---

From the repository root, install the documentation profile:

```
pip install -e ".[docs]"
```

Then build the HTML documentation:
```
make docs
```

Navigate to `build/html/index.html` for the generated HTML documenation.

## Contribution guides

- [Transforms and Signal Builders](contributing_transforms_and_signal_builders.rst)
Navigate to `docs/build/html/index.html` for the generated HTML documentation.

## Signal-generation configuration

Dataset YAML files may include an optional `signals` mapping. Unspecified
classes and parameters retain their existing randomized behavior. The schema
supports fixed pulse-shaping rolloff values for constellation signals and
DVB-S2.

```yaml
dataset_id: fixed_qpsk_alpha
dataset_length: 1000
seed: 123
impairment_level: 0
output:
  representation: iq
signal_sampling:
  mode: per_signal
dataset_metadata:
  sample_rate: 1000000
  fft_size: 256
  class_list: [qpsk]
signals:
  qpsk:
    parameters:
      alpha_rolloff:
        value: 0.35
```

`alpha_rolloff` must be a real number strictly between 0 and 1. A configured
value selects SRRC pulse shaping for constellation signals and is recorded under
`experiment_config` in the generated dataset's `dataset_info.yaml`. Unknown
classes, parameters, schema keys, and invalid values are rejected while the
configuration is loaded, before dataset generation begins.

The former QPSK-only name `alpha` remains accepted as a deprecated input alias.
It is normalized to `alpha_rolloff`; new configurations and serialized dataset
metadata use only the canonical name.

The same schema can be supplied directly to `TorchSigIterableDataset` using
its `experiment_config` argument, either as a mapping or as the path to a YAML
file containing the `signals` mapping.

For typed, programmatic configuration, construct the same effective schema
without YAML:

```python
from torchsig.utils.experiment_config import (
    ExperimentConfig,
    FixedValue,
    SignalConfig,
)

experiment_config = ExperimentConfig(
    signals={
        "qpsk": SignalConfig(
            parameters={"alpha_rolloff": FixedValue(0.35)},
        ),
    },
)

dataset = TorchSigIterableDataset(
    signal_generators=["qpsk"],
    metadata=dataset_metadata,
    experiment_config=experiment_config,
)
```

Both forms use identical validation and serialize to the same effective
configuration in `dataset_info.yaml`.
