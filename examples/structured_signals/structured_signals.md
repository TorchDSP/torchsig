# Structured Signals

TorchSig's original signal builders (FSK, PSK/QAM, OFDM, AM/FM, chirp, …) emit
streams of **random symbols**. *Structured* signals instead model an explicit
protocol frame layout — fixed sync words, training preambles, signaling
headers, and pilots — so the generated IQ has the recognizable temporal/spectral
structure of a real waveform.

All provided structured signal builders are **structural models**: the framing, preambles,
modulation, pulse shaping, and per-frame symbol counts are fairly faithful, but the
forward error correction / scrambling / bit-exact field encoding is intentionally
skipped (payload symbols are random). This is ideal for RF Machine Learning (RFML)
datasets, where structure matters more than bit-level decodability.

Note: unlike the more generic modulation signal classes, structured signals often inherently 
break contracts for specified parameter ranges such as burst lengths, bandwidths, etc. Be sure
to check the signal builder implementation and output for your application.

| Family | Class name(s) | Builder |
| ------ | ------------- | ------- |
| `adsb` | `adsb-long`, `adsb-short` | `torchsig/signals/builders/adsb.py` |
| `bluetooth` | `btle` | `torchsig/signals/builders/btle.py` |
| `cellular` | `gsm` | `torchsig/signals/builders/cellular.py` |
| `dvb`   | `dvbs2` | `torchsig/signals/builders/dvb.py` |
| `lmr`   | `dmr`, `p25` | `torchsig/signals/builders/lmr.py` |
| `lora`   | `lora` | `torchsig/signals/builders/lora.py` |
| `wifi`  | `80211a`, `80211a_rts`, `80211a_cts`, `80211a_ack` | `torchsig/signals/builders/wifi.py` |
| `zigbee` | `zigbee` | `torchsig/signals/builders/zigbee.py` |

These plug into the standard pipeline like any other class:

```python
from torchsig.datasets.datasets import TorchSigIterableDataset

dataset = TorchSigIterableDataset(
    signal_generators=["dmr", "80211a", "80211a_ack", "dvbs2"],
    metadata=my_metadata,            # needs sample_rate, bandwidth_min/max,
    target_labels=["class_name"],    # signal_duration_in_samples_min/max, ...
)
data, label = next(iter(dataset))
```

See `examples/structured_signals/structured_signals_example.py` for a runnable script that builds a
small dataset and renders the spectrogram montage
(`examples/structured_signals_spectrograms.png`).

