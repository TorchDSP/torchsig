# Default Configs for Narrowband and Wideband

## Narrowband
num_iq_samples = 262144 (512^2)
(train) num_samples = 5.7M (57 classes x 100k samples per class)
(val) num_samples = 57k (57 classes x 1k samples per class)
num_signals_min = 1 (every sample will have a signal)

## Wideband
num_iq_samples = 1048576 (1024^2)
(train) num_samples = 57k (57 classes x 1k samples per class)
(val) num_samples = 5.7k (57 classes x 100 samples per class)
num_signals_min = 3
num_signals_max = 5


## Notes
* We have dropped PAM signal types in our default list, due to rarity seen in environment.
