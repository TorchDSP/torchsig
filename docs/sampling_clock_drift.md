# Sampling-clock drift models

TorchSIG sampling-clock impairments support constant, linear, and random-walk
sampling-period error. The default is `constant`.

All ppm values describe **sampling-period error**, not sampling-frequency
error. A positive value increases the input-position increment, corresponding
to a slower sampling clock; a negative value decreases it. For small errors,
sampling-frequency error has approximately the opposite sign.

```python
from torchsig.transforms.transforms import ClockDrift

transform = ClockDrift(
    drift_model="random_walk",
    drift_ppm=(1.0, 10.0),
    drift_noise_ppm_per_sqrt_sec=(0.1, 1.0),
    sample_rate_hz=1_000_000.0,
)
```

- `drift_ppm` is the signed initial period offset and applies to every model.
- `drift_rate_ppm_per_sec` is used by `linear`. At output index `n`, its
  discrete period error is `drift_ppm + n * drift_rate_ppm_per_sec / sample_rate_hz`.
- `drift_noise_ppm_per_sqrt_sec` is used by `random_walk`. Each period-error
  innovation has standard deviation
  `drift_noise_ppm_per_sqrt_sec / sqrt(sample_rate_hz)` ppm.

The public functional and DSP interfaces accept the same four drift settings.
Jitter models remain independent and may be enabled at the same time.

## Compatibility

Previously, `drift_ppm` controlled both a constant period offset and an
undocumented random-walk strength. It now controls only the deterministic
initial offset. Select `drift_model="random_walk"` and set
`drift_noise_ppm_per_sqrt_sec` to request stochastic drift. Consequently,
seeded outputs using nonzero `drift_ppm` change, while constant drift no longer
consumes random-number draws.
