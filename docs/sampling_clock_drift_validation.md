# Sampling-clock drift validation

The diagnostic demo generates both a shared-offset comparison and an isolated
model comparison. The shared comparison retains a 10,000 ppm initial
sampling-period offset for all models. The isolated comparison uses:

| Model | Initial offset | Model parameter |
|---|---:|---:|
| Constant | 1,000 ppm | none |
| Linear | 0 ppm | 10,000 ppm/second |
| Random walk | 0 ppm | 1,000 ppm/sqrt(second) |

All runs disable jitter and use fixed random seeds. The IQ samples and timing
plots use the same accepted sampling positions, and comparisons with the ideal
clock are aligned through `nominal_indices`.

## Mathematical behavior

For nominal polyphase increment `D`, output index `n`, initial fractional
period error `delta`, linear rate `beta`, sample rate `fs`, and random-walk
innovation standard deviation `q/sqrt(fs)`, the implementation produces:

```text
constant:    position_error[n] = D * n * delta
linear:      position_error[n] = D * (n * delta + beta/fs * n*(n-1)/2)
random walk: period_error[n+1] = period_error[n] + eta[n]
```

Thus constant timing error is linear, zero-offset linear drift is quadratic,
random-walk period error has variance `n*q^2/fs`, and its integrated position
error has variance
`D^2*q^2/fs * (n-1)*n*(2*n-1)/6` in polyphase-position units.

The ensemble tests use 1,200 independent seeded realizations. Their tolerances
are derived from 4.5 standard errors for a Gaussian mean and variance rather
than from one favorable trajectory. They verify innovation mean and variance,
period-error variance, integrated-position variance, fixed-seed
reproducibility, signed constant drift, quadratic linear drift, and invariance
of physical-time statistics across sample rates at fixed observation time.

## Known implementation limitations

No DSP changes were made during diagnostic cleanup. Inspection found two edge
cases that should be addressed separately:

1. Linear and random-walk output counts are based on the ideal increment. A
   clock trajectory that advances more slowly than ideal can therefore still
   have valid positions after the precomputed nominal horizon, but those
   positions are never generated. Positive drift is safely filtered at the
   upper boundary; negative or sufficiently downward-wandering drift may omit
   otherwise valid late outputs.
2. Only the initial period scale is checked for positivity. A sufficiently
   negative linear rate or a large random-walk excursion can later produce
   `1 + period_error <= 0`, causing stationary or backward sampling positions.
   The current code may return these in-range but physically invalid paths.

A future DSP change should define an output-horizon policy for time-varying
clocks and reject or terminate trajectories at their first nonpositive period
increment. Those semantics should be approved independently because they can
change output lengths and seeded results.
