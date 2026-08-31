Contributing Transforms and Signal Builders
===========================================

This guide explains how to add a signal transform or a synthetic signal builder
to TorchSig. Read :doc:`transforms` and :doc:`signals` first for the public API,
and follow the general project process in ``CONTRIBUTING.md``.

Start with an issue that describes the signal-processing behavior, expected
metadata changes, supported inputs, and how correctness will be measured. Keep
the implementation focused: a transform contribution should not also redesign
the signal-builder API, and a new builder should not refactor unrelated
modulators.

Development setup
-----------------

Install TorchSig and its development dependencies from the repository root::

    python -m pip install -e ".[dev]"

Run the narrow tests for the files being changed before running broader checks.
The most useful project commands are::

    python -m pytest tests/transforms
    python -m pytest tests/signals
    python -m ruff check torchsig
    python -m ruff format --check torchsig
    make docs

Transforms
----------

Transforms operate on :class:`torchsig.signals.signal_types.Signal` objects.
The implementation is normally split between:

* ``torchsig/transforms/functional.py``, which contains array-level signal
  processing; and
* ``torchsig/transforms/transforms.py``, which contains the stateful
  :class:`torchsig.transforms.transforms.SignalTransform` wrapper.

Keeping numerical work in a functional makes it independently testable and
reusable. The wrapper owns parameter sampling, validation, metadata updates,
dtype enforcement, and reproducible random state.

Implement the functional
~~~~~~~~~~~~~~~~~~~~~~~~

A functional should accept explicit values and return an array. If it uses
randomness, accept an optional ``numpy.random.Generator`` instead of creating or
seeding global random state. Validate invalid values with clear exceptions and
preserve the expected shape and NumPy dtype.

For example::

    def scale_amplitude(data: np.ndarray, gain: float) -> np.ndarray:
        """Scale complex IQ samples by a linear gain."""
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("gain must be finite and nonnegative")
        return (data * gain).astype(data.dtype, copy=False)

Add the functional name to the module's ``__all__`` list if it is public. Tests
belong in ``tests/transforms/test_functional.py`` or a focused neighboring test
module. Cover normal input, boundary values, invalid values, shape, dtype, and
determinism when randomness is involved.

Implement the wrapper
~~~~~~~~~~~~~~~~~~~~~

Most new wrappers inherit from
:class:`torchsig.transforms.transforms.SignalTransform`. Initialize the base
class with required metadata and the output dtype, store constructor arguments,
and implement ``__apply__``. ``SignalTransform.__call__`` performs validation,
calls ``__apply__``, updates bookkeeping, and enforces ``data_dtype``.

For example::

    class ScaleAmplitude(SignalTransform):
        """Scale a signal by a fixed linear gain."""

        def __init__(self, gain: float, **kwargs) -> None:
            super().__init__(data_dtype=TorchSigComplexDataType, **kwargs)
            self.gain = gain

        def __apply__(self, signal: Signal) -> Signal:
            signal.data = F.scale_amplitude(signal.data, self.gain)
            return signal

Use ``required_metadata`` only for fields the transform truly needs. Update
metadata in ``__apply__`` when the operation changes values such as bandwidth,
center frequency, duration, or SNR. If an accurate update is expensive, follow
the nearby ``precise`` pattern and document the tradeoff.

Random transforms
~~~~~~~~~~~~~~~~~

Transforms inherit TorchSig's :class:`torchsig.utils.random.Seedable` behavior.
Use ``self.random_generator`` for direct random draws. For configurable values,
``self.get_distribution(value_or_range)`` creates a child distribution that
shares the transform's seeded random state. Do not call ``numpy.random.seed`` or
use NumPy's module-level random functions.

The accepted distribution conventions are:

* a list selects uniformly from its entries;
* a tuple samples uniformly between its two bounds;
* a scalar samples uniformly from zero to that value; and
* ``scaling="log10"`` applies logarithmic sampling to a tuple.

Register and document a transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a public transform:

#. Add it to the appropriate ``__all__`` list in
   ``torchsig/transforms/transforms.py`` or ``functional.py``.
#. Add focused wrapper tests under ``tests/transforms``.
#. Add or update an example under ``examples/transforms`` when visual behavior
   or usage is not obvious from the API documentation.
#. Use Google-style docstrings with arguments, return values, raised exceptions,
   units, valid ranges, and the direction/sign convention for physical values.

Signal builders
---------------

A signal builder combines one or more pure modulation functions with a subclass
of :class:`torchsig.signals.builder.BaseSignalGenerator`. Calling the generator
runs ``generate()``, attaches the generator as a transient metadata parent,
sets its class name, and applies configured transforms.

Implement modulation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a builder in a focused module under ``torchsig/signals/builders``. Keep
the waveform math in module-level functions so it can be tested without dataset
construction. Functions should:

* accept explicit sample counts, rates, bandwidths, and an optional RNG;
* validate nonpositive counts and impossible rate/bandwidth combinations;
* return a one-dimensional NumPy array of the requested length;
* return ``TorchSigComplexDataType`` unless the neighboring API requires a
  different dtype; and
* avoid global random state.

When resampling or filtering, test output length, finite values, occupied
bandwidth, and representative spectral or temporal properties. Test small and
boundary-sized inputs as well as normal inputs.

Implement the generator
~~~~~~~~~~~~~~~~~~~~~~~

Declare required metadata in ``required_metadata_fields``, choose a stable class
name with ``set_default_class_name``, and implement ``generate()``. The returned
:class:`torchsig.signals.signal_types.Signal` must contain waveform data and
accurate metadata.

A minimal generator resembles::

    class ExampleSignalGenerator(BaseSignalGenerator):
        """Generate an example waveform."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.required_metadata_fields = [
                "signal_duration_in_samples_min",
                "signal_duration_in_samples_max",
            ]
            self.set_default_class_name("example")

        def generate(self) -> Signal:
            num_samples = self.random_generator.integers(
                self["signal_duration_in_samples_min"],
                self["signal_duration_in_samples_max"] + 1,
            )
            data = example_modulator(num_samples, rng=self.random_generator)
            return Signal(data=data, center_freq=0, bandwidth=1)

Metadata may be inherited from a parent dataset, so access it through
``self["field_name"]`` rather than assuming it is stored directly on the
generator. ``validate_metadata_fields()`` checks that every required key is a
string and is available locally or through the metadata hierarchy.

Register a builder
~~~~~~~~~~~~~~~~~~

If a builder must be selectable by name in
:class:`torchsig.datasets.datasets.TorchSigIterableDataset`, update all relevant
registration locations:

#. Import the generator in ``torchsig/utils/signal_building.py`` and add each
   concrete class name to ``signal_generator_lookup_table`` with
   ``_add_signal_generator``.
#. Update ``_family_name`` and ``family_names`` when introducing a new family.
#. Add each public class name and family mapping to
   ``CLASS_FAMILY_DICT`` in ``torchsig/signals/signal_lists.py``.
#. Extend :class:`torchsig.signals.signal_lists.TorchSigSignalLists` if the new
   family needs its own grouped list.
#. Export public functions and classes from the builder module's ``__all__``.

The class name emitted by the generator must exactly match the lookup-table and
signal-list spelling. Add a lookup test so string-based dataset configuration is
verified, not just direct class construction.

Test a builder
~~~~~~~~~~~~~~

Follow the nearest module under ``tests/signals/builders``. A complete test set
normally checks:

* invalid modulator inputs;
* exact output shape and complex dtype;
* finite samples and meaningful waveform properties;
* deterministic output for equal seeds;
* variation for different seeds where appropriate;
* required metadata fields and default class name;
* inclusive sampling of configured minimum/maximum values;
* correct ``Signal`` metadata returned by ``generate()``;
* configured transforms applied by ``BaseSignalGenerator.__call__``; and
* string lookup and family registration for public dataset classes.

Use small deterministic arrays in unit tests. Spectral assertions should allow
only the tolerance required by the algorithm; do not weaken an assertion merely
to make a random test pass.

Review checklist
----------------

Before opening a merge request, confirm the following:

* Public APIs have type annotations and Google-style docstrings.
* Units, ranges, defaults, sign conventions, and exceptions are documented.
* Complex-valued NumPy and PyTorch dtypes are preserved intentionally.
* All randomness comes from the provided or inherited generator.
* Metadata remains accurate after transforms and signal generation.
* Public names are exported and builders are registered everywhere required.
* Normal, boundary, invalid-input, dtype, and deterministic cases are tested.
* The narrow test modules pass before broader transform/signal tests.
* ``python -m ruff check torchsig`` and ``make docs`` pass.
* No generated documentation, caches, benchmark results, or notebook outputs
  are committed.
