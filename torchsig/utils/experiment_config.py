"""Validated experiment configuration for signal generation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.utils.signal_building import public_generator_names

__all__ = ["ExperimentConfig", "FixedValue", "SignalConfig", "load_experiment_config"]

_ROLLOFF_SIGNAL_CLASSES = set(all_symbol_maps).union({"dvbs2"})
_FIXED_PARAMETERS = {class_name: {"alpha_rolloff"} for class_name in _ROLLOFF_SIGNAL_CLASSES}
_LEGACY_PARAMETER_ALIASES = {"qpsk": {"alpha": "alpha_rolloff"}}


@dataclass(frozen=True)
class FixedValue:
    """A generator parameter fixed to one value for every generated signal."""

    value: Any


@dataclass(frozen=True)
class SignalConfig:
    """Generator parameter overrides for one concrete signal class."""

    parameters: Mapping[str, FixedValue]

    def __post_init__(self) -> None:
        """Validate parameters and store them in a read-only mapping."""
        if not isinstance(self.parameters, Mapping):
            raise TypeError("SignalConfig.parameters must be a mapping")

        parameters = dict(self.parameters)
        for name, value in parameters.items():
            if not isinstance(name, str):
                raise TypeError("SignalConfig parameter names must be strings")
            if not isinstance(value, FixedValue):
                raise TypeError(f"SignalConfig parameter {name!r} must be a FixedValue")

        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(parameters),
        )

    def __reduce__(self):
        """Support pickling while preserving read-only parameters."""
        return type(self), (dict(self.parameters),)


class ExperimentConfig:
    """Validated, programmatic signal-generation configuration.

    Args:
        signals: Mapping of concrete signal class names to their configuration.

    Example:
        Configure a fixed QPSK pulse-shaping rolloff::

            config = ExperimentConfig(
                signals={
                    "qpsk": SignalConfig(
                        parameters={"alpha_rolloff": FixedValue(0.35)},
                    ),
                },
            )

    The initial schema supports fixed parameter values. ``FixedValue`` leaves
    room for future range, choice, and distribution value specifications.
    """

    def __init__(self, signals: Mapping[str, SignalConfig] | None = None) -> None:
        if signals is None:
            signals = {}
        if not isinstance(signals, Mapping):
            raise TypeError("ExperimentConfig.signals must be a mapping")

        validated: dict[str, SignalConfig] = {}
        for class_name, signal_config in signals.items():
            if not isinstance(class_name, str) or class_name not in public_generator_names:
                raise ValueError(f"unknown signal class {class_name!r}")
            if not isinstance(signal_config, SignalConfig):
                raise TypeError(f"configuration for signal class {class_name!r} must be a SignalConfig")

            parameters = dict(signal_config.parameters)
            aliases = _LEGACY_PARAMETER_ALIASES.get(class_name, {})
            for alias, canonical_name in aliases.items():
                if alias not in parameters:
                    continue
                if canonical_name in parameters:
                    raise ValueError(f"parameters {alias!r} and {canonical_name!r} cannot both be configured for signal class {class_name!r}")
                warnings.warn(
                    f"signals.{class_name}.parameters.{alias} is deprecated; use {canonical_name!r}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                parameters[canonical_name] = parameters.pop(alias)

            supported = _FIXED_PARAMETERS.get(class_name, set())
            unknown_parameters = set(parameters).difference(supported)
            if unknown_parameters:
                raise ValueError(f"unknown parameters for signal class {class_name!r}: {sorted(unknown_parameters)}; supported parameters: {sorted(supported)}")
            for parameter_name, parameter in parameters.items():
                _validate_fixed_value(class_name, parameter_name, parameter.value)
            validated[class_name] = SignalConfig(parameters=parameters)

        self._signals = MappingProxyType(validated)

    def __reduce__(self):
        """Support pickling while preserving read-only signal configuration."""
        return type(self), (dict(self._signals),)

    @property
    def signals(self) -> Mapping[str, SignalConfig]:
        """Read-only mapping of configured signal classes."""
        return self._signals

    def parameter_value(self, class_name: str, parameter_name: str) -> float | None:
        """Return a fixed override, or ``None`` when it is not configured."""
        canonical_name = _LEGACY_PARAMETER_ALIASES.get(class_name, {}).get(parameter_name)
        if canonical_name is not None:
            warnings.warn(
                f"parameter name {parameter_name!r} is deprecated for {class_name!r}; use {canonical_name!r}",
                DeprecationWarning,
                stacklevel=2,
            )
            parameter_name = canonical_name
        signal_config = self._signals.get(class_name)
        if signal_config is None:
            return None
        parameter = signal_config.parameters.get(parameter_name)
        return None if parameter is None else float(parameter.value)

    def parameters_for(self, class_name: str) -> Mapping[str, FixedValue]:
        """Return canonical fixed parameter overrides for a signal class."""
        signal_config = self._signals.get(class_name)
        return MappingProxyType({}) if signal_config is None else signal_config.parameters

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical configuration suitable for YAML serialization."""
        return {
            "signals": {class_name: {"parameters": {name: {"value": parameter.value} for name, parameter in signal_config.parameters.items()}} for class_name, signal_config in self._signals.items()}
        }


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return value


def _validate_fixed_value(class_name: str, parameter_name: str, value: Any) -> None:
    location = f"signals.{class_name}.parameters.{parameter_name}.value"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be a real number")
    if parameter_name == "alpha_rolloff" and not 0.0 < float(value) < 1.0:
        raise ValueError(f"{location} must be between 0 and 1 (exclusive)")


def _config_from_mapping(loaded: Any) -> ExperimentConfig:
    root = _mapping(loaded, "experiment configuration root")
    unknown_root = set(root).difference({"signals"})
    if unknown_root:
        raise ValueError(f"unknown experiment configuration settings: {sorted(unknown_root)}")
    signals = _mapping(root.get("signals", {}), "signals")
    parsed_signals: dict[str, SignalConfig] = {}

    for class_name, class_config_value in signals.items():
        if not isinstance(class_name, str) or class_name not in public_generator_names:
            raise ValueError(f"unknown signal class {class_name!r}")
        class_config = _mapping(class_config_value, f"signals.{class_name}")
        unknown_class_settings = set(class_config).difference({"parameters"})
        if unknown_class_settings:
            raise ValueError(f"unknown settings for signal class {class_name!r}: {sorted(unknown_class_settings)}")
        parameters = _mapping(class_config.get("parameters", {}), f"signals.{class_name}.parameters")

        parsed_parameters: dict[str, FixedValue] = {}
        for parameter_name, parameter_config_value in parameters.items():
            parameter_config = _mapping(parameter_config_value, f"signals.{class_name}.parameters.{parameter_name}")
            unknown_parameter_settings = set(parameter_config).difference({"value"})
            if unknown_parameter_settings:
                raise ValueError(f"unknown settings for parameter {class_name}.{parameter_name}: {sorted(unknown_parameter_settings)}")
            if "value" not in parameter_config:
                raise ValueError(f"signals.{class_name}.parameters.{parameter_name} requires a value")
            parsed_parameters[parameter_name] = FixedValue(parameter_config["value"])
        parsed_signals[class_name] = SignalConfig(parameters=parsed_parameters)

    return ExperimentConfig(signals=parsed_signals)


def load_experiment_config(
    config: ExperimentConfig | str | Path | Mapping[str, Any] | None,
) -> ExperimentConfig:
    """Load and validate an optional experiment configuration.

    Args:
        config: A YAML path, mapping, validated configuration, or ``None``.

    Returns:
        A validated configuration. ``None`` resolves to an empty configuration.

    Raises:
        TypeError: If a configuration object has an invalid type.
        ValueError: If the schema, class, parameter, or value is invalid.
    """
    if config is None:
        return ExperimentConfig()
    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise ValueError(f"invalid experiment configuration YAML in {path}: {error}") from error
        except OSError as error:
            raise ValueError(f"could not read experiment configuration {path}: {error}") from error
    else:
        loaded = config

    return _config_from_mapping(loaded)
