"""Metadata Transforms"""

import ast
import copy
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml

from torchsig.signals.signal_lists import CLASS_FAMILY_DICT
from torchsig.signals.signal_types import Signal
from torchsig.transforms.base_transforms import Transform
from torchsig.utils.printing import generate_repr_str

__all__ = [
    "GroupingLabel",
    "MetadataTransform",
    "MultiHotLabel",
    "YOLOLabel",
]


def _build_family_grouping_config() -> dict[str, Any]:
    """Build the canonical TorchSIG family-grouping preset."""
    classes_by_family: dict[str, list[str]] = {}
    for class_name, family_name in CLASS_FAMILY_DICT.items():
        classes_by_family.setdefault(family_name, []).append(class_name)

    return {
        "source": "class_name",
        "labels": {
            "name": "family_name",
            "index": "family_index",
        },
        "groups": [
            {
                "name": family_name,
                "values": sorted(classes_by_family[family_name]),
            }
            for family_name in sorted(classes_by_family)
        ],
    }


_BUILTIN_GROUPING_CONFIGS = {
    "family": _build_family_grouping_config(),
}


## Base/Helper Classes
class MetadataTransform(Transform):
    """Base class for metadata transforms.

    This class defines the basic structure of a metadata transform, which includes:
    - The ability to validate metadata before applying the transform.
    - A method for applying the transform on signal metadata.
    - A callable interface to apply the transform to a list of signal metadata.

    Attributes:
        required_metadata: List of metadata fields required for applying the target transform.

    Methods:
        __validate(metadata): Validates the signal metadata before applying the transform.
        __apply(metadata): Applies the target transform to the metadata. Should be overridden by subclasses.
        __call__(signal): Applies the transform to a list of signal metadata dictionaries.
        __str__(): Returns the string representation of the transform.
        __repr__(): Returns a detailed string representation of the transform object.
    """

    def __init__(self, required_metadata: list[str] = [], **kwargs) -> None:
        """Initialize the MetadataTransform.

        Args:
            required_metadata: List of metadata fields required for applying the target transform.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(required_metadata=required_metadata, **kwargs)

    def __validate__(self, signal):
        """Validate signal metadata before applying target transforms.

        Makes sure a signal has all required metadata for a transform;
        returns the original signal if it is valid; raises an exception otherwise.

        Args:
            signal: The signal to validate.

        Raises:
            ValueError: If metadata is missing required metadata fields or if input is not a Signal object.
        """
        if not isinstance(signal, Signal):
            raise TypeError(f"input ({type(signal)}) is not a Signal object.")
        for required_metadatum in self.required_metadata:
            if not hasattr(signal, required_metadatum):
                raise ValueError(f"key: {required_metadatum} is missing from signal metadata, but is required by {self.__class__.__name__}")
        return signal

    def __call__(self, signal: Signal) -> Signal:
        """Applies the target transform to a list of signal metadata.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal.
        """
        signals = signal.component_signals or [signal]
        for target_signal in signals:
            self.__apply__(target_signal)
        return signal

    def __apply__(self, signal):
        """Applies the target transform to a single signal metadata.

        Args:
            signal: The signal to transform.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Returns a detailed string representation of the transform object.

        Returns:
            A string representation of the transform object.
        """
        return generate_repr_str(self, exclude_params=["required_metadata"])


class MultiHotLabel(MetadataTransform):
    """Add a sample-level multi-hot classification label.

    Each class present in a signal's components is represented by a ``1`` in
    the output vector. Repeated instances of the same class still produce a
    single ``1``. For a signal without components, its own ``class_index`` is
    used. An empty composite signal produces an all-zero vector.

    This transform is intended for multilabel classification of composite
    samples such as wideband signals. The number of classes can be supplied
    explicitly or inferred from the signal's ``class_names`` metadata.

    Args:
        num_classes: Length of the output vector. If ``None``, infer the
            length from the signal's ``class_names`` metadata.
        output_key: Metadata key under which to store the vector.
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        targets_metadata: Metadata fields added by the transform.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        output_key: str = "multi_hot_label",
        **kwargs,
    ) -> None:
        if num_classes is not None and (not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1):
            raise ValueError("num_classes must be a positive integer or None")
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("output_key must be a non-empty string")

        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.output_key = output_key
        self.targets_metadata = [output_key]

    def __call__(self, signal: Signal) -> Signal:
        """Add a multi-hot vector representing all classes in ``signal``.

        Args:
            signal: Signal whose component class indices should be encoded.

        Returns:
            The input signal with the sample-level label added.

        Raises:
            TypeError: If ``signal`` is not a ``Signal`` or a class index is
                not an integer.
            ValueError: If the class count cannot be inferred or a class
                index is outside the output vector.
        """
        self.__validate__(signal)

        num_classes = self.num_classes
        if num_classes is None:
            metadata = signal.get_full_metadata()
            if "class_names" not in metadata:
                raise ValueError("num_classes was not provided and class_names is missing from signal metadata")
            num_classes = len(metadata["class_names"])
            if num_classes < 1:
                raise ValueError("class_names must contain at least one class")

        label = np.zeros(num_classes, dtype=np.float32)
        if signal.component_signals:
            signals = signal.component_signals
        elif "class_index" in signal.metadata:
            signals = [signal]
        else:
            signals = []

        for component in signals:
            if not hasattr(component, "class_index"):
                raise ValueError("class_index is missing from signal metadata")
            class_index = component.class_index
            if not isinstance(class_index, (int, np.integer)) or isinstance(class_index, (bool, np.bool_)):
                raise TypeError("class_index must be an integer")
            if class_index < 0 or class_index >= num_classes:
                raise ValueError(f"class_index {class_index} is outside [0, {num_classes})")
            label[int(class_index)] = 1.0

        signal[self.output_key] = label
        return signal

    def __apply__(self, signal: Signal) -> Signal:
        """Apply the transform to a single signal.

        ``MultiHotLabel`` aggregates a complete sample in ``__call__`` and
        therefore does not apply labels independently to components.

        Args:
            signal: Signal to transform.

        Returns:
            The transformed signal.
        """
        return self(signal)


class YOLOLabel(MetadataTransform):
    """Adds a YOLO_label to a signal.

    This transform adds a YOLO_label to a signal in the form of a list of tuples (cid, cx, cy, width, height).

    Attributes:
        required_metadata: List of metadata fields required for applying the transform.
        targets_metadata: List of metadata fields that will be added by the transform.
    """

    def __init__(self, **kwargs):
        """Initialize the YOLOLabel transform.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            required_metadata=[
                "class_index",
                "start",
                "bandwidth",
                "center_freq",
                "dataset_metadata",
            ],
            **kwargs,
        )
        self.targets_metadata = ["yolo_label"]

    def __apply__(self, signal: Signal) -> Signal:
        """Applies the YOLOLabel transform to a single signal.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal with YOLO_label added.
        """
        class_index = signal.class_index
        # normalized to width of sample
        width = signal.duration
        # normalize bandwidth with sample rate
        height = signal.bandwidth / signal.sample_rate
        x_center = signal.start + (width / 2.0)
        # normalize center frequency with sample rate
        # subtract from 1 since (0,0) for YOLO is upper left, but we define (0,0) lower left
        y_center = 1 - ((signal.sample_rate / 2.0) + signal.center_freq) / signal.sample_rate
        yolo_label = (class_index, x_center, y_center, width, height)
        signal["yolo_label"] = yolo_label
        return signal


class GroupingLabel(MetadataTransform):
    """Add generic group labels using ordered rules from YAML or a dictionary.

    Each group must define a ``name`` and exactly one matching rule:

    - ``values``: A list of exact source values.
    - ``regex``: A regular expression searched against the string source value.
    - ``formula``: A restricted boolean expression using ``value``.
    - ``default: true``: Match any value not handled by an earlier group.

    Groups are evaluated in order and the first match wins. Their order also
    determines the numeric group index.

    The formula language supports arithmetic, boolean operators, comparisons,
    membership, literals, and the safe string methods ``startswith``,
    ``endswith``, ``lower``, ``upper``, ``isdigit``, and ``isalpha``. It
    cannot import modules, access arbitrary attributes, or call arbitrary
    functions.

    Example YAML:
        .. code-block:: yaml

            source: class_name
            labels:
              name: modulation_group
              index: modulation_group_index
            groups:
              - name: linear
                values: [bpsk, qpsk]
              - name: frequency_shift
                regex: '^[248]g?fsk$'
              - name: high_order_qam
                formula: 'value.endswith("qam") and value != "16qam"'
              - name: all
                default: true

    Args:
        config: Built-in preset name, YAML file path, or a mapping with the
            same schema. The ``"family"`` preset uses TorchSIG's canonical
            class-to-family mapping.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    _SAFE_STRING_METHODS: ClassVar[set[str]] = {
        "endswith",
        "isalpha",
        "isdigit",
        "lower",
        "startswith",
        "upper",
    }
    _SAFE_AST_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.UAdd,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.In,
        ast.NotIn,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Call,
        ast.Attribute,
    )

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        **kwargs,
    ) -> None:
        """Initialize a grouping transform from YAML or a mapping."""
        grouping_config = self._load_config(config)
        self.source = grouping_config.get("source", "class_name")
        labels = grouping_config.get("labels", {})
        self.name_label = labels.get("name", "group_name")
        self.index_label = labels.get("index", "group_index")
        self.groups = grouping_config.get("groups")

        self._validate_config()
        self._compiled_rules = [self._compile_rule(group) for group in self.groups]

        super().__init__(required_metadata=[self.source], **kwargs)
        self.targets_metadata = [self.name_label, self.index_label]

    @staticmethod
    def _load_config(
        config: str | Path | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load and copy grouping configuration."""
        if isinstance(config, Mapping):
            return copy.deepcopy(dict(config))
        if isinstance(config, str) and config in _BUILTIN_GROUPING_CONFIGS:
            return copy.deepcopy(_BUILTIN_GROUPING_CONFIGS[config])
        if isinstance(config, (str, Path)):
            path = Path(config)
            loaded = yaml.safe_load(path.read_text())
            if not isinstance(loaded, dict):
                raise TypeError("grouping YAML root must be a mapping")
            return loaded
        raise TypeError("config must be a YAML path or mapping")

    @staticmethod
    def available_presets() -> tuple[str, ...]:
        """Return the names of built-in grouping configurations."""
        return tuple(sorted(_BUILTIN_GROUPING_CONFIGS))

    def _validate_config(self) -> None:
        """Validate grouping configuration structure and label names."""
        for field_name, value in (
            ("source", self.source),
            ("labels.name", self.name_label),
            ("labels.index", self.index_label),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.name_label == self.index_label:
            raise ValueError("group name and index labels must be different")
        if not isinstance(self.groups, list) or not self.groups:
            raise ValueError("groups must be a non-empty list")

        names = []
        for group_index, group in enumerate(self.groups):
            if not isinstance(group, dict):
                raise TypeError("each group must be a mapping")
            name = group.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("each group name must be a non-empty string")
            names.append(name)

            rule_names = {
                "values",
                "regex",
                "formula",
                "default",
            } & group.keys()
            if len(rule_names) != 1:
                raise ValueError(f"group {name!r} must define exactly one of values, regex, formula, or default")
            if "default" in group:
                if group["default"] is not True:
                    raise ValueError(f"group {name!r} default rule must be true")
                if group_index != len(self.groups) - 1:
                    raise ValueError("the default group must be last")

        if len(names) != len(set(names)):
            raise ValueError("group names must be unique")

        probability_groups = [group for group in self.groups if "probability" in group]
        likelihood_groups = [group for group in self.groups if "likelihood" in group]
        if probability_groups and likelihood_groups:
            raise ValueError("grouping config cannot mix probability and likelihood")

        if probability_groups:
            probabilities = []
            for group in self.groups:
                if "probability" not in group:
                    warnings.warn(
                        f"probability is missing for group {group['name']!r}; defaulting to 0.0",
                        UserWarning,
                        stacklevel=3,
                    )
                    group["probability"] = 0.0
                probability = group["probability"]
                probabilities.append(
                    self._validate_sampling_weight(
                        probability,
                        "probability",
                        group["name"],
                        allow_zero=True,
                    )
                )
            probability_sum = float(np.sum(probabilities))
            if not np.isclose(probability_sum, 1.0, atol=1e-8):
                raise ValueError(f"group probabilities must sum to 1.0, found {probability_sum}")
        elif likelihood_groups:
            for group in likelihood_groups:
                self._validate_sampling_weight(
                    group["likelihood"],
                    "likelihood",
                    group["name"],
                    allow_zero=False,
                )

    @staticmethod
    def _validate_sampling_weight(
        value: Any,
        weight_name: str,
        group_name: str,
        *,
        allow_zero: bool,
    ) -> float:
        """Validate a probability or likelihood stored on a group."""
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(f"group {group_name!r} {weight_name} must be a real number")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"group {group_name!r} {weight_name} must be finite")
        if value < 0.0 or (value == 0.0 and not allow_zero):
            comparator = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"group {group_name!r} {weight_name} must be {comparator}")
        return value

    def _compile_rule(self, group: dict[str, Any]) -> tuple[str, Any]:
        """Validate and compile one configured matching rule."""
        if "default" in group:
            return "default", None

        if "values" in group:
            values = group["values"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"group {group['name']!r} values must be a non-empty list")
            return "values", values

        if "regex" in group:
            pattern = group["regex"]
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"group {group['name']!r} regex must be a non-empty string")
            try:
                return "regex", re.compile(pattern)
            except re.error as error:
                raise ValueError(f"group {group['name']!r} has invalid regex: {error}") from error

        formula = group["formula"]
        if not isinstance(formula, str) or not formula:
            raise ValueError(f"group {group['name']!r} formula must be a non-empty string")
        return "formula", self._compile_formula(formula)

    def _compile_formula(self, formula: str) -> Any:
        """Compile a formula after enforcing the restricted expression grammar."""
        try:
            expression = ast.parse(formula, mode="eval")
        except SyntaxError as error:
            raise ValueError(f"invalid grouping formula: {error.msg}") from error

        for node in ast.walk(expression):
            if not isinstance(node, self._SAFE_AST_NODES):
                raise ValueError(  # noqa: TRY004 - invalid formula syntax
                    f"grouping formula uses unsupported syntax: {type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id != "value":
                raise ValueError("grouping formula may only reference the name 'value'")
            if isinstance(node, ast.Attribute) and node.attr not in self._SAFE_STRING_METHODS:
                raise ValueError(f"grouping formula method {node.attr!r} is not allowed")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Attribute):
                    raise ValueError(  # noqa: TRY004 - invalid formula call
                        "grouping formula may only call safe string methods"
                    )
                if node.keywords:
                    raise ValueError("grouping formula method calls do not accept keywords")

        return expression.body

    def _evaluate_formula(  # noqa: PLR0911 - one branch per allowed AST node
        self,
        node: ast.AST,
        value: Any,
    ) -> Any:
        """Evaluate a previously validated formula syntax tree."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [self._evaluate_formula(item, value) for item in node.elts]
            if isinstance(node, ast.List):
                return items
            if isinstance(node, ast.Tuple):
                return tuple(items)
            return set(items)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(self._evaluate_formula(item, value)) for item in node.values)
            return any(bool(self._evaluate_formula(item, value)) for item in node.values)
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_formula(node.operand, value)
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            return -operand
        if isinstance(node, ast.BinOp):
            left = self._evaluate_formula(node.left, value)
            right = self._evaluate_formula(node.right, value)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return left % right
        if isinstance(node, ast.Compare):
            left = self._evaluate_formula(node.left, value)
            for operator, comparator in zip(
                node.ops,
                node.comparators,
                strict=True,
            ):
                right = self._evaluate_formula(comparator, value)
                if isinstance(operator, ast.Eq):
                    matches = left == right
                elif isinstance(operator, ast.NotEq):
                    matches = left != right
                elif isinstance(operator, ast.In):
                    matches = left in right
                elif isinstance(operator, ast.NotIn):
                    matches = left not in right
                elif isinstance(operator, ast.Lt):
                    matches = left < right
                elif isinstance(operator, ast.LtE):
                    matches = left <= right
                elif isinstance(operator, ast.Gt):
                    matches = left > right
                else:
                    matches = left >= right
                if not matches:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            target = self._evaluate_formula(node.func.value, value)
            method = getattr(target, node.func.attr)
            args = [self._evaluate_formula(argument, value) for argument in node.args]
            return method(*args)
        raise TypeError(f"unsupported validated formula node: {type(node).__name__}")

    def _rule_matches(
        self,
        rule_type: str,
        rule: Any,
        value: Any,
    ) -> bool:
        """Return whether one compiled rule matches a source value."""
        if rule_type == "values":
            return value in rule
        if rule_type == "regex":
            return rule.search(str(value)) is not None
        if rule_type == "default":
            return True
        return bool(self._evaluate_formula(rule, value))

    def match(self, value: Any) -> tuple[str, int]:
        """Return the configured group name and index for a source value.

        This method applies the same ordered rules used by the metadata
        transform without requiring or mutating a :class:`Signal`. It can be
        used to configure sampling distributions before signal generation.

        Args:
            value: Value from the configured source metadata field.

        Returns:
            A tuple containing the matched group name and numeric index.

        Raises:
            ValueError: If no configured group matches the value.
        """
        for group_index, (group, compiled_rule) in enumerate(zip(self.groups, self._compiled_rules, strict=True)):
            rule_type, rule = compiled_rule
            if self._rule_matches(rule_type, rule, value):
                return group["name"], group_index

        raise ValueError(f"value {value!r} did not match any configured group")

    def __apply__(self, signal: Signal) -> Signal:
        """Assign the first matching group to a signal.

        Raises:
            ValueError: If the source field is missing or no group matches.
        """
        if not hasattr(signal, self.source):
            raise ValueError(f"GroupingLabel requires signal metadata field {self.source!r}")
        value = getattr(signal, self.source)

        try:
            group_name, group_index = self.match(value)
        except ValueError as error:
            raise ValueError(f"value {value!r} from metadata field {self.source!r} did not match any configured group") from error

        signal[self.name_label] = group_name
        signal[self.index_label] = group_index
        return signal
