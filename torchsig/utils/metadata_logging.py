"""Correlation context for structured TorchSIG metadata logging."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pprint import pformat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torchsig.utils.abstractions import MetadataResolution

MetadataContextValue: TypeAlias = str | int | float | bool | None
MetadataDebugEvent: TypeAlias = Literal["lookup", "set", "delete", "snapshot"]
_CONTEXT_VALUE_LIMIT = 200
_METADATA_DEBUG_EVENTS = frozenset({"lookup", "set", "delete", "snapshot"})
_MISSING_METADATA_VALUE = object()
_log = logging.getLogger("torchsig.metadata")


class MetadataDebugFormatter(logging.Formatter):
    """Format structured metadata operation and summary records for humans."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a compact record with correlation and resolution fields."""
        if getattr(record, "metadata_event", None) == "summary":
            return (
                "metadata debug summary: "
                f"emitted={getattr(record, 'metadata_emitted_events', None)} "
                f"suppressed={getattr(record, 'metadata_suppressed_events', None)} "
                f"filtered={getattr(record, 'metadata_filtered_events', None)}"
            )

        fields = getattr(record, "metadata_correlation_fields", {})
        value = getattr(record, "metadata_value", "<values disabled>")
        return (
            f"session={getattr(record, 'metadata_session_id', None)} "
            f"sample={getattr(record, 'metadata_sample_index', None)} "
            f"worker={getattr(record, 'metadata_worker_id', None)} "
            f"stage={fields.get('stage')} "
            f"event={getattr(record, 'metadata_event', None)} "
            f"key={getattr(record, 'metadata_key', None)} "
            f"source={getattr(record, 'metadata_source', None)} "
            f"depth={getattr(record, 'metadata_depth', None)} "
            f"value={value}"
        )


class MetadataSnapshotFormatter(logging.Formatter):
    """Format structured metadata snapshot and summary records for humans."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a readable completed snapshot or debug-session summary."""
        if getattr(record, "metadata_event", None) == "summary":
            return f"metadata snapshot summary: emitted={getattr(record, 'metadata_emitted_events', None)} suppressed={getattr(record, 'metadata_suppressed_events', None)}"

        context = getattr(record, "metadata_correlation_fields", {})
        snapshot = {
            "session_id": getattr(record, "metadata_session_id", None),
            "dataset_id": getattr(record, "metadata_dataset_id", None),
            "sample_index": getattr(record, "metadata_sample_index", None),
            "worker_id": getattr(record, "metadata_worker_id", None),
            "stage": context.get("stage"),
            "data_shape": getattr(record, "metadata_data_shape", None),
            "data_dtype": getattr(record, "metadata_data_dtype", None),
            "metadata": getattr(record, "metadata_snapshot", None),
            "component_metadata": getattr(
                record,
                "metadata_component_snapshots",
                (),
            ),
        }
        return "completed metadata snapshot:\n" + pformat(
            snapshot,
            sort_dicts=False,
            width=100,
        )


@dataclass(frozen=True)
class MetadataDebugConfig:
    """Configuration for structured metadata debug logging.

    Attributes:
        keys: Exact metadata keys to log, or ``None`` to log every key.
        events: Metadata operations to log.
        max_events: Maximum event records to emit, or ``None`` for no limit.
        include_values: Whether records include bounded value representations.
        value_repr_limit: Maximum characters in a logged value representation.
    """

    keys: frozenset[str] | None
    events: frozenset[MetadataDebugEvent]
    max_events: int | None
    include_values: bool
    value_repr_limit: int


@dataclass(frozen=True)
class MetadataDebugStatistics:
    """Counts of metadata events handled during a debug session.

    Attributes:
        emitted_events: Event records emitted through the metadata logger.
        suppressed_events: Selected events rejected by rate limits or the
            logger's effective level.
        filtered_events: Events intentionally excluded by key or event filters.
    """

    emitted_events: int
    suppressed_events: int
    filtered_events: int = 0


@dataclass
class _MetadataDebugSession:
    """Mutable logging state owned by one hierarchical metadata object."""

    config: MetadataDebugConfig
    emitted_events: int = 0
    suppressed_events: int = 0
    filtered_events: int = 0

    @property
    def statistics(self) -> MetadataDebugStatistics:
        """Return an immutable snapshot of the session counters."""
        return MetadataDebugStatistics(
            emitted_events=self.emitted_events,
            suppressed_events=self.suppressed_events,
            filtered_events=self.filtered_events,
        )

    def should_emit(self, event: MetadataDebugEvent, key: str | None = None) -> bool:
        """Apply filters and rate limits before constructing a log record."""
        if event not in self.config.events or (key is not None and self.config.keys is not None and key not in self.config.keys):
            self.filtered_events += 1
            return False
        if (self.config.max_events is not None and self.emitted_events >= self.config.max_events) or not _log.isEnabledFor(logging.DEBUG):
            self.suppressed_events += 1
            return False
        return True

    def emit_event(
        self,
        event: MetadataDebugEvent,
        key: str,
        resolution: MetadataResolution,
        object_type: str,
        value: Any = _MISSING_METADATA_VALUE,
    ) -> None:
        """Emit one structured metadata event after filtering succeeds."""
        extra = {
            "metadata_event": event,
            "metadata_key": key,
            "metadata_source": resolution.source,
            "metadata_found": resolution.found,
            "metadata_depth": resolution.depth,
            "metadata_owner_type": resolution.owner_type,
            "metadata_overrides_parent": resolution.overrides_parent,
            "metadata_cycle_detected": resolution.cycle_detected,
            "metadata_path": resolution.path,
            "metadata_object_type": object_type,
            **_metadata_logging_extra(),
        }
        if self.config.include_values:
            value_repr, truncated = _bounded_metadata_value_repr(
                value,
                self.config.value_repr_limit,
            )
            extra["metadata_value"] = value_repr
            extra["metadata_value_truncated"] = truncated

        _log.debug(
            "metadata %s: key=%r source=%s depth=%s owner=%s",
            event,
            key,
            resolution.source,
            resolution.depth,
            resolution.owner_type,
            extra=extra,
        )
        self.emitted_events += 1

    def emit_snapshot(
        self,
        metadata_object: Any,
        *,
        include_components: bool,
    ) -> None:
        """Emit one structured snapshot of a completed metadata object."""
        full_metadata = _get_full_metadata_without_hooks(metadata_object)
        selected_metadata = _select_snapshot_metadata(
            full_metadata,
            self.config.keys,
        )
        components = (
            _get_attribute_without_metadata_hooks(
                metadata_object,
                "component_signals",
                [],
            )
            if include_components
            else []
        )
        component_metadata = [
            _select_snapshot_metadata(
                _get_full_metadata_without_hooks(component),
                self.config.keys,
            )
            for component in components
        ]
        data = _get_attribute_without_metadata_hooks(metadata_object, "data", None)
        extra = {
            "metadata_event": "snapshot",
            "metadata_object_type": type(metadata_object).__name__,
            "metadata_snapshot_keys": tuple(selected_metadata),
            "metadata_component_snapshot_keys": tuple(tuple(metadata) for metadata in component_metadata),
            "metadata_component_count": len(component_metadata),
            "metadata_data_shape": getattr(data, "shape", None),
            "metadata_data_dtype": str(getattr(data, "dtype", "")) or None,
            **_metadata_logging_extra(),
        }
        if self.config.include_values:
            extra["metadata_snapshot"] = _snapshot_value_reprs(
                selected_metadata,
                self.config.value_repr_limit,
            )
            extra["metadata_component_snapshots"] = tuple(_snapshot_value_reprs(metadata, self.config.value_repr_limit) for metadata in component_metadata)

        _log.debug(
            "metadata snapshot: object=%s keys=%d components=%d",
            type(metadata_object).__name__,
            len(selected_metadata),
            len(component_metadata),
            extra=extra,
        )
        self.emitted_events += 1

    def emit_summary(self, object_type: str) -> None:
        """Emit one summary record for this debug session."""
        if not _log.isEnabledFor(logging.DEBUG):
            return
        _log.debug(
            "metadata debug summary: emitted=%d suppressed=%d filtered=%d",
            self.emitted_events,
            self.suppressed_events,
            self.filtered_events,
            extra={
                "metadata_event": "summary",
                "metadata_object_type": object_type,
                **_metadata_logging_extra(),
                "metadata_emitted_events": self.emitted_events,
                "metadata_suppressed_events": self.suppressed_events,
                "metadata_filtered_events": self.filtered_events,
                "metadata_debug_keys": self.config.keys,
                "metadata_debug_events": self.config.events,
                "metadata_debug_max_events": self.config.max_events,
                "metadata_debug_include_values": self.config.include_values,
            },
        )


def _create_metadata_debug_session(
    *,
    keys: set[str] | frozenset[str] | None,
    events: set[str] | frozenset[str] | None,
    max_events: int | None,
    include_values: bool,
    value_repr_limit: int,
) -> _MetadataDebugSession:
    """Validate configuration and create a fresh metadata debug session."""
    if keys is not None:
        if not isinstance(keys, (set, frozenset)) or not all(isinstance(key, str) for key in keys):
            raise TypeError("keys must be a set of strings or None")
        normalized_keys = frozenset(keys)
    else:
        normalized_keys = None

    if events is None:
        normalized_events = _METADATA_DEBUG_EVENTS
    elif not isinstance(events, (set, frozenset)) or not all(isinstance(event, str) for event in events):
        raise TypeError("events must be a set of strings or None")
    else:
        unknown_events = events - _METADATA_DEBUG_EVENTS
        if unknown_events:
            raise ValueError(f"unknown metadata debug events: {sorted(unknown_events)}")
        normalized_events = frozenset(events)

    if max_events is not None and (not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 0):
        raise ValueError("max_events must be a non-negative integer or None")
    if not isinstance(include_values, bool):
        raise TypeError("include_values must be a boolean")
    if not isinstance(value_repr_limit, int) or isinstance(value_repr_limit, bool) or value_repr_limit < 1:
        raise ValueError("value_repr_limit must be a positive integer")

    return _MetadataDebugSession(
        MetadataDebugConfig(
            keys=normalized_keys,
            events=normalized_events,
            max_events=max_events,
            include_values=include_values,
            value_repr_limit=value_repr_limit,
        )
    )


def _bounded_metadata_value_repr(value: Any, limit: int) -> tuple[str, bool]:
    """Return a safe, bounded representation of a metadata value."""
    if value is _MISSING_METADATA_VALUE:
        return "<missing>", False
    try:
        value_repr = repr(value)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        value_repr = f"<repr failed: {type(exc).__name__}>"
    if len(value_repr) <= limit:
        return value_repr, False
    return value_repr[: max(0, limit - 3)] + "...", True


def _select_snapshot_metadata(
    metadata: dict[str, Any],
    keys: frozenset[str] | None,
) -> dict[str, Any]:
    """Select snapshot keys while preserving metadata insertion order."""
    if keys is None:
        return metadata
    return {key: value for key, value in metadata.items() if key in keys}


def _get_full_metadata_without_hooks(metadata_object: Any) -> dict[str, Any]:
    """Resolve inherited metadata without generating metadata access events."""
    hierarchy = []
    visited: set[int] = set()
    current = metadata_object
    while current is not None and id(current) not in visited:
        try:
            local_metadata = object.__getattribute__(current, "_metadata")
        except AttributeError:
            break
        visited.add(id(current))
        hierarchy.append(local_metadata)
        try:
            current = object.__getattribute__(current, "parent")
        except AttributeError:
            break

    full_metadata: dict[str, Any] = {}
    for local_metadata in reversed(hierarchy):
        full_metadata.update(local_metadata)
    return full_metadata


def _get_attribute_without_metadata_hooks(
    metadata_object: Any,
    name: str,
    default: Any,
) -> Any:
    """Read an ordinary attribute without falling back to metadata lookup."""
    try:
        return object.__getattribute__(metadata_object, name)
    except AttributeError:
        return default


def _snapshot_value_reprs(
    metadata: dict[str, Any],
    limit: int,
) -> dict[str, str]:
    """Return bounded representations for all values in a metadata snapshot."""
    return {key: _bounded_metadata_value_repr(value, limit)[0] for key, value in metadata.items()}


@dataclass(frozen=True)
class MetadataLoggingContext:
    """Correlation information attached to metadata log records.

    Attributes:
        session_id: Identifier shared by records from one debug session.
        dataset_id: Optional dataset identifier.
        sample_index: Optional generated or loaded sample index.
        worker_id: Optional DataLoader or application worker identifier.
        correlation_fields: Additional normalized user-defined fields.
    """

    session_id: str | None = None
    dataset_id: str | None = None
    sample_index: MetadataContextValue = None
    worker_id: MetadataContextValue = None
    correlation_fields: tuple[tuple[str, MetadataContextValue], ...] = ()

    @property
    def fields(self) -> Mapping[str, MetadataContextValue]:
        """Return user-defined correlation fields as a read-only mapping."""
        return MappingProxyType(dict(self.correlation_fields))


_metadata_logging_context: ContextVar[MetadataLoggingContext | None] = ContextVar(
    "torchsig_metadata_logging_context",
    default=None,
)


def _normalize_context_value(
    name: str,
    value: MetadataContextValue,
) -> MetadataContextValue:
    """Validate and bound a correlation value."""
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"metadata logging context value {name!r} must be a scalar or None")
    if isinstance(value, str) and len(value) > _CONTEXT_VALUE_LIMIT:
        return value[: _CONTEXT_VALUE_LIMIT - 3] + "..."
    return value


def get_metadata_logging_context() -> MetadataLoggingContext:
    """Return the correlation context active in the current execution context."""
    return _metadata_logging_context.get() or MetadataLoggingContext()


@contextmanager
def metadata_logging_context(
    *,
    session_id: str | None = None,
    dataset_id: str | None = None,
    sample_index: MetadataContextValue = None,
    worker_id: MetadataContextValue = None,
    fields: Mapping[str, MetadataContextValue] | None = None,
) -> Iterator[MetadataLoggingContext]:
    """Temporarily attach correlation information to metadata log records.

    Unspecified named values inherit from the surrounding context. When no
    session identifier exists, a new UUID is generated. User-defined fields are
    merged with surrounding fields, with inner values taking precedence.

    Args:
        session_id: Debug-session identifier, or ``None`` to inherit or create.
        dataset_id: Dataset identifier, or ``None`` to inherit.
        sample_index: Sample index, or ``None`` to inherit.
        worker_id: Worker identifier, or ``None`` to inherit.
        fields: Additional scalar correlation fields.

    Yields:
        The effective context for the duration of the block.

    Raises:
        TypeError: If identifiers, field names, or field values are invalid.
    """
    current = get_metadata_logging_context()
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string or None")
    if dataset_id is not None and not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string or None")
    if fields is not None and not hasattr(fields, "items"):
        raise TypeError("fields must be a mapping or None")

    merged_fields = dict(current.correlation_fields)
    if fields is not None:
        for key, value in fields.items():
            if not isinstance(key, str) or not key:
                raise TypeError("metadata logging context field names must be non-empty strings")
            merged_fields[key] = _normalize_context_value(key, value)

    effective_session_id = session_id or current.session_id or str(uuid.uuid4())
    effective_dataset_id = dataset_id if dataset_id is not None else current.dataset_id
    effective_sample_index = sample_index if sample_index is not None else current.sample_index
    effective_worker_id = worker_id if worker_id is not None else current.worker_id
    context = MetadataLoggingContext(
        session_id=_normalize_context_value("session_id", effective_session_id),
        dataset_id=_normalize_context_value("dataset_id", effective_dataset_id),
        sample_index=_normalize_context_value("sample_index", effective_sample_index),
        worker_id=_normalize_context_value("worker_id", effective_worker_id),
        correlation_fields=tuple(sorted(merged_fields.items())),
    )
    token = _metadata_logging_context.set(context)
    try:
        yield context
    finally:
        _metadata_logging_context.reset(token)


def _metadata_logging_extra() -> dict[str, object]:
    """Build standard correlation fields for a metadata log record."""
    context = get_metadata_logging_context()
    return {
        "metadata_session_id": context.session_id,
        "metadata_dataset_id": context.dataset_id,
        "metadata_sample_index": context.sample_index,
        "metadata_worker_id": context.worker_id,
        "metadata_process_id": os.getpid(),
        "metadata_thread_id": threading.get_ident(),
        "metadata_correlation_fields": dict(context.correlation_fields),
    }


__all__ = [
    "MetadataContextValue",
    "MetadataDebugConfig",
    "MetadataDebugFormatter",
    "MetadataDebugStatistics",
    "MetadataLoggingContext",
    "MetadataSnapshotFormatter",
    "get_metadata_logging_context",
    "metadata_logging_context",
]
