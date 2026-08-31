"""Debug logging support for hierarchical metadata objects."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from torchsig.utils.metadata_logging import (
    _MISSING_METADATA_VALUE,
    MetadataDebugConfig,
    MetadataDebugStatistics,
    _create_metadata_debug_session,
    _MetadataDebugSession,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class MetadataDebugMixin:
    """Provide structured debug logging for hierarchical metadata access.

    Classes using this mixin must provide:

    - an ``_metadata`` dictionary
    - a ``parent`` attribute
    - an ``explain_metadata`` method
    - an ``_is_metadata_debug_target`` method
    """

    def _initialize_metadata_debug(self) -> None:
        """Initialize metadata debug state.

        This must be called by the concrete class during initialization.
        """
        self._metadata_debug_enabled = False
        self._metadata_debug_session: _MetadataDebugSession | None = None

    @property
    def metadata_debug_enabled(self) -> bool:
        """Whether structured metadata debug logging is enabled."""
        instance_attributes = object.__getattribute__(self, "__dict__")
        return bool(instance_attributes.get("_metadata_debug_enabled", False))

    @property
    def metadata_debug_config(self) -> MetadataDebugConfig | None:
        """Return the current or most recent debug configuration."""
        session = self._get_metadata_debug_session()
        return None if session is None else session.config

    @property
    def metadata_debug_statistics(self) -> MetadataDebugStatistics:
        """Return event counts for the current metadata debug session."""
        session = self._get_metadata_debug_session()
        if session is None:
            return MetadataDebugStatistics(0, 0)
        return session.statistics

    def enable_metadata_debug(
        self,
        *,
        keys: set[str] | frozenset[str] | None = None,
        events: set[str] | frozenset[str] | None = None,
        max_events: int | None = None,
        include_values: bool = False,
        value_repr_limit: int = 200,
    ) -> None:
        """Enable structured metadata debug logging.

        Records are emitted at ``DEBUG`` level through the
        ``torchsig.metadata`` logger. This method does not configure handlers
        or logging levels.

        Args:
            keys: Exact metadata keys to log, or ``None`` for every key.
            events: Operations to log from ``lookup``, ``set``, ``delete``,
                and ``snapshot``. ``None`` enables all operations.
            max_events: Maximum event records to emit. ``None`` is unlimited.
            include_values: Include bounded value representations in records.
            value_repr_limit: Maximum length of included value representations.
        """
        self._metadata_debug_session = _create_metadata_debug_session(
            keys=keys,
            events=events,
            max_events=max_events,
            include_values=include_values,
            value_repr_limit=value_repr_limit,
        )
        self._metadata_debug_enabled = True

    def disable_metadata_debug(self) -> None:
        """Emit a summary and disable metadata debug logging."""
        if self.metadata_debug_enabled:
            self._emit_metadata_debug_summary()

        self._metadata_debug_enabled = False

    def log_metadata_snapshot(
        self,
        metadata_object: Any | None = None,
        *,
        include_components: bool = False,
    ) -> None:
        """Log one structured metadata snapshot.

        Snapshot values are included only when debugging was enabled with
        ``include_values=True``. Array data is represented only by shape and
        dtype.

        Args:
            metadata_object: Completed object to snapshot. Defaults to this
                object.
            include_components: Include objects in the target's
                ``component_signals`` collection.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        target = self if metadata_object is None else metadata_object

        if not self._is_metadata_debug_target(target):
            raise TypeError("metadata_object must be a HierarchicalMetadataObject")
        if not isinstance(include_components, bool):
            raise TypeError("include_components must be a boolean")

        session = self._get_metadata_debug_session()
        if not self.metadata_debug_enabled or session is None or not session.should_emit("snapshot"):
            return

        session.emit_snapshot(
            target,
            include_components=include_components,
        )

    @contextmanager
    def metadata_debug(
        self,
        *,
        keys: set[str] | frozenset[str] | None = None,
        events: set[str] | frozenset[str] | None = None,
        max_events: int | None = None,
        include_values: bool = False,
        value_repr_limit: int = 200,
    ) -> Iterator[Any]:
        """Temporarily enable structured metadata debug logging.

        The previous session and enabled state are restored when the context
        exits, including when an exception is raised.

        Yields:
            This object with metadata debugging enabled.
        """
        previous_enabled = self.metadata_debug_enabled
        previous_session = self._get_metadata_debug_session()

        self.enable_metadata_debug(
            keys=keys,
            events=events,
            max_events=max_events,
            include_values=include_values,
            value_repr_limit=value_repr_limit,
        )

        try:
            yield self
        finally:
            self._emit_metadata_debug_summary()
            self._metadata_debug_enabled = previous_enabled
            self._metadata_debug_session = previous_session

    def _log_metadata_event(
        self,
        event: Literal["lookup", "set", "delete"],
        key: str,
        value: Any = _MISSING_METADATA_VALUE,
    ) -> None:
        """Emit a filtered, structured metadata debug record."""
        session = self._get_metadata_debug_session()

        if not self.metadata_debug_enabled or session is None or not session.should_emit(event, key):
            return

        resolution = self.explain_metadata(key)

        if session.config.include_values and value is _MISSING_METADATA_VALUE:
            value = self._get_metadata_value_for_debug(key)

        session.emit_event(
            event,
            key,
            resolution,
            type(self).__name__,
            value,
        )

    def _get_metadata_debug_session(
        self,
    ) -> _MetadataDebugSession | None:
        """Return the current debug session without normal attribute lookup."""
        instance_attributes = object.__getattribute__(self, "__dict__")
        return instance_attributes.get("_metadata_debug_session")

    def _emit_metadata_debug_summary(self) -> None:
        """Emit a summary for the current session, when one exists."""
        session = self._get_metadata_debug_session()
        if session is not None:
            session.emit_summary(type(self).__name__)

    def _get_metadata_value_for_debug(self, key: str) -> Any:
        """Resolve a value without invoking normal metadata lookup hooks."""
        if key == "metadata":
            metadata = object.__getattribute__(self, "_metadata")
            return metadata.copy()

        current: Any | None = self
        visited: set[int] = set()

        while current is not None:
            current_id = id(current)
            if current_id in visited:
                break

            visited.add(current_id)

            try:
                current_metadata = object.__getattribute__(
                    current,
                    "_metadata",
                )
            except AttributeError:
                break

            if key in current_metadata:
                return current_metadata[key]

            try:
                parent = object.__getattribute__(current, "parent")
            except AttributeError:
                break

            if not self._is_metadata_debug_target(parent):
                break

            current = parent

        return _MISSING_METADATA_VALUE

    def _is_metadata_debug_target(self, target: Any) -> bool:
        """Return whether an object can be processed by this debug facility."""
        raise NotImplementedError
