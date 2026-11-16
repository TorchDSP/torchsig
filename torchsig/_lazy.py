"""Helpers for lazily exported package attributes."""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypeAlias

LazyExport: TypeAlias = tuple[str, str | None]


def lazy_getattr(name: str, module_globals: dict[str, Any], exports: dict[str, LazyExport]) -> Any:
    """Import, cache, and return a lazily exported module or symbol."""
    try:
        module_name, attribute_name = exports[name]
    except KeyError:
        module_name = module_globals["__name__"]
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from None

    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    module_globals[name] = value
    return value


def lazy_dir(module_globals: dict[str, Any], exports: dict[str, LazyExport]) -> list[str]:
    """Return module globals together with names available through lazy lookup."""
    return sorted(module_globals.keys() | exports.keys())
