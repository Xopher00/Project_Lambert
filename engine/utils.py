"""Shared utilities for the engine DSL pipeline."""
from __future__ import annotations
from typing import Any


def resolve(dotted: str, namespace: dict[str, Any]) -> Any:
    """Walk a dotted name through a namespace dict."""
    parts = dotted.split('.')
    obj = namespace.get(parts[0])
    if obj is None:
        raise NameError(f"Name {parts[0]!r} not found in provided namespace")
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj
