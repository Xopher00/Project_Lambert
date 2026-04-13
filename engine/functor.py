"""
functor.py — Functor declaration DSL for Layer B.

Provides Case, Functor, and CoalgResult for declaring recursive data types
and driving the generic Interpreter in core/interpreter.py.
"""

from dataclasses import dataclass

_NO_OUTPUT = object()


@dataclass
class Case:
    name:      str
    recursive: int
    data:      int
    output:    int = 0

    def __post_init__(self):
        if self.recursive < 0:
            raise ValueError("recursive must be >= 0")
        if self.data < 0:
            raise ValueError("data must be >= 0")
        if self.output not in (0, 1):
            raise ValueError("output must be 0 or 1")


@dataclass
class Functor:
    cases: list

    def __post_init__(self):
        names = [c.name for c in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate case names in Functor")
        self._by_name = {c.name: c for c in self.cases}

    def __getitem__(self, name: str) -> Case:
        if name not in self._by_name:
            raise KeyError(f"Unknown case '{name}' for this Functor")
        return self._by_name[name]


@dataclass
class CoalgResult:
    case_name:   str
    payload:     list
    next_states: list
    output:      object = _NO_OUTPUT
