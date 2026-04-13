"""
arch.py — Compiled architecture runtime classes.

ArchDef is the compiled result of a DSL source block. ArchInterpreter wraps
the algebra and coalgebra sides of an arch declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .path_engine import LegSpec, explain as _explain, trace as _trace
from .functor import Functor, Interpreter


# ---------------------------------------------------------------------------
# Internal compiled state
# ---------------------------------------------------------------------------

@dataclass
class _ArchData:
    """Internal: compiled algebra + coalgebra for a single arch declaration."""
    algebra_functor:   Functor  | None = None
    algebra_cell:      Callable | None = None
    coalgebra_functor: Functor  | None = None
    coalgebra_cell:    Callable | None = None


# ---------------------------------------------------------------------------
# ArchInterpreter
# ---------------------------------------------------------------------------

class ArchInterpreter:
    """Interpreter for an arch declaration — wraps algebra and coalgebra sides."""

    def __init__(self, name: str, algebra: Interpreter | None = None,
                 coalgebra: Interpreter | None = None):
        self.name = name
        self._algebra = algebra
        self._coalgebra = coalgebra

    def run_algebra(self, data, decompose: Callable):
        if self._algebra is None:
            raise ValueError(f"Arch '{self.name}' has no algebra declaration")
        return self._algebra.run_algebra(data, decompose)

    def run_coalgebra(self, state, token_iter=None, stop: Callable = None):
        if self._coalgebra is None:
            raise ValueError(f"Arch '{self.name}' has no coalgebra declaration")
        return self._coalgebra.run_coalgebra(state, token_iter=token_iter, stop=stop)


# ---------------------------------------------------------------------------
# ArchDef
# ---------------------------------------------------------------------------

@dataclass
class ArchDef:
    """
    Compiled result of a DSL source block.

    paths        : dict[str, Callable]    all named paths + individual legs + fans
    leg_semiring : dict[str, str]         leg name -> semiring group name
    functors     : dict[str, Functor]     declared functors (legacy)
    """
    paths:        dict[str, Callable]
    leg_semiring: dict[str, str]
    functors:     dict[str, Functor]         = field(default_factory=dict)
    _leg_specs:   dict[str, LegSpec]         = field(default_factory=dict, repr=False)
    _equations:   dict[str, str]             = field(default_factory=dict, repr=False)
    _path_legs:   dict[str, list[str]]       = field(default_factory=dict, repr=False)
    _cells:       dict[str, Callable]        = field(default_factory=dict, repr=False)
    _archs:       dict[str, _ArchData]       = field(default_factory=dict, repr=False)

    def explain(self, name: str) -> str:
        if name in self._path_legs:
            return _explain(name, self._path_legs[name], self._equations)
        if name in self._leg_specs:
            return _explain(name, [name], self._equations)
        if name in self.paths:
            return f"Path: {name} (fan or cross-semiring — no leg expansion)"
        raise KeyError(f"Unknown leg or path: {name!r}")

    def trace(self, name: str, x, y, temp: float = 0.0) -> list[tuple]:
        if name in self._path_legs:
            return _trace(self._path_legs[name], self.paths, self._equations, x, y, temp)
        if name in self._leg_specs:
            return _trace([name], self.paths, self._equations, x, y, temp)
        raise KeyError(f"Unknown leg or path: {name!r}")

    def interpreter(self, name: str, params=None, temp: float = 0.0):
        """Create an interpreter for a named arch or legacy functor.

        For arch declarations, returns an ``ArchInterpreter`` with
        ``.run_algebra()`` and/or ``.run_coalgebra()`` methods.

        For legacy functor declarations, returns a plain ``Interpreter``.

        ``params`` receives ``{'paths': self.paths}`` automatically so that
        cell functions can access compiled path callables.
        """
        merged = {'paths': self.paths}
        if params:
            merged.update(params)

        # Arch declarations (preferred)
        if name in self._archs:
            ad = self._archs[name]
            alg = None
            if ad.algebra_functor is not None and ad.algebra_cell is not None:
                alg = Interpreter(ad.algebra_functor, ad.algebra_cell, merged, temp)
            coalg = None
            if ad.coalgebra_functor is not None and ad.coalgebra_cell is not None:
                coalg = Interpreter(ad.coalgebra_functor, ad.coalgebra_cell, merged, temp)
            return ArchInterpreter(name, alg, coalg)

        # Legacy functor fallback
        if name not in self.functors:
            raise KeyError(f"Unknown arch or functor: {name!r}")
        if name not in self._cells:
            raise ValueError(
                f"Functor '{name}' has no cell binding. "
                f"Add 'cell = <dotted.name>' to the functor or per-case declarations."
            )
        return Interpreter(
            functor=self.functors[name],
            cell=self._cells[name],
            params=merged,
            temp=temp,
        )
