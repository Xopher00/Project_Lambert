"""
arch.py — Compiled architecture runtime classes.

ArchDef is the compiled result of a DSL source block. ArchInterpreter wraps
the algebra and coalgebra sides of an arch declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .path_engine import MorphismSpec, explain as _explain, trace as _trace
from .functor import Functor, Interpreter


# ---------------------------------------------------------------------------
# Internal compiled state
# ---------------------------------------------------------------------------

@dataclass
class _ArchData:
    """Internal: compiled algebra + coalgebra for a single arch declaration."""
    algebra_functor:      Functor  | None = None
    algebra_cell:         Callable | None = None
    coalgebra_functor:    Functor  | None = None
    coalgebra_cell:       Callable | None = None
    observer_convergence: Callable | None = None
    observer_loss:        Callable | None = None
    accumulate_legs:      dict | None = None  # morphism_name -> mode ('cat')


# ---------------------------------------------------------------------------
# ArchInterpreter
# ---------------------------------------------------------------------------

class ArchInterpreter:
    """Interpreter for an arch declaration — wraps algebra and coalgebra sides."""

    def __init__(self, name: str, algebra: Interpreter | None = None,
                 coalgebra: Interpreter | None = None,
                 convergence_fn: Callable | None = None,
                 loss_fn: Callable | None = None,
                 accumulate_legs: dict | None = None):
        self.name = name
        self._algebra = algebra
        self._coalgebra = coalgebra
        self._convergence_fn = convergence_fn
        self._loss_fn = loss_fn
        self._accumulate_legs = accumulate_legs

    def run_algebra(self, data, decompose: Callable):
        if self._algebra is None:
            raise ValueError(f"Arch '{self.name}' has no algebra declaration")
        return self._algebra.run_algebra(data, decompose)

    def run_coalgebra(self, state, token_iter=None, stop: Callable = None,
                      convergence_threshold: float = 1e-6):
        if self._coalgebra is None:
            raise ValueError(f"Arch '{self.name}' has no coalgebra declaration")
        effective_stop = stop
        if self._convergence_fn is not None:
            conv_fn = self._convergence_fn
            user_stop = stop
            prev = [None]
            thr = convergence_threshold
            def effective_stop(step, state, outputs,
                               _conv=conv_fn, _us=user_stop, _prev=prev,
                               _thr=thr):
                if _prev[0] is not None:
                    import numpy as _np
                    residual = _conv(state, _prev[0], 0.0)
                    converged = bool(_np.max(_np.abs(residual)) < _thr)
                else:
                    converged = False
                _prev[0] = state
                return converged or (_us is not None and _us(step, state, outputs))
        return self._coalgebra.run_coalgebra(
            state, token_iter=token_iter, stop=effective_stop,
            accumulate_legs=self._accumulate_legs,
        )


# ---------------------------------------------------------------------------
# ArchDef
# ---------------------------------------------------------------------------

@dataclass
class ArchDef:
    """
    Compiled result of a DSL source block.

    paths             : dict[str, Callable]  all named paths + individual morphisms + fans
    morphism_semiring : dict[str, str]       morphism name -> semiring group name
    """
    paths:             dict[str, Callable]
    morphism_semiring: dict[str, str]
    _morphism_specs:   dict[str, MorphismSpec]         = field(default_factory=dict, repr=False)
    _equations:        dict[str, str]                  = field(default_factory=dict, repr=False)
    _path_morphisms:   dict[str, list[str]]            = field(default_factory=dict, repr=False)
    _archs:            dict[str, _ArchData]            = field(default_factory=dict, repr=False)
    sort_defs:         dict                            = field(default_factory=dict, repr=False)

    def explain(self, name: str) -> str:
        if name in self._path_morphisms:
            return _explain(name, self._path_morphisms[name], self._equations)
        if name in self._morphism_specs:
            return _explain(name, [name], self._equations)
        if name in self.paths:
            return f"Path: {name} (fan or cross-semiring — no morphism expansion)"
        raise KeyError(f"Unknown morphism or path: {name!r}")

    def trace(self, name: str, x, y, temp: float = 0.0) -> list[tuple]:
        if name in self._path_morphisms:
            return _trace(self._path_morphisms[name], self.paths, self._equations, x, y, temp)
        if name in self._morphism_specs:
            return _trace([name], self.paths, self._equations, x, y, temp)
        raise KeyError(f"Unknown morphism or path: {name!r}")

    def loss(self, name: str, x, y, temp: float = 0.0):
        """Compute the observer loss for a named arch."""
        if name not in self._archs:
            raise KeyError(f"Unknown arch: {name!r}")
        ad = self._archs[name]
        if ad.observer_loss is None:
            raise ValueError(f"Arch '{name}' has no observer loss path declared")
        return ad.observer_loss(x, y, temp)

    def interpreter(self, name: str, params=None, temp: float = 0.0):
        """Create an interpreter for a named arch or legacy functor."""
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
            return ArchInterpreter(
                name, alg, coalg,
                convergence_fn=ad.observer_convergence,
                loss_fn=ad.observer_loss,
                accumulate_legs=ad.accumulate_legs,
            )

        raise KeyError(f"Unknown arch: {name!r}")
