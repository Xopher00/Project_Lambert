"""
Compiled architecture definitions and their runtime interpreters.

ArchDef is the compiled result of a full DSL source block — it holds every
compiled path, morphism, fan, and arch declaration, plus introspection methods
(explain, trace, loss). ArchInterpreter wraps a single arch declaration's
algebra and coalgebra interpreters for convenient evaluation.

Key abstractions:
  _ArchData        — internal: compiled algebra + coalgebra for one arch declaration
  ArchInterpreter  — runtime wrapper with run_algebra and run_coalgebra methods
  ArchDef          — top-level compiled artifact with explain/trace/loss/interpreter API

Depends on: path_engine.py (MorphismSpec, explain, trace),
            functor.py (Functor, Interpreter)
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
    iterate_groups:       dict | None = None  # group_name -> [case_names] (data flow order)
    iterate_base:         str | None = None   # base case name (leaf)
    iterate_epilogue:     list | None = None  # cases after iterate block


# ---------------------------------------------------------------------------
# ArchInterpreter
# ---------------------------------------------------------------------------

class ArchInterpreter:
    """Interpreter for an arch declaration — wraps algebra and coalgebra sides."""

    def __init__(self, name: str, algebra: Interpreter | None = None,
                 coalgebra: Interpreter | None = None,
                 convergence_fn: Callable | None = None,
                 loss_fn: Callable | None = None,
                 accumulate_legs: dict | None = None,
                 iterate_groups: dict | None = None,
                 iterate_base: str | None = None,
                 iterate_epilogue: list | None = None):
        self.name = name
        self._algebra = algebra
        self._coalgebra = coalgebra
        self._convergence_fn = convergence_fn
        self._loss_fn = loss_fn
        self._accumulate_legs = accumulate_legs
        self._iterate_groups = iterate_groups
        self._iterate_base = iterate_base
        self._iterate_epilogue = iterate_epilogue or []

    def run_algebra(self, data, decompose=None, layers=None, extras=None):
        """Run the algebra fold.

        Two calling conventions:
          run_algebra(tree, decompose)     — fold a pre-built tree (original)
          run_algebra(x0, layers=[...])    — build tree from iterate groups, then fold

        Parameters
        ----------
        data       : tree node (old) or initial value x0 (new)
        decompose  : callable(node) -> (case_name, payload, children), or None
        layers     : list of payloads for iterate cases, or None
        extras     : dict merged into each layer payload, or None
        """
        if self._algebra is None:
            raise ValueError(f"Arch '{self.name}' has no algebra declaration")
        if layers is not None:
            if self._iterate_groups is None:
                raise ValueError(
                    f"Arch '{self.name}': layers= provided but no cases "
                    f"have iterate= declared"
                )
            tree = self._build_tree(data, layers, extras)
            return self._algebra.run_algebra(tree, lambda n: n)
        if decompose is None:
            raise ValueError(
                f"Arch '{self.name}': must provide either decompose= or layers="
            )
        return self._algebra.run_algebra(data, decompose)

    def _build_tree(self, x0, layers, extras):
        """Construct an algebra tree from iterate groups.

        Cases are declared in data flow order (attn before ffn). Tree nesting
        reverses this: the last declared iterate case is the outermost wrapper.
        """
        if self._iterate_base is None:
            raise ValueError(
                f"Arch '{self.name}': iterate requires a leaf case (recursive=0)"
            )
        node = (self._iterate_base, [x0], [])
        for payload in layers:
            merged = {**payload, **(extras or {})} if extras else payload
            # Reverse declaration order for tree nesting:
            # declared attn, ffn -> tree nests as ffn(attn(prev))
            for group_name, case_names in self._iterate_groups.items():
                for case_name in reversed(case_names):
                    node = (case_name, [merged], [node])
        # Epilogue cases (non-iterate cases after the iterate block)
        for case_name in self._iterate_epilogue:
            node = (case_name, [None], [node])
        return node

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
                iterate_groups=ad.iterate_groups,
                iterate_base=ad.iterate_base,
                iterate_epilogue=ad.iterate_epilogue,
            )

        raise KeyError(f"Unknown arch: {name!r}")
