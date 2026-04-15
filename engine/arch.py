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

Depends on: runtime.py (MorphismSpec, explain, trace),
            functor.py (Functor, Interpreter)

References
----------
Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
theory of all architectures. ICML 2024.  cite{gavranovic2024b}

Schultz, P., & Wisnesky, R. (2025). Algebraic data integration.
*Journal of Functional Programming*, 27.  cite{schultz2025}

Ramsauer, H. et al. (2021). Hopfield networks is all you need.
ICLR 2021.  cite{ramsauer2021}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .runtime import MorphismSpec, explain as _explain, trace as _trace
from .functor import Functor, Interpreter


# ---------------------------------------------------------------------------
# Internal compiled state
# ---------------------------------------------------------------------------

@dataclass
class _ArchData:
    """Internal: compiled endofunctor + algebra/coalgebra cells."""
    functor:              Functor  | None = None   # unified endofunctor F
    algebra_cell:         Callable | None = None
    coalgebra_cell:       Callable | None = None
    observer_convergence: Callable | None = None
    observer_loss:        Callable | None = None
    accumulate_legs:      dict | None = None  # morphism_name -> mode ('cat')
    iterate_groups:       dict | None = None  # group_name -> [case_names] (data flow order)
    iterate_base:         str | None = None   # base case name (leaf)
    iterate_epilogue:     list | None = None  # cases after iterate block
    state_type:           object | None = None  # Hydra TypeRecord for coalgebra state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_convergence_stop(conv_fn: Callable, user_stop: Callable | None,
                           threshold: float) -> Callable:
    """Return an effective_stop callable that halts on convergence or user signal.

    Convergence is measured by the max-abs residual of *conv_fn* between
    consecutive states.  The mutable cell ``prev`` is captured in the closure
    so callers need not manage it.

    References
    ----------
    Ramsauer, H. et al. (2021). Hopfield networks is all you need.
    ICLR 2021.  cite{ramsauer2021}
    """
    prev = [None]

    def effective_stop(step, state, outputs):
        import numpy as _np
        if prev[0] is not None:
            residual = conv_fn(state, prev[0], 0.0)
            converged = bool(_np.max(_np.abs(residual)) < threshold)
        else:
            converged = False
        prev[0] = state
        return converged or (user_stop is not None and user_stop(step, state, outputs))

    return effective_stop


# ---------------------------------------------------------------------------
# ArchInterpreter
# ---------------------------------------------------------------------------

class ArchInterpreter:
    """Interpreter for an arch declaration — wraps algebra and coalgebra sides.

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """

    def __init__(self, name: str, algebra: Interpreter | None = None,
                 coalgebra: Interpreter | None = None,
                 convergence_fn: Callable | None = None,
                 accumulate_legs: dict | None = None,
                 iterate_groups: dict | None = None,
                 iterate_base: str | None = None,
                 iterate_epilogue: list | None = None):
        self.name = name
        self._algebra = algebra
        self._coalgebra = coalgebra
        self._convergence_fn = convergence_fn
        self._accumulate_legs = accumulate_legs
        self._iterate_groups = iterate_groups
        self._iterate_base = iterate_base
        self._iterate_epilogue = iterate_epilogue or []

    def run_algebra(self, tree, decompose):
        """Fold a pre-built tree using the algebra.

        Parameters
        ----------
        tree       : root tree node
        decompose  : callable(node) -> (case_name, payload, children)
        """
        if self._algebra is None:
            raise ValueError(f"Arch '{self.name}' has no algebra declaration")
        return self._algebra.run_algebra(tree, decompose)

    def run_algebra_layers(self, x0, layers, extras=None):
        """Build a tree from iterate groups, then fold it with the algebra.

        Parameters
        ----------
        x0      : initial value (leaf payload)
        layers  : list of payloads for iterate cases
        extras  : dict merged into each layer payload, or None

        References
        ----------
        Gavranović, B. et al. (2024). Position: Categorical deep learning is an
        algebraic theory of all architectures. ICML 2024.  cite{gavranovic2024b}
        """
        if self._algebra is None:
            raise ValueError(f"Arch '{self.name}' has no algebra declaration")
        if self._iterate_groups is None:
            raise ValueError(
                f"Arch '{self.name}': layers= provided but no cases "
                f"have iterate= declared"
            )
        tree = self._build_tree(x0, layers, extras)
        return self._algebra.run_algebra(tree, lambda n: n)

    def _build_tree(self, x0, layers, extras):
        """Construct an algebra tree from iterate groups.

        Cases are declared in data flow order (attn before ffn). Tree nesting
        reverses this: the last declared iterate case is the outermost wrapper.

        References
        ----------
        Gavranović, B. et al. (2024). Position: Categorical deep learning is an
        algebraic theory of all architectures. ICML 2024.  cite{gavranovic2024b}
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
            effective_stop = _make_convergence_stop(
                self._convergence_fn, stop, convergence_threshold
            )
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

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}

    Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
    PODS 2007, pp. 31–40.  cite{green2007}
    """
    paths:             dict[str, Callable]
    morphism_semiring: dict[str, str]
    _morphism_specs:   dict[str, MorphismSpec]         = field(default_factory=dict, repr=False)
    _equations:        dict[str, str]                  = field(default_factory=dict, repr=False)
    _path_morphisms:   dict[str, list[str]]            = field(default_factory=dict, repr=False)
    _archs:            dict[str, _ArchData]            = field(default_factory=dict, repr=False)
    sort_defs:         dict                            = field(default_factory=dict, repr=False)
    _morphism_terms:   dict                            = field(default_factory=dict, repr=False)
    _path_terms:       dict                            = field(default_factory=dict, repr=False)
    _fan_terms:        dict                            = field(default_factory=dict, repr=False)
    _hydra_primitives: dict                            = field(default_factory=dict, repr=False)
    _arch_terms:       dict                            = field(default_factory=dict, repr=False)
    _arch_types:       dict                            = field(default_factory=dict, repr=False)

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
            if ad.functor is not None and ad.algebra_cell is not None:
                alg = Interpreter(ad.functor, ad.algebra_cell, merged, temp)
            coalg = None
            # Coalgebra uses the unified functor F
            if ad.functor is not None and ad.coalgebra_cell is not None:
                coalg = Interpreter(ad.functor, ad.coalgebra_cell, merged, temp)
            return ArchInterpreter(
                name, alg, coalg,
                convergence_fn=ad.observer_convergence,
                accumulate_legs=ad.accumulate_legs,
                iterate_groups=ad.iterate_groups,
                iterate_base=ad.iterate_base,
                iterate_epilogue=ad.iterate_epilogue,
            )

        raise KeyError(f"Unknown arch: {name!r}")

    @property
    def _all_terms(self) -> dict:
        """All Hydra term dicts merged."""
        merged = {}
        for d in (self._morphism_terms, self._path_terms, self._fan_terms, self._arch_terms):
            merged.update(d)
        return merged

    @property
    def module(self):
        """Assemble a Hydra Module from compiled architecture terms."""
        from engine.sorts import setup_hydra_path
        setup_hydra_path()
        from hydra.core import Name
        from hydra.packaging import Module, Namespace, DefinitionTerm, TermDefinition
        from hydra.dsl.python import Just, Nothing

        ns = Namespace("ua.engine.compiled")
        defs = tuple(
            DefinitionTerm(TermDefinition(
                name=Name(f"ua.engine.compiled.{name}"),
                term=term,
                type=Nothing(),
            ))
            for name, term in self._all_terms.items()
        )
        return Module(
            namespace=ns,
            definitions=defs,
            term_dependencies=(),
            type_dependencies=(Namespace("ua.engine"),),
            description=Just("Compiled architecture module"),
        )

    @property
    def graph(self):
        """Build a Hydra Graph with engine primitives and bound terms."""
        from engine.primitives import build_engine_graph
        from hydra.core import Name

        bound = {
            Name(f"ua.engine.compiled.{name}"): term
            for name, term in self._all_terms.items()
        }
        return build_engine_graph(self._hydra_primitives, bound_terms=bound)
