"""
Runtime closure factories for tensor operations.

Core functions:
    compile_morphism — turn a MorphismSpec into a cached callable
    chain            — compose callables sequentially
    fan              — fan-out with merge
    explain          — human-readable path description
    trace            — step-by-step execution with shapes

Example
-------
::

    from engine.runtime import MorphismSpec, compile_morphism, chain

    specs = [
        MorphismSpec("realize",   join_fn, "j,ji->i", "j", "i"),
        MorphismSpec("propagate", join_fn, "i,ij->j", "i", "j"),
    ]
    morphisms = {s.name: compile_morphism(s) for s in specs}
    attend = chain([morphisms["realize"], morphisms["propagate"]])
    result = attend(x, y, temp=0.0)

References
----------
Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
*Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
theory of all architectures. ICML 2024.  cite{gavranovic2024b}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as _np


# ---------------------------------------------------------------------------
# Backend — pluggable array operations
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    """Array operation backend. Default is numpy; swap for torch or others."""
    minimum:     Callable   # elementwise min of two arrays
    maximum:     Callable   # elementwise max of two arrays
    concatenate: Callable   # concatenate sequence of arrays along axis
    abs:         Callable   # elementwise absolute value
    max:         Callable   # reduce-max (returns scalar or array)
    add:         Callable   # elementwise x + y
    multiply:    Callable   # elementwise x * y
    sum:         Callable   # reduce-sum over axis
    min:         Callable   # reduce-min over axis


NUMPY_BACKEND = Backend(
    minimum=_np.minimum,
    maximum=_np.maximum,
    concatenate=_np.concatenate,
    abs=_np.abs,
    max=_np.max,
    add=_np.add,
    multiply=_np.multiply,
    sum=_np.sum,
    min=_np.min,
)


def _resolve(dotted: str, namespace: dict) -> object:
    """Walk a dotted name through a namespace dict."""
    parts = dotted.split('.')
    obj = namespace.get(parts[0])
    if obj is None:
        raise NameError(f"Name {parts[0]!r} not found in provided namespace")
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# CompiledMorphism — named callable wrapper for better tracebacks
# ---------------------------------------------------------------------------

class CompiledMorphism:
    __slots__ = ('_fn', 'name', 'equation')

    def __init__(self, fn: Callable, name: str, equation: str = ''):
        self._fn = fn
        self.name = name
        self.equation = equation

    def __call__(self, x, y):
        return self._fn(x, y)

    def __repr__(self):
        if self.equation:
            return f"Morphism({self.name!r}, eq={self.equation!r})"
        return f"Morphism({self.name!r})"


# ---------------------------------------------------------------------------
# MorphismSpec — declaration for a single path morphism
# ---------------------------------------------------------------------------

@dataclass
class MorphismSpec:
    """Declaration for a single relational path morphism.

    Parameters
    ----------
    name      : identifier used in path specs
    op        : semiring op; signature ``(compiled_eq, *args, temp=) -> tensor``
    equation  : einsum string
    src_sort  : sort label consumed by this morphism
    tgt_sort  : sort label produced by this morphism
    equation_compiler : turns the equation string into whatever ``op``
                        expects as its first argument
    transform         : reorders ``(x, y)`` before passing to ``op``

    References
    ----------
    Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
    *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """
    name:              str
    op:                Callable
    equation:          str
    src_sort:          str
    tgt_sort:          str
    equation_compiler: Callable = field(default=lambda eq: eq, repr=False)
    transform:         Callable = field(default=lambda x, y: (x, y), repr=False)
    arity:             str      = 'binary'  # 'binary' | 'unary' | 'pointwise' | 'ternary'
    accumulate:        str | None = None    # 'cat' | None — accumulation mode
    accumulate_fields: list[str] | None = None  # field names for field-level accumulate
    src_type:          object | None = None  # hydra.core.Type, when declared
    tgt_type:          object | None = None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compile_morphism(spec: MorphismSpec) -> CompiledMorphism:
    """Compile a MorphismSpec into a ``(x, y) -> result`` callable."""
    compiled_eq = spec.equation_compiler(spec.equation)
    op, transform_fn = spec.op, spec.transform
    if spec.arity == 'unary':
        fn = lambda x, _: op(x)
    elif spec.arity == 'pointwise':
        fn = lambda x, y: op(compiled_eq, x, y)
    elif spec.arity == 'ternary':
        fn = lambda x, y: op(compiled_eq, x, y[0], y[1])
    else:
        fn = lambda x, y: op(compiled_eq, *transform_fn(x, y))
    return CompiledMorphism(fn, name=spec.name, equation=spec.equation)


def chain(callables: list[Callable]) -> Callable:
    """Chain callables sequentially: each output feeds the next as ``x``.

    Implements V-functor composition gf: X → Z where each callable is a
    V-functor and the chain is their sequential composition.

    References
    ----------
    Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
    *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
    """
    if len(callables) == 1:
        return callables[0]
    names = [getattr(c, 'name', repr(c)) for c in callables]
    def prog(x, y, fns=callables):
        z = x
        for f in fns:
            z = f(z, y)
        return z
    return CompiledMorphism(prog, name=f"chain({' >> '.join(names)})")


def chain_with_augments(steps: list[tuple[str, Callable]]) -> Callable:
    """Chain steps where 'augment' steps merge into y instead of transforming x.

    Augment steps act as right Kan extensions: they enrich the relational context
    without modifying the primary signal.

    References
    ----------
    Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via
    quantale-enriched two-variable adjunctions. *Applied Categorical Structures*,
    29, 823–858.  cite{shen2021}
    """
    names = [getattr(fn, 'name', repr(fn)) for _, fn in steps]
    def prog(x, y, _steps=steps):
        z = x
        y_cur = y
        for kind, fn in _steps:
            if kind == 'augment':
                aug = fn(z, y_cur)
                y_cur = {**y_cur, **aug}
            else:
                z = fn(z, y_cur)
        return z
    return CompiledMorphism(prog, name=f"chain({' >> '.join(names)})")


def fan(branches: dict[str, Callable], merge: Callable) -> Callable:
    """Fan-out: run all branches on the same ``(x, y, temp)``, merge results.

    Implements the V-category product X × Y.

    References
    ----------
    Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
    *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
    """
    branch_names = list(branches.keys())
    def prog(x, y, branches=branches, merge=merge):
        return merge({name: fn(x, y) for name, fn in branches.items()})
    return CompiledMorphism(prog, name=f"fan({' & '.join(branch_names)})")



def explain(path_name: str, names: list[str],
            equations: dict[str, str]) -> str:
    """Human-readable description of a path."""
    lines = [f"Path: {path_name}", f"Normal form: {' '.join(names)}"]
    for i, name in enumerate(names, 1):
        lines.append(f"  {i}. {name}  [{equations.get(name, '?')}]")
    return "\n".join(lines)


def trace(names: list[str], morphism_callables: dict[str, Callable],
          equations: dict[str, str], x, y) -> list[tuple]:
    """Execute step-by-step, returning ``(name, equation, shape, value)`` rows."""
    rows = [("input", None, getattr(x, "shape", None), x)]
    z = x
    for name in names:
        z = morphism_callables[name](z, y)
        rows.append((name, equations.get(name), getattr(z, "shape", None), z))
    return rows


