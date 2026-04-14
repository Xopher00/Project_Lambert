"""
path_engine.py — Equation-driven path compiler for relational tensor operations.

Core functions:
    compile_morphism — turn a MorphismSpec into a cached callable
    chain            — compose callables sequentially
    fan              — fan-out with merge
    check_sorts      — validate sort adjacency
    explain          — human-readable path description
    trace            — step-by-step execution with shapes

PathEngine class is retained for backward compatibility with code that
constructs engines directly (e.g. streamlined/composer.py).

Example
-------
::

    from engine.path_engine import MorphismSpec, compile_morphism, chain

    specs = [
        MorphismSpec("realize",   join_fn, "j,ji->i", "j", "i"),
        MorphismSpec("propagate", join_fn, "i,ij->j", "i", "j"),
    ]
    morphisms = {s.name: compile_morphism(s) for s in specs}
    attend = chain([morphisms["realize"], morphisms["propagate"]])
    result = attend(x, y, temp=0.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


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


LegSpec = MorphismSpec  # backward compat — used by streamlined/composer.py tests


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compile_morphism(spec: MorphismSpec) -> Callable:
    """Compile a MorphismSpec into a ``(x, y, temp) -> result`` callable."""
    compiled_eq = spec.equation_compiler(spec.equation)
    op, transform_fn = spec.op, spec.transform
    if spec.arity == 'unary':
        return lambda x, _, temp: op(compiled_eq, x, temp=temp)
    if spec.arity == 'pointwise':
        return lambda x, y, temp: op(compiled_eq, x, y, temp=temp)
    if spec.arity == 'ternary':
        return lambda x, y, temp: op(compiled_eq, x, y[0], y[1], temp=temp)
    return lambda x, y, temp: op(compiled_eq, *transform_fn(x, y), temp=temp)


def chain(callables: list[Callable]) -> Callable:
    """Chain callables sequentially: each output feeds the next as ``x``."""
    if len(callables) == 1:
        return callables[0]
    def prog(x, y, temp, fns=callables):
        z = x
        for f in fns:
            z = f(z, y, temp)
        return z
    return prog


def chain_with_augments(steps: list[tuple[str, Callable]]) -> Callable:
    """Chain steps where 'augment' steps merge into y instead of transforming x."""
    def prog(x, y, temp, _steps=steps):
        z = x
        y_cur = y
        for kind, fn in _steps:
            if kind == 'augment':
                aug = fn(z, y_cur, temp)
                y_cur = {**y_cur, **aug}
            else:
                z = fn(z, y_cur, temp)
        return z
    return prog


def fan(branches: dict[str, Callable], merge: Callable) -> Callable:
    """Fan-out: run all branches on the same ``(x, y, temp)``, merge results."""
    def prog(x, y, temp, branches=branches, merge=merge):
        return merge({name: fn(x, y, temp) for name, fn in branches.items()})
    return prog


def check_sorts(morphisms: dict[str, MorphismSpec], names: list[str]) -> str | None:
    """Return an error message if sorts don't compose, or None if valid."""
    for i in range(1, len(names)):
        prev_tgt = morphisms[names[i - 1]].tgt_sort
        curr_src = morphisms[names[i]].src_sort
        if prev_tgt != curr_src:
            return (
                f"Type mismatch at step {i}: "
                f"{names[i - 1]!r} outputs {prev_tgt!r} but "
                f"{names[i]!r} expects {curr_src!r}"
            )
    return None


def explain(path_name: str, names: list[str],
            equations: dict[str, str]) -> str:
    """Human-readable description of a path."""
    lines = [f"Path: {path_name}", f"Normal form: {' '.join(names)}"]
    for i, name in enumerate(names, 1):
        lines.append(f"  {i}. {name}  [{equations.get(name, '?')}]")
    return "\n".join(lines)


def trace(names: list[str], morphism_callables: dict[str, Callable],
          equations: dict[str, str], x, y, temp: float = 0.0) -> list[tuple]:
    """Execute step-by-step, returning ``(name, equation, shape, value)`` rows."""
    rows = [("input", None, getattr(x, "shape", None), x)]
    z = x
    for name in names:
        z = morphism_callables[name](z, y, temp)
        rows.append((name, equations.get(name), getattr(z, "shape", None), z))
    return rows


