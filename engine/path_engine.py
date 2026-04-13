"""
path_engine.py — Equation-driven path compiler for relational tensor operations.

Core functions:
    compile_leg   — turn a LegSpec into a cached callable
    chain         — compose callables sequentially
    fan           — fan-out with merge
    check_sorts   — validate sort adjacency
    explain       — human-readable path description
    trace         — step-by-step execution with shapes

PathEngine class is retained for backward compatibility with code that
constructs engines directly (e.g. streamlined/composer.py).

Example
-------
::

    from engine.path_engine import LegSpec, compile_leg, chain

    specs = [
        LegSpec("realize",   join_fn, "j,ji->i", "j", "i"),
        LegSpec("propagate", join_fn, "i,ij->j", "i", "j"),
    ]
    legs = {s.name: compile_leg(s) for s in specs}
    attend = chain([legs["realize"], legs["propagate"]])
    result = attend(x, y, temp=0.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# LegSpec — declaration for a single path leg
# ---------------------------------------------------------------------------

@dataclass
class LegSpec:
    """Declaration for a single relational path leg.

    Parameters
    ----------
    name      : identifier used in path specs
    op        : semiring op; signature ``(compiled_eq, *args, temp=) → tensor``
    equation  : einsum string
    src_sort  : sort label consumed by this leg
    tgt_sort  : sort label produced by this leg
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


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compile_leg(spec: LegSpec) -> Callable:
    """Compile a LegSpec into a ``(x, y, temp) -> result`` callable."""
    ceq = spec.equation_compiler(spec.equation)
    op, tf = spec.op, spec.transform
    return lambda x, y, temp: op(ceq, *tf(x, y), temp=temp)


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


def fan(branches: dict[str, Callable], merge: Callable) -> Callable:
    """Fan-out: run all branches on the same ``(x, y, temp)``, merge results."""
    def prog(x, y, temp, branches=branches, merge=merge):
        return merge({name: fn(x, y, temp) for name, fn in branches.items()})
    return prog


def check_sorts(legs: dict[str, LegSpec], names: list[str]) -> str | None:
    """Return an error message if sorts don't compose, or None if valid."""
    for i in range(1, len(names)):
        prev_tgt = legs[names[i - 1]].tgt_sort
        curr_src = legs[names[i]].src_sort
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


def trace(names: list[str], leg_callables: dict[str, Callable],
          equations: dict[str, str], x, y, temp: float = 0.0) -> list[tuple]:
    """Execute step-by-step, returning ``(name, equation, shape, value)`` rows."""
    rows = [("input", None, getattr(x, "shape", None), x)]
    z = x
    for name in names:
        z = leg_callables[name](z, y, temp)
        rows.append((name, equations.get(name), getattr(z, "shape", None), z))
    return rows


# ---------------------------------------------------------------------------
# PathEngine — backward-compatible class wrapper
#
# Retained for streamlined/composer.py and direct-construction use cases.
# New code should use compile() from the DSL instead.
# ---------------------------------------------------------------------------

@runtime_checkable
class PathNormalizer(Protocol):
    def normalize(self, tokens: list[str]) -> list[str]: ...


class _IdentityNormalizer:
    def __init__(self, legs: dict[str, LegSpec]):
        self._legs = legs

    def normalize(self, tokens: list[str]) -> list[str]:
        if check_sorts(self._legs, tokens) is not None:
            return []
        return tokens


class _AliasNormalizer:
    def __init__(self, legs: dict[str, LegSpec], aliases: dict[str, list[str]]):
        self._legs    = legs
        self._aliases = aliases

    def normalize(self, tokens: list[str]) -> list[str]:
        current = tokens
        while True:
            changed, expanded = False, []
            for tok in current:
                if tok in self._aliases:
                    expanded.extend(self._aliases[tok])
                    changed = True
                else:
                    expanded.append(tok)
            current = expanded
            if not changed:
                break
        if check_sorts(self._legs, current) is not None:
            return []
        return current


def _parse_equation(eq: str) -> tuple[str, str]:
    """Infer (src_sort, tgt_sort) from an einsum string. Legacy helper."""
    lhs, rhs = eq.split("->")
    vec = next(p for p in lhs.split(",") if len(p) == 1)
    return vec, rhs


def _coerce_to_leg_specs(ops) -> list[LegSpec]:
    """Accept either a list of LegSpec or the legacy ``{name: (op, eq[, tf])}`` dict."""
    if isinstance(ops, list):
        return ops
    specs = []
    for name, entry in ops.items():
        op, eq_str = entry[0], entry[1]
        tf = entry[2] if len(entry) >= 3 else lambda x, y: (x, y)
        src, tgt = _parse_equation(eq_str)
        specs.append(LegSpec(name, op, eq_str, src, tgt, transform=tf))
    return specs


class PathEngine:
    """Backward-compatible class wrapper around the functional API.

    New code should use ``engine.compile(dsl_source, namespace)`` instead.
    """

    def __init__(self, ops, normalizer=None, aliases=None):
        leg_list = _coerce_to_leg_specs(ops)
        self._legs      = {s.name: s for s in leg_list}
        self._aliases   = frozenset(aliases or ())
        self._normalizer = normalizer or _IdentityNormalizer(self._legs)
        self._compiled  = {name: compile_leg(spec) for name, spec in self._legs.items()}
        self._cache: dict[tuple[str, ...], Callable] = {}
        self.equations: dict[str, str] = {n: s.equation for n, s in self._legs.items()}

    def _parse(self, spec: str) -> list[str]:
        tokens = spec.replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty path spec")
        for tok in tokens:
            if tok not in self._legs and tok not in self._aliases:
                valid = ", ".join(sorted(self._legs))
                raise ValueError(f"Unknown operation {tok!r}. Valid: {valid}")
        names = self._normalizer.normalize(tokens)
        if not names:
            err = check_sorts(self._legs, tokens)
            if err:
                raise TypeError(err)
            raise TypeError(f"Path {spec!r} reduces to bottom")
        return names

    def compile(self, spec: str) -> Callable:
        names = self._parse(spec)
        key = tuple(names)
        if key not in self._cache:
            self._cache[key] = chain([self._compiled[n] for n in names])
        return self._cache[key]

    def op(self, spec: str) -> Callable:
        return self.compile(spec)

    def run(self, spec: str, x, y, temp: float):
        return self.compile(spec)(x, y, temp)

    def explain(self, spec: str) -> str:
        names = self._parse(spec)
        return explain(spec, names, self.equations)

    def trace(self, spec: str, x, y, temp: float = 0.0) -> list[tuple]:
        names = self._parse(spec)
        return trace(names, self._compiled, self.equations, x, y, temp)
