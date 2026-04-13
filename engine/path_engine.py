"""
path_engine.py — Equation-driven path compiler for relational tensor operations.

Architecture (declaration → validation → normalization → compilation):

    LegSpec          — explicit declaration of a single path leg
    PathValidator    — tokenize and check names/sorts (no rewriting)
    PathNormalizer   — protocol; extension point for custom normalizers
    IdentityNormalizer — validates sorts, returns tokens unchanged (default)
    PathCompiler     — turn a validated name list into a cached callable
    PathEngine       — thin orchestrator; public API unchanged

Example
-------
::

    from streamlined.relational_einsum import join_einsum_forward, residuate_einsum_forward
    from engine.path_engine import PathEngine, LegSpec

    legs = [
        LegSpec("realize",   join_einsum_forward,      "j,ji->i", "j", "i"),
        LegSpec("propagate", join_einsum_forward,      "i,ij->j", "i", "j"),
        LegSpec("abstract",  residuate_einsum_forward, "ij,i->j", "i", "j",
                transform=lambda x, y: (y, x)),
        LegSpec("support",   residuate_einsum_forward, "ji,j->i", "j", "i",
                transform=lambda x, y: (y.T, x)),
    ]

    engine = PathEngine(legs)
    f = engine.op("realize propagate realize")
    result = f(x, y, temp=0.0)

Backwards-compatible tuple format is still accepted::

    engine = PathEngine({
        "realize":   (join_einsum_forward, "j,ji->i"),
        "propagate": (join_einsum_forward, "i,ij->j"),
    })
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_sort_mismatch(legs: dict, tokens: list[str]) -> int | None:
    """Return the index of the first adjacency where sorts do not compose, or None."""
    for i in range(1, len(tokens)):
        if legs[tokens[i - 1]].tgt_sort != legs[tokens[i]].src_sort:
            return i
    return None


# ---------------------------------------------------------------------------
# LegSpec — explicit leg declaration
# ---------------------------------------------------------------------------

@dataclass
class LegSpec:
    """Declaration for a single relational path leg.

    Parameters
    ----------
    name      : identifier used in path specs
    op        : semiring op; signature ``(compiled_eq, *args, temp=) → tensor``
    equation  : einsum string, used for compilation and explain/trace output
    src_sort  : sort label consumed by this leg
    tgt_sort  : sort label produced by this leg
    equation_compiler : callable that turns the equation string into whatever
                        ``op`` expects as its first argument
    transform         : reorders ``(x, y)`` before passing to ``op``; default identity
    """
    name:              str
    op:                Callable
    equation:          str
    src_sort:          str
    tgt_sort:          str
    equation_compiler: Callable = field(default=lambda eq: eq, repr=False)
    transform:         Callable = field(default=lambda x, y: (x, y), repr=False)


def _parse_equation(eq: str) -> tuple[str, str]:
    """Infer (src_sort, tgt_sort) from an einsum string.

    Used only in the backwards-compatible tuple shim; not on the critical path.

    >>> _parse_equation("j,ji->i")
    ('j', 'i')
    >>> _parse_equation("ij,i->j")
    ('i', 'j')
    """
    lhs, rhs = eq.split("->")
    vec = next(p for p in lhs.split(",") if len(p) == 1)
    return vec, rhs


def _coerce_to_leg_specs(ops: dict[str, tuple] | list[LegSpec]) -> list[LegSpec]:
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


# ---------------------------------------------------------------------------
# PathValidator — pure, stateless sort/name checking
# ---------------------------------------------------------------------------

class PathValidator:
    """Tokenize path specs and validate names and adjacent sorts.

    No rewriting is performed here. Raises ``ValueError`` for unknown names
    and ``TypeError`` for sort mismatches.
    """

    def __init__(self, legs: dict[str, LegSpec]) -> None:
        self._legs = legs

    def tokenize(self, spec: str) -> list[str]:
        tokens = spec.replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty path spec")
        return tokens

    def validate_names(self, tokens: list[str]) -> None:
        for tok in tokens:
            if tok not in self._legs:
                valid = ", ".join(sorted(self._legs))
                raise ValueError(f"Unknown operation {tok!r}. Valid: {valid}")

    def validate_sorts(self, tokens: list[str]) -> None:
        """Raise TypeError at the first adjacency where sorts do not compose."""
        i = _first_sort_mismatch(self._legs, tokens)
        if i is not None:
            prev_tgt = self._legs[tokens[i - 1]].tgt_sort
            curr_src = self._legs[tokens[i]].src_sort
            raise TypeError(
                f"Type mismatch at step {i}: "
                f"{tokens[i - 1]!r} outputs {prev_tgt!r} but "
                f"{tokens[i]!r} expects {curr_src!r}"
            )

    def validate(self, tokens: list[str]) -> None:
        self.validate_names(tokens)
        self.validate_sorts(tokens)


# ---------------------------------------------------------------------------
# PathNormalizer — protocol + two concrete implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class PathNormalizer(Protocol):
    """Normalize a validated token list.

    Returns the (possibly rewritten) token list, or ``[]`` if the path is
    a type error (invalid adjacency or reduces to bottom).
    """
    def normalize(self, tokens: list[str]) -> list[str]: ...


class IdentityNormalizer:
    """Validate sort adjacency, then return tokens unchanged.

    This is the default backend. It has no external dependencies and adds
    no overhead beyond a single linear scan.
    """

    def __init__(self, legs: dict[str, LegSpec]) -> None:
        self._legs = legs

    def normalize(self, tokens: list[str]) -> list[str]:
        if _first_sort_mismatch(self._legs, tokens) is not None:
            return []
        return tokens


# ---------------------------------------------------------------------------
# PathCompiler — compile + cache
# ---------------------------------------------------------------------------

class PathCompiler:
    """Build and cache callables from validated leg-name sequences.

    Cache key is ``tuple(names)`` (the normalized form), so two path specs
    that reduce to the same normal form share one compiled callable.
    """

    def __init__(self, legs: dict[str, LegSpec]) -> None:
        self._callables: dict[str, Callable] = {
            name: _make_leg_callable(spec)
            for name, spec in legs.items()
        }
        self._cache: dict[tuple[str, ...], Callable] = {}

    def compile(self, names: list[str]) -> Callable:
        key = tuple(names)
        if key not in self._cache:
            self._cache[key] = _chain(self._callables, names)
        return self._cache[key]

    def run_leg(self, name: str, x, y, temp):
        return self._callables[name](x, y, temp)


def _make_leg_callable(spec: LegSpec) -> Callable:
    ceq = spec.equation_compiler(spec.equation)
    op, tf = spec.op, spec.transform
    return lambda x, y, temp: op(ceq, *tf(x, y), temp=temp)


def _chain(callables: dict[str, Callable], names: list[str]) -> Callable:
    legs = [callables[n] for n in names]
    if len(legs) == 1:
        return legs[0]
    def prog(x, y, temp, legs=legs):
        z = x
        for f in legs:
            z = f(z, y, temp)
        return z
    return prog


# ---------------------------------------------------------------------------
# PathEngine — thin orchestrator
# ---------------------------------------------------------------------------

class PathEngine:
    """
    Equation-driven compiler for relational tensor paths.

    Parameters
    ----------
    ops : list[LegSpec] | dict[str, tuple]
        Leg declarations. Preferred form is a list of ``LegSpec`` objects.
        Legacy ``{name: (op, eq[, transform])}`` dict is also accepted.
    normalizer : PathNormalizer | None
        Normalization backend. Defaults to ``IdentityNormalizer`` (sort
        validation only, no rewriting). Pass a custom ``PathNormalizer``
        implementation to enable rewriting.
    """

    def __init__(
        self,
        ops: dict[str, tuple] | list[LegSpec],
        normalizer: PathNormalizer | None = None,
    ) -> None:
        leg_list = _coerce_to_leg_specs(ops)
        self._legs      = {s.name: s for s in leg_list}
        self._validator = PathValidator(self._legs)
        self._normalizer = normalizer if normalizer is not None else IdentityNormalizer(self._legs)
        self._compiler  = PathCompiler(self._legs)
        self._parse_cache: dict[str, list[str]] = {}

        # Expose equations for explain/trace (read-only view)
        self.equations: dict[str, str] = {n: s.equation for n, s in self._legs.items()}

    def _parse(self, spec: str) -> list[str]:
        if spec not in self._parse_cache:
            tokens = self._validator.tokenize(spec)
            self._validator.validate_names(tokens)
            names = self._normalizer.normalize(tokens)
            if not names:
                self._validator.validate_sorts(tokens)  # raises TypeError on mismatch
                raise TypeError(f"Path {spec!r} reduces to bottom")  # normalizer-specific bottom
            self._parse_cache[spec] = names
        return self._parse_cache[spec]

    def compile(self, spec: str) -> Callable:
        """Return a cached callable ``(x, y, temp)`` for a path spec."""
        return self._compiler.compile(self._parse(spec))

    def op(self, spec: str) -> Callable:
        """Alias for compile."""
        return self.compile(spec)

    def run(self, spec: str, x, y, temp: float):
        """Compile and immediately execute a path."""
        return self.compile(spec)(x, y, temp)

    def explain(self, spec: str) -> str:
        """Human-readable description of a path's normal form."""
        names = self._parse(spec)
        lines = [f"Path: {spec}", f"Normal form: {' '.join(names)}"]
        for i, name in enumerate(names, 1):
            lines.append(f"  {i}. {name}  [{self.equations[name]}]")
        return "\n".join(lines)

    def trace(self, spec: str, x, y, temp: float = 0.0) -> list[tuple]:
        """Execute step-by-step, returning (name, equation, shape, value) rows."""
        names = self._parse(spec)
        rows = [("input", None, getattr(x, "shape", None), x)]
        z = x
        for name in names:
            z = self._compiler.run_leg(name, z, y, temp)
            rows.append((name, self.equations[name], getattr(z, "shape", None), z))
        return rows
