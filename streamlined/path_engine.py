"""
path_engine.py — Equation-driven path compiler for relational tensor operations.

Users supply a single dict mapping operation names to ``(tensor_op, equation)``
or ``(tensor_op, equation, transform)`` tuples. The engine derives a simple
sort-transition system from the einsum equations, registers type-invalidity and
closed-loop redundancy rules in an egglog EGraph, and uses equality saturation
to normalize and validate path specs.

Example
-------
::

    from core.relational_einsum import join_einsum_forward, residuate_einsum_forward
    from lattice.path_engine import PathEngine

    engine = PathEngine({
        "realize":   (join_einsum_forward,      "j,ji->i"),
        "propagate": (join_einsum_forward,      "i,ij->j"),
        "abstract":  (residuate_einsum_forward, "ij,i->j", lambda x, y: (y, x)),
        "support":   (residuate_einsum_forward, "ji,j->i", lambda x, y: (y.T, x)),
    })
    f = engine.op("realize propagate realize")
    result = f(x, y, temp=0.0)
"""

from __future__ import annotations

import re
from typing import Callable

from egglog import EGraph, Expr, StringLike, rewrite, vars_
from torch_semiring_einsum import compile_equation


def _parse_equation(eq: str) -> tuple[str, str]:
    """Return (input_index, output_index) from an einsum string.

    >>> _parse_equation("j,ji->i")
    ('j', 'i')
    >>> _parse_equation("ij,i->j")
    ('i', 'j')
    """
    lhs, rhs = eq.split("->")
    vec = next(p for p in lhs.split(",") if len(p) == 1)
    return vec, rhs


class _PathExpr(Expr):
    @classmethod
    def atom(cls, name: StringLike) -> "_PathExpr": ...
    @classmethod
    def bottom(cls) -> "_PathExpr": ...
    def seq(self, other: "_PathExpr") -> "_PathExpr": ...


class PathEngine:
    """
    Equation-driven compiler for relational tensor paths.

    Parameters
    ----------
    ops : dict[str, tuple]
        ``name → (tensor_op, equation_string)`` or
        ``name → (tensor_op, equation_string, arg_transform)``.

        ``tensor_op``    : ``(compiled_eq, *args, temp=temp) → tensor``
        ``arg_transform``: ``(x, y) → tuple``, defaults to ``(x, y)``
    """

    def __init__(self, ops: dict[str, tuple]) -> None:
        self.equations: dict[str, str] = {}
        self._legs: dict[str, Callable] = {}

        for name, entry in ops.items():
            op, eq_str = entry[0], entry[1]
            tf = entry[2] if len(entry) >= 3 else lambda x, y: (x, y)
            self.equations[name] = eq_str
            ceq = compile_equation(eq_str)
            self._legs[name] = lambda x, y, temp, op=op, ceq=ceq, tf=tf: op(ceq, *tf(x, y), temp=temp)

        self._types = {n: _parse_equation(eq) for n, eq in self.equations.items()}
        self._egraph = EGraph()
        self._names: dict[str, list[str]] = {}
        self._cache: dict[str, Callable] = {}

        self._register_rules()

    def _invalid_pair_rules(self, atoms: dict[str, _PathExpr]) -> list:
        """Rules: x ; y -> bottom when output sort of x != input sort of y."""
        rules = []
        for x, (_, x_out) in self._types.items():
            x_atom = atoms[x]
            for y, (y_in, _) in self._types.items():
                if x_out != y_in:
                    rules.append(
                        rewrite(x_atom.seq(atoms[y])).to(_PathExpr.bottom())
                    )
        return rules

    def _loop_rules(self, atoms: dict[str, _PathExpr]) -> list:
        """
        Rules for cancelling typed round-trips.

        If x : a -> b and y : b -> a, then loop = x ; y is an a -> a round-trip.
        We register:
            (x ; y) ; z -> z    for any z with input sort a
            z ; (x ; y) -> z    for any z with output sort a
        """
        rules = []
        for x, (x_in, x_out) in self._types.items():
            x_atom = atoms[x]
            for y, (y_in, y_out) in self._types.items():
                if x_out != y_in or x_in != y_out:
                    continue
                loop = x_atom.seq(atoms[y])
                for z, (z_in, z_out) in self._types.items():
                    z_atom = atoms[z]
                    if z_in == x_in:
                        rules.append(rewrite(loop.seq(z_atom)).to(z_atom))
                    if z_out == x_in:
                        rules.append(rewrite(z_atom.seq(loop)).to(z_atom))
        return rules

    def _register_rules(self) -> None:
        """Register invalid-pair and closed-loop redundancy rules."""
        a, = vars_("a", _PathExpr)
        rules = [
            rewrite(a.seq(_PathExpr.bottom())).to(_PathExpr.bottom()),
            rewrite(_PathExpr.bottom().seq(a)).to(_PathExpr.bottom()),
        ]
        atoms = {name: _PathExpr.atom(name) for name in self._types}
        rules.extend(self._invalid_pair_rules(atoms))
        rules.extend(self._loop_rules(atoms))
        self._egraph.register(*rules)

    def _expr(self, names: list[str]) -> _PathExpr:
        expr = _PathExpr.atom(names[0])
        for n in names[1:]:
            expr = expr.seq(_PathExpr.atom(n))
        return expr

    def _normalize(self, names: list[str]) -> list[str]:
        """Run egglog saturation and return the extracted op-name sequence."""
        e = self._egraph.let(" ".join(names), self._expr(names))
        self._egraph.saturate()
        s = str(self._egraph.extract(e))
        if "_PathExpr.bottom()" in s:
            return []
        return re.findall(r'_PathExpr\.atom\("([^"]+)"\)', s)

    def _tokenize(self, spec: str) -> list[str]:
        """Split a path spec into operation tokens."""
        tokens = spec.replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty path spec")
        return tokens

    def _validate_tokens(self, tokens: list[str]) -> None:
        """Ensure every token names a known operation."""
        for tok in tokens:
            if tok not in self.equations:
                valid = ", ".join(sorted(self.equations))
                raise ValueError(f"Unknown operation {tok!r}. Valid: {valid}")

    def _raise_type_error(self, spec: str, tokens: list[str]) -> None:
        """Raise the most specific type error available for a failed path."""
        for i in range(1, len(tokens)):
            _, prev_out = self._types[tokens[i - 1]]
            curr_in, _ = self._types[tokens[i]]
            if prev_out != curr_in:
                raise TypeError(
                    f"Type mismatch at step {i}: "
                    f"{tokens[i - 1]!r} outputs {prev_out!r} but "
                    f"{tokens[i]!r} expects {curr_in!r}"
                )
        raise TypeError(f"Type error in path {spec!r}")

    def _parse(self, spec: str) -> list[str]:
        """Tokenize, validate, and normalize a path spec."""
        if spec in self._names:
            return self._names[spec]
        tokens = self._tokenize(spec)
        self._validate_tokens(tokens)
        names = self._normalize(tokens)
        if not names:
            self._raise_type_error(spec, tokens)
        self._names[spec] = names
        return names

    def compile(self, spec: str) -> Callable:
        """Return a cached compiled callable ``(x, y, temp)`` for a path spec."""
        if spec in self._cache:
            return self._cache[spec]
        names = self._parse(spec)
        legs = [self._legs[n] for n in names]
        if len(legs) == 1:
            prog = legs[0]
        else:
            def prog(x, y, temp, legs=legs):
                z = x
                for f in legs:
                    z = f(z, y, temp)
                return z
        self._cache[spec] = prog
        return prog

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
            z = self._legs[name](z, y, temp)
            rows.append((name, self.equations[name], getattr(z, "shape", None), z))
        return rows