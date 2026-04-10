"""
Coder module. Defines the encode→decode skeleton that underlies both
Attend (Σ direction) and Recall (Π direction) in the CQL adjoint triple
Σ_F ⊣ Δ_F ⊣ Π_F.

In the categorical reading (Schultz et al. 2017 cite{schultz2017}), each direction
corresponds to a universal construction:
- Σ_F (attend_coder): left Kan extension — given a query in concept space, find
  the best matching entity pattern and return the concept-space reconstruction.
- Π_F (recall_coder): right Kan extension — given an entity vector, close it into
  the smallest formal concept whose extent contains it via alternating Residuate.

"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable

@dataclass(frozen=True, slots=True)
class Step:
    name: str            # canonical semantic name: realize, propagate, abstract, support
    swap: bool = False   # swap x and y arguments
    same: bool = False   # use x for both arguments (diagonal mode)
    trans: bool = False  # transpose y before passing

@dataclass(frozen=True, slots=True)
class Path:
    source: str
    steps: tuple[Step, ...]
    prog: Callable | None = None

# ---------------------------------------------------------------------------
# Algebraic type table
# ---------------------------------------------------------------------------

# Input and output space of each leg.  Used by _check_types() to enforce the
# E↔C alternation law and by _normalize() to identify same-pair roundtrips.
#
#   realize   (se) : C → E    (Σ.encode  — extent of a concept)
#   propagate (sd) : E → C    (Σ.decode  — image under Join)
#   abstract  (pe) : E → C    (Π.encode  — intent of an entity)
#   support   (pd) : C → E    (Π.decode  — preimage under Residuate)
#
# Same-adjoint-pair roundtrips are algebraically guaranteed idempotent:
#   (pd sd)² = pd sd,  (sd pd)² = sd pd   — join/propagate pair
#   (se pe)² = se pe,  (pe se)² = pe se   — residuate/abstract pair
LEG_TYPE = {
    "realize":   ("C", "E"),
    "propagate": ("E", "C"),
    "abstract":  ("E", "C"),
    "support":   ("C", "E"),
}

# Same-pair roundtrips eligible for algebraically sound idempotence folding.
# Key = (first_name, second_name); value = True means the pair is guaranteed.
_SAME_PAIR = {
    ("support",   "propagate"),  # pd sd  — join pair, closure on E
    ("propagate", "support"),    # sd pd  — join pair, projection on C
    ("realize",   "abstract"),   # se pe  — residuate pair, closure on C
    ("abstract",  "realize"),    # pe se  — residuate pair, projection on E
}

# Mixed-pair roundtrips that are empirically (but not algebraically) idempotent.
# Only folded when fold_empirical=True is passed to compile().
_MIXED_PAIR = {
    ("realize",   "propagate"),  # se sd  — Attend
    ("abstract",  "support"),    # pe pd  — Recall
}

# ---------------------------------------------------------------------------
# PathCoder
# ---------------------------------------------------------------------------

class PathCoder:
    """
    Compiler and executor for factorized adjoint paths in the Lambert core stack.

    PathCoder exposes the factorized form of the adjoint pipeline

        Σ.encode → Σ.decode → Δ → Π.encode → Π.decode

    as a small path language over four atomic legs:

        realize, propagate, abstract, support
        (also accepted as short codes: se, sd, pe, pd)

    Each leg is implemented by tensor-level relational operators such as `Join`
    and `Residuate`, supplied through a leg table. Path strings are parsed into
    reusable path objects and compiled to direct callables — all leg lookup and
    modifier logic is resolved once at compile time.

    References
    ----------
    Schultz, P., Spivak, D. I., Vasilakopoulou, C. & Wisnesky, R. (2025). cite{schultz2025}
    Algebraic Databases. §7: Σ_F ⊣ Δ_F ⊣ Π_F triple.  cite{schultz2017}
    Sanchez, E. (1976). Residuate adjoint structure.  cite{sanchez1976}
    """

    LEG_INFO = {
        "realize":   ("Σ.encode", "Join(x, y.T)"),
        "propagate": ("Σ.decode", "Join(x, y)"),
        "abstract":  ("Π.encode", "Residuate(y, x)"),
        "support":   ("Π.decode", "Residuate(y.T, x)"),
    }
    TOKEN_ALIASES = {
        # short codes → canonical name
        "se": "realize",
        "sd": "propagate",
        "pe": "abstract",
        "pd": "support",
        # semantic names (identity — so parse() is uniform)
        "realize":   "realize",
        "propagate": "propagate",
        "abstract":  "abstract",
        "support":   "support",
    }
    # accepted modifier tokens
    MODIFIERS = {"symmetry", "diagonal", "converse"}

    def __init__(self, legs, fold_empirical=False):
        self.legs = {
            "realize":   legs[0],   # se  (Σ.encode)
            "propagate": legs[1],   # sd  (Σ.decode)
            "abstract":  legs[2],   # pe  (Π.encode)
            "support":   legs[3],   # pd  (Π.decode)
        }
        self.fold_empirical = fold_empirical
        self._cache = {}

    @staticmethod
    def _check_types(steps):
        """
        Verify that consecutive steps alternate E↔C.

        Raises TypeError at the first boundary where the output space of one
        step does not match the input space of the next.  A single step is
        always valid.
        """
        for i in range(1, len(steps)):
            prev_out = LEG_TYPE[steps[i - 1].name][1]
            curr_in  = LEG_TYPE[steps[i].name][0]
            if prev_out != curr_in:
                raise TypeError(
                    f"Type mismatch at step {i}: "
                    f"{steps[i-1].name!r} outputs {prev_out!r} but "
                    f"{steps[i].name!r} expects {curr_in!r}"
                )

    @staticmethod
    def _fold(steps, pair):
        """
        Fold consecutive repeated step-pairs that appear in `pair`.

        Scans left-to-right; whenever the two steps at the tail of the
        emitted list exactly match the next two incoming steps (same name
        and same modifiers), the duplicate pair is dropped.

        Called with _SAME_PAIR for algebraically guaranteed idempotence
        (pd sd, sd pd, se pe, pe se) or with _MIXED_PAIR for empirical
        folding (se sd, pe pd) when the caller opts in.
        """
        out = list(steps)
        i = 2
        while i < len(out):
            a, b = out[i - 2], out[i - 1]
            if (i + 1 < len(out)
                    and (a.name, b.name) in pair
                    and out[i] == a and out[i + 1] == b):
                del out[i:i + 2]
            else:
                i += 1
        return tuple(out)

    def parse(self, spec: str) -> Path:
        tokens = spec.replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty path spec")
        steps = []
        for tok in tokens:
            parts = tok.split(":")
            head = parts[0].lower()
            mods = {p.lower() for p in parts[1:] if p}
            if head not in self.TOKEN_ALIASES:
                valid = ", ".join(sorted(self.TOKEN_ALIASES))
                raise ValueError(f"Unknown token {tok!r}. Valid: {valid}")
            unknown = mods - self.MODIFIERS
            if unknown:
                raise ValueError(
                    f"Unknown modifier(s) in {tok!r}: {', '.join(sorted(unknown))}. "
                    f"Valid: {', '.join(sorted(self.MODIFIERS))}"
                )
            steps.append(Step(
                name=self.TOKEN_ALIASES[head],
                swap="symmetry" in mods,
                same="diagonal" in mods,
                trans="converse" in mods,
            ))
        steps = tuple(steps)
        self._check_types(steps)
        steps = self._fold(steps, _SAME_PAIR)
        return Path(" ".join(tok.split(":")[0].lower() for tok in tokens), steps)

    def _compile_step(self, step: Step) -> Callable:
        """
        Resolve one Step into a callable (x, y, temp) with modifiers baked in.

        Leg lookup and all modifier logic are resolved once here so that the
        returned callable has no branching or dict dispatch at runtime.
        """
        f = self.legs[step.name]
        if step.swap:
            f0 = f; f = lambda x, y, t, f=f0: f(y, x, t)
        if step.same:
            f0 = f; f = lambda x, y, t, f=f0: f(x, x, t)
        if step.trans:
            f0 = f; f = lambda x, y, t, f=f0: f(x, y.T, t)
        return f

    def compile(self, spec: str | Path, fold_empirical=None) -> Path:
        path = self.parse(spec) if isinstance(spec, str) else spec
        do_fold = fold_empirical if fold_empirical is not None else self.fold_empirical
        cache_key = (path.source, do_fold)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if do_fold:
            path = replace(path, steps=self._fold(path.steps, _MIXED_PAIR))
        fns = [self._compile_step(s) for s in path.steps]
        if len(fns) == 1:
            prog = fns[0]
        else:
            def prog(x, y, temp, fns=fns):
                z = x
                for f in fns:
                    z = f(z, y, temp)
                return z
        compiled = replace(path, prog=prog)
        self._cache[cache_key] = compiled
        return compiled

    def op(self, spec: str | Path) -> callable:
        """Return a compiled callable (x, y, temp) for a path spec."""
        return self.compile(spec).prog

    def run(self, path: Path | str, x, y, temp):
        if isinstance(path, str):
            path = self.compile(path)
        return path.prog(x, y, temp)

    def explain(self, spec):
        path = self.parse(spec)
        lines = [
            f"Path: {path.source}",
            "Factorization: Σ.encode → Σ.decode → Δ → Π.encode → Π.decode",
        ]
        for i, step in enumerate(path.steps, 1):
            role, op_str = self.LEG_INFO[step.name]
            suffix = " (swapped x/y)" if step.swap else ""
            lines.append(f"{i}. {step.name} = {role:9s} [{op_str}]{suffix}")
        return "\n".join(lines)

    def trace(self, spec, x, y, temp=0.0):
        path = self.parse(spec)
        rows = [("input", None, getattr(x, "shape", None), x)]
        z = x
        for step in path.steps:
            role, op_str = self.LEG_INFO[step.name]
            f = self._compile_step(step)
            z = f(z, y, temp)
            rows.append((step.name, f"{role} [{op_str}]", getattr(z, "shape", None), z))
        return rows
