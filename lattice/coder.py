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

from dataclasses import dataclass, replace

@dataclass(frozen=True, slots=True)
class Step:
    adj: str
    half: str
    swap: bool = False
    same: bool = False
    trans: bool = False

@dataclass(frozen=True, slots=True)
class Path:
    source: str
    steps: tuple[Step, ...]
    prog: callable | None = None

# ---------------------------------------------------------------------------
# PathCoder
# ---------------------------------------------------------------------------
    
class PathCoder:
    """
    Compiler and executor for factorized adjoint paths in the Lambert core stack.

    PathCoder exposes the factorized form of the adjoint pipeline

        Σ.encode → Σ.decode → Δ → Π.encode → Π.decode

    as a small path language over four atomic legs:

        se, sd, pe, pd

    Each leg is implemented by tensor-level relational operators such as `Join`
    and `Residuate`, supplied through a leg table. Path strings are parsed into
    reusable path objects and can then be executed directly or wrapped as
    callables for fixpoint iteration.
    
    References
    ----------
    Schultz, P., Spivak, D. I., Vasilakopoulou, C. & Wisnesky, R. (2025). cite{schultz2025}
    Algebraic Databases. §7: Σ_F ⊣ Δ_F ⊣ Π_F triple.  cite{schultz2017}
    Sanchez, E. (1976). Residuate adjoint structure.  cite{sanchez1976}
    """

    ADJOINT_CODES = {"s", "p"}
    HALF_CODES = {"e", "d"}
    MODE_CODES = {"x", "m"}
    TRANSPOSE_CODES = {"t"}
    LEG_INFO = {
        ("s", "e"): ("Σ.encode", "Join(x, y.T)"),
        ("s", "d"): ("Σ.decode", "Join(x, y)"),
        ("p", "e"): ("Π.encode", "Residuate(y, x)"),
        ("p", "d"): ("Π.decode", "Residuate(y.T, x)"),
    }
    TOKEN_ALIASES = {
        "sd": ("s", "d"),
        "pd": ("p", "d"),
        "pe": ("p", "e"),
        "se": ("s", "e"),

        "propagate": ("s", "d"),
        "support": ("p", "d"),
        "abstract": ("p", "e"),
        "realize": ("s", "e"),
    }
    MODIFIER_ALIASES = {
        "symmetry": "swap",
        "diagonal": "same",
        "converse": "trans",
    }

    def __init__(self, legs):
        self.legs = { 
            "s": {"e": legs[0], "d": legs[1]},
            "p": {"e": legs[2], "d": legs[3]}
        }
        self._cache = {}
    
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
                raise ValueError(f"Invalid token {tok!r}. Valid step names: {valid}")
            unknown_mods = mods - set(self.MODIFIER_ALIASES)
            if unknown_mods:
                valid_mods = ", ".join(sorted(self.MODIFIER_ALIASES))
                bad = ", ".join(sorted(unknown_mods))
                raise ValueError(
                    f"Invalid modifier(s) in {tok!r}: {bad}. Valid modifiers: {valid_mods}"
                )
            a, h = self.TOKEN_ALIASES[head]
            steps.append(
                Step(
                    adj=a,
                    half=h,
                    swap="swap" in mods,
                    same="same" in mods,
                    trans="trans" in mods,
                )
            )
        return Path(" ".join(tokens), tuple(steps))
    
    def leg(self, step, x, y, temp):
        a, h = step.adj, step.half
        f = self.legs[a][h]
        x0, y0 = (y, x) if step.swap else (x, y)
        if step.same: y0 = x0
        if step.trans: y0 = y0.T
        return f(x0, y0, temp)
    
    def compile(self, spec: str | Path) -> Path:
        path = self.parse(spec) if isinstance(spec, str) else spec
        cached = self._cache.get(path.source)
        if cached is not None:
            return cached
        def prog(x, y, temp):
            z = x
            for step in path.steps:
                z = self.leg(step, z, y, temp)
            return z
        compiled = replace(path, prog=prog)
        self._cache[path.source] = compiled
        return compiled
    
    def op(self, path: Path):
        """
        Return a callable (x, y, temp) from a compiled path or spec string.
        """
        if isinstance(path, str) or path.prog is None:
            path = self.compile(path)
        prog = path.prog
        def step(x, y, temp):
            return prog(x, y, temp)
        return step
    
    def run(self, path: Path | str, x, y, temp):
        if isinstance(path, str):
            path = self.compile(path)
        if path.prog is not None:
            return path.prog(x, y, temp)
        z = x
        for step in path.steps:
            z = self.leg(step, z, y, temp)
        return z
       
    def explain(self, spec):
        path = self.parse(spec)
        lines = [
            f"Path: {path.source}",
            "Factorization: Σ.encode → Σ.decode → Δ → Π.encode → Π.decode",
        ]
        for i, step in enumerate(path.steps, 1):
            role, op = self.LEG_INFO[(step.adj, step.half)]
            suffix = " (swapped x/y)" if step.swap else ""
            lines.append(f"{i}. {step.adj}{step.half}{'x' if step.swap else ''} = {role:9s} [{op}]{suffix}")
        return "\n".join(lines)
    
    def trace(self, spec, x, y, temp=0.0):
        path = self.parse(spec)
        rows = []
        z = x
        rows.append(("input", None, getattr(z, "shape", None), z))

        for step in path.steps:
            role, op = self.LEG_INFO[(step.adj, step.half)]
            label = f"{step.adj}{step.half}{'x' if step.swap else ''}"
            call = f"{role} [{'swap' if step.swap else 'normal'}: {op}]"
            z = self.leg(step, z, y, temp)
            rows.append((label, call, getattr(z, "shape", None), z))

        return rows