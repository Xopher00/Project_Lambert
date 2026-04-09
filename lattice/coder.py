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

from dataclasses import dataclass
from core.fixpoint import FixpointIterator

@dataclass(frozen=True)
class Path:
    source: str
    steps: tuple[tuple[str, str], ...]

# ---------------------------------------------------------------------------
# BiCoder
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
    LEG_INFO = {
        ("s", "e"): ("Σ.encode", "Join(x, rel.T)"),
        ("s", "d"): ("Σ.decode", "Join(x, rel)"),
        ("p", "e"): ("Π.encode", "Residuate(rel, x)"),
        ("p", "d"): ("Π.decode", "Residuate(rel.T, x)"),
    }

    def __init__(self, legs, emb=None):
        self.legs = { 
            "s": {"e": legs[0], "d": legs[1]},
            "p": {"e": legs[2], "d": legs[3]}
        }
        self.emb = emb

    def bind(self, emb):
        return PathCoder(self.legs, emb=emb)
       
    def _rel(self, rel=None):
        rel = self.emb if rel is None else rel
        if rel is None:
            raise TypeError("BiCoder needs a relation: bind one or pass rel explicitly.")
        return rel
    
    def parse(self, spec: str) -> Path:
        source = "".join(spec.split())

        if len(source) % 2 != 0:
            raise ValueError(
                f"Invalid spec {spec!r}: expected pairs like se, pd, ..."
            )

        steps = []
        for i in range(0, len(source), 2):
            a = source[i]
            h = source[i + 1]

            if a not in self.ADJOINT_CODES:
                raise ValueError(f"Invalid adjoint code {a!r} at position {i}")
            if h not in self.HALF_CODES:
                raise ValueError(f"Invalid half code {h!r} at position {i+1}")

            steps.append((a, h))

        return Path(source=source, steps=tuple(steps))
    
    def leg(self, a, h, rel=None):
        rel = self._rel(rel)
        f = self.legs[a][h]
        return lambda x, temp: f(x, rel, temp)
    
    def run(self, path: Path | str, x, temp, rel=None):
        if isinstance(path, str):
            path = self.parse(path)

        rel = self._rel(rel)
        z = x
        for a, h in path.steps:
            z = self.legs[a][h](z, rel, temp)
        return z
    
    def compile(self, spec: str) -> Path:
        return self.parse(spec)
    
    def op(self, path: Path | str, rel=None):
        """
        Return a callable (x, temp) -> y from a compiled path or spec string.
        """
        if isinstance(path, str):
            path = self.compile(path)

        rel = self._rel(rel)

        def step(x, temp):
            return self.run(path, x, temp, rel=rel)

        return step
     
    def flow(self, state0, path=None, step=None, rel=None, **kw):
        if step is None:
            if path is None:
                raise TypeError("flow needs either `path` or `step`.")
            step = self.op(path, rel=rel)
        return FixpointIterator(f=step, state0=state0, **kw)
    
    def explain(self, spec):
        path = self.parse(spec)
        lines = [
            f"Path: {path.source}",
            "Factorization: Σ.encode → Σ.decode → Δ → Π.encode → Π.decode",
        ]
        for i, (a, h) in enumerate(path.steps, 1):
            role, op = self.LEG_INFO[(a, h)]
            lines.append(f"{i}. {a}{h}  = {role:9s}  [{op}]")
        return "\n".join(lines)
    
    def trace(self, spec, x, temp=0.0, rel=None):
        rel = self._rel(rel)
        path = self.parse(spec)

        rows = []
        z = x
        rows.append(("input", None, getattr(z, "shape", None), z))

        for a, h in path.steps:
            role, op = self.LEG_INFO[(a, h)]
            z = self.legs[a][h](z, rel, temp)
            rows.append((f"{a}{h}", f"{role} [{op}]", getattr(z, "shape", None), z))

        return rows