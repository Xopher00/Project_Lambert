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

A `Coder` instance wraps a pair of callables `(encode, decode)` and provides
`apply(x, emb, temp)` = `decode(encode(x, emb, temp), emb, temp)`. Two canonical
instances are defined at module level:
- `attend_coder`: Σ direction (Join/Join)
- `recall_coder`: Π direction (Residuate/Residuate)

`BiCoder` holds a forward and inverse `Coder` and delegates by name.

Layering: core/tensor.py → lattice/coder.py → lattice/embed.py
This module does NOT import from lattice.embed (would create a circular import).
"""

from operator import attrgetter
from core.algebra import *
from core.tensor import Tensor
from core.fixpoint import FixpointIterator


class Coder(Tensor):
    """
    Named abstraction for the encode→decode step in the CQL adjoint triple.

    Wraps a pair of callables `(encode, decode)` and provides an `apply`
    method that runs the full encode→decode step.

    In the CQL adjoint triple (Σ_F ⊣ Δ_F ⊣ Π_F):
    - Σ direction: encode = Join(x, emb.T, temp), decode = Join(·, emb, temp)
    - Π direction: encode = Residuate(emb, x, temp), decode = Residuate(emb.T, ·, temp)

    The encode and decode callables are responsible for all reshaping.

    Parameters
    ----------
    encode : callable(x, emb, temp) -> intermediate, optional
        Encodes x into an intermediate representation through emb.
        If None, `apply` will raise TypeError.
    decode : callable(intermediate, emb, temp) -> output, optional
        Decodes the intermediate representation back through emb.
        If None, `apply` will raise TypeError.

    Examples
    --------
    >>> coder = Coder(
    ...     encode=lambda x, e, t: coder.Join(x.reshape(1, -1), e.T, t),
    ...     decode=lambda s, e, t: coder.Join(s, e, t).squeeze(),
    ... )
    >>> result = coder.apply(q, emb, temp=0.0)

    References
    ----------
    Schultz, P., Spivak, D. I., Vasilakopoulou, C. & Wisnesky, R. (2025).
    Algebraic Databases. §7: Σ_F ⊣ Δ_F ⊣ Π_F triple.  cite{schultz2017}
    Sanchez, E. (1976). Residuate adjoint structure.  cite{sanchez1976}
    """

    def __init__(self, encode=None, decode=None):
        super().__init__()
        self.encode = encode
        self.decode = decode

    def apply(self, x, emb, temp):
        """
        Run one encode→decode step through the embedding matrix.

            intermediate = self.encode(x, emb, temp)
            return self.decode(intermediate, emb, temp)

        Parameters
        ----------
        x : ndarray
            Input vector. Shape depends on the direction: (k,) for Σ, (n,) for Π.
        emb : ndarray, shape (n, k)
            Embedding matrix.
        temp : float
            Temperature passed to encode and decode.

        Returns
        -------
        ndarray
            Output vector. Shape determined by decode.

        Raises
        ------
        TypeError
            If self.encode or self.decode is None.
        """
        if self.encode is None or self.decode is None:
            raise TypeError(
                "Coder.apply requires both encode and decode to be set. "
                f"encode={self.encode!r}, decode={self.decode!r}"
            )
        intermediate = self.encode(x, emb, temp)
        return self.decode(intermediate, emb, temp)

# ---------------------------------------------------------------------------
# BiCoder
# ---------------------------------------------------------------------------

    
class BiCoder:
    """
    Bidirectional coder holding a forward and inverse Coder.

    Delegates `apply_forward` to `self.forward.apply` and `apply_inverse`
    to `self.inverse.apply`, providing a single object for both directions
    of the CQL adjoint triple.

    Parameters
    ----------
    forward : Coder
        The forward direction coder (e.g., attend_coder for Σ).
    inverse : Coder
        The inverse direction coder (e.g., recall_coder for Π).
    """
    """
    A bicoder bound to a specicing embedding matrix
    can run apply, encode, decode, and fixpoint iterations on either adjoint Pi or Sigma
    """

    def __init__(self, sigma: Coder, pi: Coder, emb=None):
        self.sigma = sigma
        self.pi = pi
        self.emb = emb

    def _get(self, adjoint):
        if adjoint in ("sigma", "Σ", "attend", self.sigma):
            return self.sigma
        if adjoint in ("pi", "Π", "recall", self.pi):
            return self.pi
        raise ValueError(f"Unknown adjoint: {adjoint!r}")
    
    def _require_emb(self):
        if self.emb is None:
            raise TypeError("BiCoder is unbound; call .bind(emb) or pass emb to __init__.")

    def bind(self, emb):
        return BiCoder(self.sigma, self.pi, emb)

    def apply(self, x, adjoint, temp):
        self._require_emb()
        return self._get(adjoint).apply(x, self.emb, temp)
    
    def encode(self, x, adjoint, temp):
        self._require_emb()
        return self._get(adjoint).encode(x, self.emb, temp)
    
    def decode(self, x, adjoint, temp):
        self._require_emb()
        return self._get(adjoint).decode(x, self.emb, temp)
    
    def roundtrip(self, space, encode):
        start = ("concept", "entity").index(space)
        coders = ("sigma", "pi")
        f1, f2 = map(
            attrgetter("encode" if encode else "decode"),
            self._get(coders[start:]) + self._get(coders[:start])
        )
        return lambda x, temp: f2(f1(x, self.emb, temp), self.emb, temp)

    def flow(self, state0, adjoint, step=None, **kw):
        self._require_emb()
        f = step if adjoint is None else (lambda x, t: self.apply(x, adjoint, t))
        return FixpointIterator(f=f, state0=state0, **kw)
    
class Adapter:
    def __init__(self, emb, adjoints=None, lossy=False):
        self.tensor = Tensor()
        self.bicoder = BiCoder(adjoints or self.define_adjoints())

        self.attend_step = lambda x, t: self.bicoder.apply(x, "sigma", t)
        self.recall_step = lambda x, t: self.bicoder.apply(x, "pi", t)

    def define_adjoints(self):
        Attend = Coder(
            encode=lambda x, e, t: self.Join(x.reshape(1,-1), e.T, t),   # (1,k) → (1,n)
            decode=lambda s, e, t: self.Join(s, e, t).squeeze(),          # (1,n) → (k,)
        )
        Recall = Coder(
            encode=lambda x, e, t: self.Residuate(e, x.reshape(-1,1), t).reshape(-1,1),  # (n,) → (k,1)
            decode=lambda s, e, t: self.Residuate(e.T, s, t).reshape(-1),                # (k,1) → (n,)
        )
        return Attend, Recall
    
    def attend_flow(self, state0, **kw):
        return self.bicoder.flow(state0=state0, adjoint="sigma", **kw)

    def recall_flow(self, state0, **kw):
        return self.bicoder.flow(state0=state0, adjoint="pi", **kw)

    def bind(self, emb):
        return self.bicoder.bind(emb)
