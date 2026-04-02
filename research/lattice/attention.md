# Attention

`lattice/attention.py`

The transformer attention mechanism and Hopfield memory retrieval turn out to be
the same mathematical operation viewed from different theoretical starting points.
Krotov & Hopfield (2021) show this formally: in §3.2 (Model B), they derive that
a dense associative memory with contrastive normalisation on its hidden layer, when
hidden neurons are integrated out in the fast-memory limit, produces an update rule
that is mathematically identical to dot-product attention. The connection is not an
analogy — it is a derivation.

Lambert's `Attention` class was constructed with this equivalence in mind, but
approached from the memory-retrieval side rather than the transformer side. The
update rule — score a query against stored patterns, normalise, reconstruct — is
implemented under max-min composition instead of dot-product arithmetic.
`Join(q, emb.T)` replaces the inner product; `SoftMax` over the result replaces
exponential normalisation; a second `Join` back through `emb` reconstructs the
output. The structure is identical to the Krotov & Hopfield derivation. What
changes is the semiring, and with it the notion of similarity: where dot-product
attention measures geometric proximity, max-min composition measures lattice
containment — Lambert retrieves the concept that most strongly subsumes the query,
not the one most correlated with it.

This step is `Attend`, the primitive at the core of every retrieval iteration.
`Attention` wraps it in a fixpoint loop (`_step`), seeded from a partial pattern
and iterated until the query stabilises on the nearest attractor in the concept
lattice.

> Krotov, D. & Hopfield, J. (2021). Large Associative Memory Problem in
> Neurobiology and Machine Learning. *ICLR 2021.* — §3.2 Model B: derivation of
> attention as the fast-memory limit of dense associative memory.

---

## The correction principle

At each iteration of `_step`, the raw output of `Attend` is corrected to stay
within what the embedding can justify. The greatest entity-space vector consistent
with the current concept activation is derived via `Residuate`, mapped back into
concept space via `Join`, and the query is clipped to whichever is lower — what it
claimed, or what the embedding can actually support. The query cannot converge to a
concept-space state with no valid realisation in entity space.

This is not a heuristic regulariser. It is the exact algebraic constraint that
keeps the fixpoint consistent with the stored concepts, grounded in the adjoint
structure of max-min composition established by Sanchez (1976).

> Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
> *Information and Control*, 30, 38–48. — adjoint structure `A ∘ B ≤ C iff B ≤ C ÷ A`;
> basis for the correction in `_step`.

---

## Conjunctive queries

When `_query` receives multiple entity indices, it takes the elementwise minimum
across their embedding rows — the meet in the concept lattice. Brito et al.
(Theorem 8) establish that the infimum of a set of concepts is the intersection of
their extents, which in fuzzy membership is the pointwise minimum. Retrieval from
this starting point finds entities satisfying all query constraints simultaneously.
The dual — supremum, pointwise maximum — is available via `EmbedSet` but not used
here.

> Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.* — Theorem 8, infimum and
> supremum of concepts in the complete lattice.

---

## Multi-head retrieval

The single-head mechanism reasons within one relational space. `MultiHeadAttention`
extends this by running one head per relation, each independently converging on its
own concept basin, then combining their outputs via an outer fixpoint in
`_outer_step`.

The theoretically correct combination is the lattice infimum across heads — the
elementwise minimum, keeping only entities satisfying all relations simultaneously.
This follows from Brito et al. (Theorem 8): the infimum of concepts is the
intersection of their extents. Bělohlávek (2000) Theorem 2 gives a direct proof:
each head's stable points form a complete lattice, and the intersection of complete
lattices closed under meet is also a complete lattice. The current implementation
uses hard `np.minimum` in `_outer_step`, which is the correct operation.

> Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.* — Theorem 8, infimum of concepts.

> Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory.
> *Information Sciences*, 128, 91–103. — Theorem 2: stable points of each head form a
> complete lattice; their intersection is the multi-relational concept lattice.
