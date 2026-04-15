> **Legacy documentation.** This document describes `lattice/attention.py`, which has been moved to `legacy/lattice/attention.py`. The patterns described here are now expressed through the engine DSL in `engine/`. See `research/theory.md` for the current architecture overview, and `research/engine/` for DSL documentation.

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

---

## Attend/Recall duality — architectural invariant

Every operation in the attention stack is an instance of one of two dual directions
inherited from the adjoint structure of max-min composition. Naming this duality
makes the asymmetry between what the model currently does and what it could do
explicit.

**Encode → decode (the forward / Σ direction).** `Attend` pushes a query forward
through the embedding:

```
q → Join(q, emb.T) → Join(scores, emb)
```

The first `Join` scores the query against every stored entity; the second
reconstructs a new entity-space vector as a max-min combination of the patterns
that scored highest. This is the left-adjoint direction — the existential,
generative direction. In the CQL data-migration formalism of Schultz & Wisnesky
(2025), this corresponds to the left pushforward functor Σ_F, which maps instances
forward along a schema morphism by constructing new instances via a coend. Σ is the
creative half of the adjoint triple: it can produce outputs that go beyond what is
explicitly stored.

**Grounding (the correction / Δ direction).** The correction step in `_step`
pulls the raw `Attend` output back to the greatest entity-space vector the current
embedding can actually support, via `Residuate`. This is the middle functor
direction — the restrictive direction. In the CQL triple it corresponds to Δ_F
(the pullback functor), which restricts instances along a schema morphism without
adding content. Δ is the middle functor in Σ_F ⊣ Δ_F ⊣ Π_F: it is right adjoint
to Σ_F and left adjoint to Π_F. Π_F (the learning rule direction) is the right
adjoint. Residuate finds the greatest solution consistent with the stored relation;
the correction step clips the query to stay within that solution.

**Learning / construction (the right pushforward / Π direction).** The learning
rule `W = Residuate(Y, X)` (described in `embed.md`) is the right-adjoint extreme:
it finds the greatest weight matrix simultaneously consistent with all stored
pattern pairs. In the CQL triple this is Π_F (the right pushforward), which is
right adjoint to Δ_F. Π is universal-over-all-inputs; it builds a structure that
accommodates every stored observation as a necessary consequence.

Together these three directions form a Galois adjoint triple:

```
Σ_F ⊣ Δ_F ⊣ Π_F
```

that is, Σ_F is left adjoint to Δ_F, and Δ_F is left adjoint to Π_F. In Lambert's
algebra the same triple appears as:

```
Attend (forward Join)  ⊣  Residuate-correction (pullback)  ⊣  Learning rule (Residuate(Y,X))
```

The current implementation uses only the middle functor continuously (correction in
every `_step` iteration) and the right functor not at all (learning rule is absent).
The left functor, `Attend`, is used for retrieval but its output is immediately
corrected back by Δ, so the net effect is restriction rather than generation. The
model is structurally a Δ/Π machine. Σ is available at the algebraic level (Join is
implemented) but is not wired into any stand-alone generative query path.

This adjoint triple is the categorical home of the encode→decode pattern:
- **Encode:** `ConceptEmbed` maps the entity relation matrix R into concept space
  via the Galois adjunction O*/A∧ (Bělohlávek 2000, Theorems 1–2).
- **Decode via Δ (recall):** `Attend` followed by the `Residuate` correction retrieves
  the tightest existing concept above the query — a restriction, not a generation.
- **Decode via Σ (generate):** chaining `Join(q, EmbR)` steps through the Tucker
  core would produce concept-space predictions for entities not observed during
  construction — the Σ direction. This path is architecturally sound but not yet
  wired.

The Galois adjunction underlying fixpoint convergence (Ore 1944, via Bělohlávek
2000) is the two-step version of the same triple: O* and A∧ are mutually adjoint,
and their composition is idempotent because adjunctions compose. Shen & Tang (2021)
place this in the enriched categorical setting: the fixpoint lattice M_φ is the
complete V-category of fixed points of the Isbell adjunction induced by the relation
matrix, and both Kan extensions (left and right) exist within it.

> Schultz, P. & Wisnesky, R. (2025). Algebraic Data Integration. *arXiv:1503.03571v8.* —
> §4.2: Σ_F ⊣ Δ_F ⊣ Π_F as the three adjoint data migration functors induced by a
> schema mapping; Σ as left pushforward (coend / existential), Δ as pullback, Π as
> right pushforward (end / universal).

> Schultz, P., Spivak, D. I., Vasilakopoulou, C. & Wisnesky, R. (2017). Algebraic
> Databases. *Theory and Applications of Categories*, 32(16), 547–619.
> arXiv:1602.03501v3. — §7: Definition 7.1 (Δ_F as pullback), Proposition
> 7.3 (Π_F as right Kan extension right adjoint to Δ_F), Proposition 7.4 (Σ_F as
> left Kan extension left adjoint to Δ_F); Lemma 8.18: Σ_F ≅ Λ_{F̂} ⊣ Δ_F ≅ Λ_{F̃} ≅
> Γ_{F̃} ⊣ Π_F ≅ Γ_{F̃} establishing the full adjoint triple in the equipment Data.

> Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory.
> *Information Sciences*, 128, 91–103. — Theorems 1–2: two-step convergence and
> completeness of the concept lattice via the O*/A∧ Galois adjunction.

> Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via
> quantale-enriched two-variable adjunctions. *Applied Categorical Structures*,
> 29, 823–858. — Theorem 6.2: the fixpoint lattice M_φ is a complete V-category
> in which both left and right Kan extensions exist.
