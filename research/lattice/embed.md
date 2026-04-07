# Embed

`lattice/embed.py` — the first layer of the lattice stack.

Inherits from `Tensor` and adds the ability to select a compact set of embedding
dimensions from a relation matrix. The central idea: instead of working with the
full relation `R: (n_entities, n_attributes)`, find a small set of columns of `R`
that together cover all entities. Each selected column corresponds to a **formal
concept** — a stable, mutually-closed pair of an entity set (extent) and an
attribute set (intent).

---

## Formal concepts

A **formal concept** of a relation `R: (n × m)` is a pair `(A, B)` where `A` is
the **extent** (entities) and `B` is the **intent** (attributes), and the two are
mutually closed: `A` is exactly the entities that possess all attributes in `B`,
and `B` is exactly the attributes shared by all entities in `A`.

Brito et al. formalise this as two adjoint maps:

```
O* = A  :  object set    → attribute set   (inf-of-implication over rows)
A∧ = O  :  attribute set → object set      (inf-of-implication over columns)
```

A formal concept is a fixpoint of their composition. Brito et al. prove that all
such fixpoints form a complete lattice — the **concept lattice** of `R` (Theorem 25).
This completeness is what guarantees `_concept_fixpoint` converges.

> Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.* — adjoint maps `O*`, `A∧`;
> concept lattice completeness, Theorem 25.

Bělohlávek (2000) provides a stronger and more specific convergence result directly
for this setting. His Fuzzy Logical BAM uses the same two update steps — equation
by equation, with Gödel implication as the residuum and `min` as conjunction,
which is exactly Lambert's algebra. **Theorem 1** proves the alternation is stable
and reaches its stable point in exactly two discrete time steps, via the idempotence
of Galois adjunctions (`A^{↑↓↑} = A^↑`, Ore 1944). A state ⟨A,B⟩ is stable if
and only if it is a formal concept. **Theorem 2** proves the full set of stable
points forms a complete lattice. Lambert's `max_iters=20` is conservative — the
algebra guarantees convergence in at most 2 steps.

> Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory.
> *Information Sciences*, 128, 91–103. — Theorem 1: two-step convergence of the
> `O*`/`A∧` alternation; Theorem 2: stable points form the concept lattice.

The set of all fixed points — the concept lattice Mφ — is characterised more
generally by Shen & Tang (2021) as the complete V-category of fixed points of
the Isbell adjunction induced by the relation matrix φ: A^op ⊗ B → V (Theorem
6.2). This is the categorical home of Lambert's embedding: each column of `emb`
is an element of Mφ, and the completeness of Mφ as a V-category is why
multi-head combination by elementwise minimum (lattice meet) produces a valid
concept rather than an arbitrary vector.

> Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via
> quantale-enriched two-variable adjunctions. *Applied Categorical Structures*,
> 29, 823–858. — Theorem 6.2: Mφ = Fix(φ↓φ↑) is a complete V-category;
> Proposition 5.3: Kan adjunctions recover Join and Residuate.

---

## `_concept_fixpoint`

Iterates the `O*` / `A∧` alternation to fixpoint, anchored at a seed column:

```
b     = Residuate(R_active, a)      # O*:  entity vector → attribute vector (intent)
a_new = Residuate(R_active.T, b)    # A∧:  attribute vector → entity vector (extent)
```

`a` is the entity vector (extent); `b` is the attribute vector (intent). Each
`Residuate` call computes the greatest solution to `A ∘ B = C` under max-min
composition — pointwise as `B[y,z] = min_i Implies(A[i,y], C[i,z])`. Sanchez
(1976) establishes the algebraic correctness of this operation.

> Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
> *Information and Control*, 30, 38–48. — Theorem 5; greatest solution to
> `A ∘ B = C` under max-min composition.

The function returns the converged extent, used in `ConceptEmbed` as a
deduplication key.

---

## `ConceptEmbed`

Implements coverage-based concept selection: iterates over columns of `R`, closes
each into a formal concept via `_concept_fixpoint`, and accepts it as a
representative dimension only if it covers at least one entity not yet covered.
This is the greedy coverage procedure of Trnecka & Vyjidacek (2020).

> Trnecka, M. & Vyjidacek, R. (2020). Revisiting the GreCon Algorithm for Boolean
> Matrix Factorization. *CLA 2020.* — coverage-based concept selection as matrix
> decomposition.

The final embedding is `emb = R[:, rep_cols]` — the raw representative columns of
`R`. The converged extent is used only for keying and then discarded; the embedding
preserves the original relational values from `R` for algebraic provenance.

**Single-entity special case.** When only one entity is active in a seed column,
the key is `(j,)` rather than the quantised fixpoint. Single-entity concepts with
identical attribute patterns converge to numerically identical fixpoints, which
would incorrectly merge distinct singleton entities into one concept column. The
column-index key prevents this.

**Coverage safety net.** A concept key can appear in `rep_cols` more than once if
a later occurrence covers new entities. No entity should be orphaned from the
embedding.

---

## `Project` and `Expand`

Tucker-style projection and reconstruction under max-min composition:

```
Project:  emb.T ∘ R ∘ emb   → (k, k)   relation in concept space
Expand:   emb   ∘ M ∘ emb.T → (n, n)   approximate reconstruction
```

`Project` is dimensionally compatible only when `n_attributes == n_entities`, which
does not hold for typical rectangular relation matrices. For the standard case
(`R: n_entities × n_features`), `emb: (n_entities, k)` and the Tucker core
`EmbR: (k, k)` is a concept-to-concept relation.

`EmbR` is computed by `ConceptEmbed` and stored in `self.heads[name]['EmbR']` after
`Lambert.run()`. It is currently **not wired into any query path** — `Query` goes
directly to MHA on entity-space embeddings. This is the primary structural reason
the model is retrievative rather than generative.

**What `EmbR` would enable.** A query in concept space — `q: (k,)` — traversed
through `EmbR` via `Join(q, EmbR)` produces a new concept-space vector representing
one relational hop. Chaining multiple `EmbR` matrices (one per relation type) would
support multi-hop inference: "concept C₁ relates to concept C₂ via relation R₁, and
C₂ relates to C₃ via R₂." Projecting back to entity space via `Expand` or
`Join(result, emb.T)` would produce entity-level predictions for entities not
explicitly observed in any single training example — generation rather than recall.

This is the intended purpose of `Project` and `Expand`. The algebra is sound; the
missing piece is wiring `EmbR` into the query path.

---

## The learning rule

The model currently reads relation matrices but never writes back to them. Belohlavek
(2000, Algorithm, eq. 2) gives the construction rule for an FLBAM weight matrix from
stored pattern pairs `{(Aᵖ, Bᵖ)}`:

```
I_ij = ∨_p  A^p(gᵢ) ⊗ B^p(mⱼ)
```

where `⊗` is the Gödel residuum — Lambert's `Residuate`. In Sussner & Valle (2006)
the same construction appears as:

```
W = Y ⊗ₙ Xᵀ  =  Residuate(Y, X)
```

This builds the weight matrix directly from a set of input/output pattern pairs
`(X, Y)`. In Lambert's terms: given a set of `(entity_vector, concept_vector)` pairs,
`Residuate(Y, X)` constructs the relation matrix `R` that stores all of them as
stable attractors.

The algebra for this is already present (`Residuate` is implemented and correct).
What is absent is the layer that calls it with training pairs and writes the result
back into the embeddings. Without this, the model cannot accumulate new knowledge
incrementally or synthesise relation values for entities not observed during
construction.

> Belohlavek, R. (2000). Fuzzy logical bidirectional associative memory.
> *Information Sciences*, 128, 91–103. — Algorithm, eq. 2: construction of I from
> stored patterns.

> Sussner, P., & Valle, M. E. (2006). Implicative fuzzy associative memories.
> *IEEE Transactions on Fuzzy Systems*, 14(6), 791–807. — eq. for W = Y ⊗ₙ Xᵀ;
> construction rule identical to Lambert's Residuate.

---

## Attend/Recall duality — relation to the CQL adjoint triple

The operations in this file occupy specific positions in the Galois adjoint triple
Σ_F ⊣ Δ_F ⊣ Π_F that governs all data migration in the CQL / algebraic-database
framework (Schultz & Wisnesky 2025; Schultz, Spivak, Vasilakopoulou & Wisnesky 2025).

**`ConceptEmbed` and `Project` (encode direction).** `ConceptEmbed` builds the
embedding `emb` by closing columns of `R` under the O*/A∧ adjunction — the
lattice-theoretic analogue of the pullback Δ. `Project` compresses the relation
matrix into concept space via `emb.T ∘ R ∘ emb`, producing `EmbR: (k, k)`. This
is the encoding step: it moves data from entity space into the compact concept
representation that the adjoint triple operates on.

**`Expand` and the unwired `EmbR` path (Σ direction).** The intended generative
query path — `Join(q, EmbR)` chained across relation types, then decoded back to
entity space via `Expand` — is the left-adjoint (Σ_F) direction. Σ constructs new
instances existentially: it produces entity-level predictions for situations not
explicitly observed during construction. This direction is algebraically sound (all
required operations exist) but is not currently wired into any query path. See
`attention.md § Attend/Recall duality` for the full categorical framing.

**The learning rule (Π direction).** `W = Residuate(Y, X)` is the right-adjoint
extreme Π_F: it finds the greatest weight matrix simultaneously consistent with all
stored pattern pairs, which in the CQL formalism is the right Kan extension (right
pushforward) along the schema mapping. Every stored `(entity_vector,
concept_vector)` pair is a necessary consequence of the constructed W. The
operation is algebraically present (Residuate is implemented) but the layer that
calls it with training pairs and writes back to R is absent.

The consequence of this asymmetry is the generation gap described in `explorer.md`:
a model that uses only Δ (restriction via correction) and lacks Σ (forward
projection via EmbR) and Π (learning rule via Residuate write-back) is a read-only
retrieval machine. All three operations exist in the algebra; only Δ is continuously
active at runtime.

> Schultz, P. & Wisnesky, R. (2025). Algebraic Data Integration. *arXiv:1503.03571v8.* —
> §4.2: the three adjoint data-migration functors Σ_F ⊣ Δ_F ⊣ Π_F; intuition: Δ as
> projection, Π as product/filter, Σ as union/merge.

> Schultz, P., Spivak, D. I., Vasilakopoulou, C. & Wisnesky, R. (2017). Algebraic
> Databases. *Theory and Applications of Categories*, 32(16), 547–619.
> arXiv:1602.03501v3. — §7, Propositions 7.3–7.4: Π_F as right Kan
> extension (right adjoint to Δ_F), Σ_F as left Kan extension (left adjoint to Δ_F);
> §8.18: full triple Σ_F ≅ Λ_{F̂} ⊣ Δ_F ⊣ Π_F ≅ Γ_{F̃} in the equipment Data.

---

## `EmbedSet`

Represents a group of entities as a single query vector:

```
EmbedSet({x₁, x₂, x₃}, emb, T) = SmoothMax(emb[[x₁, x₂, x₃], :], T, axis=0)
```

At `T=0` this is the least upper bound of the group in the concept lattice. At
`T>0` it is a smooth approximation slightly above that bound.

---

## `GramMatrix`

Entity-entity similarity via shared embedding dimensions:

```
GramMatrix(M) = M ∘ M.T
```

Entry `(x, x')` = `max_d min(M[x,d], M[x',d])` — the strongest concept dimension
that both `x` and `x'` participate in. Similarity is computed over rows. If
column-side similarity is needed, the caller transposes before passing:
`GramMatrix(M.T, temp)`.

---

## `Attend`

One step of Hopfield-style pattern retrieval:

```
q → q ∘ emb.T ∘ emb
```

The first Join scores the query against all entity rows in the embedding; the
second reconstructs the output as a max-min combination of those patterns.
Structurally identical to one update step of a modern Hopfield network,
implemented under max-min composition rather than softmax dot-product attention.

> Ramsauer, H. et al. (2020, revised 2021). Hopfield Networks is All You Need. *arXiv:2008.02217.*

Called inside `Attention._step` on every iteration of every attention head's
fixpoint loop. Not intended to be called directly by external code.
