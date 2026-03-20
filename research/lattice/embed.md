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

> Brito, P. et al. *Fuzzy Formal Concept Analysis.* — adjoint maps `O*`, `A∧`;
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
> *Neural Network World*, 10(5). — Theorem 1: two-step convergence of the
> `O*`/`A∧` alternation; Theorem 2: stable points form the concept lattice.

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

Intended for multi-hop reasoning in concept space — replacing `R: (n, m)` with a
compressed `EmbR: (k, k)` for downstream operations. Both are currently **under
review** and not called by any active code. `Project` is dimensionally compatible
only when `n_attributes == n_entities`, which does not hold for typical rectangular
relation matrices.

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

> Ramsauer, H. et al. (2020). Hopfield Networks is All You Need. *arXiv:2008.07320.*

Called inside `Attention._step` on every iteration of every attention head's
fixpoint loop. Not intended to be called directly by external code.
