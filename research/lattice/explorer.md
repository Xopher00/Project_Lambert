# CategoryExplorer

`lattice/explorer.py`

`CategoryExplorer` sits above `MultiHeadAttention` and systematically discovers
the full structure of the concept lattice implied by the data. Where `Attention`
answers a single query, `CategoryExplorer` exhausts the space — querying every
entity in turn and collecting all stable attractors the system can reach.

The process runs in two phases. The first discovers first-order categories: entity
clusters that are simultaneously closed under all relation types. The second closes
the lattice over those categories, finding all higher-order structure implied by
their combinations.

---

## First phase: category discovery

A category in Lambert is the converged outer fixpoint state of
`MultiHeadAttention.retrieve` — an entity-space vector recording the graded
membership of each entity in a stable attractor across all relational heads
simultaneously.

The theoretical grounding is the fuzzy formal concept of Brito et al. (Definition
17): a pair `⟨O, A⟩` satisfying `O* = A` and `A∧ = O` — a simultaneous fixpoint
of the adjoint closure. In standard FCA this closure is computed over a single
relation. `CategoryExplorer` generalises it: the relevant context is not a single
`R` matrix but the conjunction of all relational spaces encoded in the MHA heads.
A category is an entity cluster that is simultaneously closed under every relation
type the system is aware of — a multi-relational formal concept.

The `explore` method realises this by querying the MHA with each entity in turn
and storing the converged extent as a new category if its key has not been seen.
Entities already assigned to a discovered category are skipped — the attractor
they belong to is already known.

> Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.* — Definition 15: fuzzy formal
> context as a triple `⟨O, A, I_f⟩`; Definition 17: fuzzy formal concept as a
> simultaneous fixpoint of `O*` and `A∧`. The category here is the multi-relational
> generalisation of that fixpoint.

---

## The closure operator at this layer

`CategoryExplorer` overrides `_concept_fixpoint` from `Embed`. The parent
alternates two `Residuate` steps — the `O*` / `A∧` adjoint pair applied to a
single relation matrix. This finds a concept closed within one relational space.

At this layer there is no single `R`. The relevant closure condition requires
simultaneous stability across all heads. The override replaces the Residuate
alternation with a call to `mha.retrieve`: each iteration queries all heads at
once and takes the outer fixpoint state as the new state. The fixpoint converges
when the entity set is stable under every head's retrieval simultaneously — the
correct closure operator for a multi-relational context.

This is not an approximation of the Residuate-based closure. It is the
appropriate operation for a richer formal context: a single `Residuate` step can
only satisfy the adjoint condition for one relation, while `mha.retrieve` satisfies
it for all relations jointly.

---

## Category keying: extent over intent

Each discovered category is identified by a quantised hash of the outer fixpoint
state — the extent vector. An earlier version keyed by per-head intent vectors
instead.

The intent key was unstable. Krotov & Hopfield (2021) establish that the attractor
reached by Hopfield dynamics depends on the energy minimum, which depends on
initialisation. Two queries that converge to the same entity cluster may follow
different trajectories, and the per-head concept-space states along those
trajectories differ even when the final entity set does not. The outer fixpoint
converges the entity-space state directly; the intents are derived quantities that
may vary across runs for the same structural category. In practice, intent keying
produced duplicates — the same category discovered twice with slightly different
intent vectors. Extent keying collapses them correctly.

The extent is the stable quantity by construction. It is what the outer loop
directly converges.

> Krotov, D. & Hopfield, J. (2021). Large Associative Memory Problem in
> Neurobiology and Machine Learning. *ICLR 2021.* — attractor convergence is stable
> in the energy minimum reached; intermediate states and paths to the attractor
> are not.

---

## Second phase: lattice closure

After `explore` discovers first-order categories, `explore_lattice` builds a new
matrix `emb_new: (n_entities, n_categories)` where each column is one category's
extent vector. It then runs `ConceptEmbed` on this matrix.

In `emb_new`, the "attributes" are the first-order categories themselves. A formal
concept of this matrix is a pair: a set of entities whose membership is jointly
closed under some combination of first-order categories. This is second-order FCA —
concepts of concepts. The discovered higher-order structure groups entities that are
coherent not within any individual relation, but across combinations of first-order
categories simultaneously.

Brito et al. (Theorem 8) establish that the concept lattice of any formal context
is complete — the infimum and supremum of any set of concepts exist and are
themselves concepts. This guarantees that `ConceptEmbed` on `emb_new` finds all
valid higher-order compositions: every entity grouping that has a closed
realisation in the space of first-order categories will appear as a concept column
in the returned embedding.

The theoretical status of the resulting embedding is still under investigation.
The structure of higher-order concept compositions at this level of abstraction —
what it means for a concept-of-concepts to be "correct" relative to the original
data — is an open question.

> Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.* — Theorem 8: completeness of
> the concept lattice guarantees all valid higher-order compositions are reachable.
