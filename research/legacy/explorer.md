> **Legacy documentation.** This document describes `lattice/explorer.py`, which has been moved to `legacy/lattice/explorer.py`. The patterns described here are now expressed through the engine DSL in `engine/`. See `research/theory.md` for the current architecture overview, and `research/engine/` for DSL documentation.

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

## Redundancies in the current implementation

**`_concept_fixpoint` override.** `CategoryExplorer` overrides `_concept_fixpoint`
from `Embed`, replacing the parent's alternating `Residuate` steps with a call to
`mha.retrieve`. The override is only ever called by `ConceptEmbed` inside
`explore_lattice` (phase 2). At that point it receives columns of `emb_new` as
seeds — but each column of `emb_new` is already a converged MHA extent vector from
phase 1. Re-running `mha.retrieve` on an already-converged state returns the same
state. The override adds no new concepts or categories; no entity ever enumerated in
phase 1 is reclassified.

The parent `Embed._concept_fixpoint` (alternating `Residuate` on `emb_new`) is the
algebraically correct operation for phase 2: it finds formal concepts of the
category-extent matrix, which is second-order FCA on a well-defined rectangular
context. Belohlavek (2000) Theorem 1 guarantees convergence in two steps for that
operation. The override loses that guarantee and does the same work more slowly.

**`learn=True` parameter on `explore`.** The parameter is declared in the method
signature but never read inside the method body. It is dead.

These are not blocking issues — the pipeline produces correct results despite them —
but they obscure what the code is doing and make phase 2 harder to reason about.

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

> Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.* — Theorem 8: completeness of
> the concept lattice guarantees all valid higher-order compositions are reachable.

---

## What the explorer does not do: the generation gap

After `explore_lattice` completes, the model has:

- A set of first-order categories (entity clusters closed under all relation types simultaneously).
- A second-order embedding `emb: (n_entities, k)` over the closed concept lattice.
- A Tucker core `EmbR: (k, k)` per head and one for the category matrix.

What it cannot do is **generate** — produce relation values or category memberships
for situations not directly observed. All queries go through MHA retrieval, which
returns entities already stored in the embedding. Nothing is inferred beyond recall.

Two things are structurally absent:

**1. `EmbR` is not wired into any query path.** The Tucker core `EmbR` compresses
each relation into concept space. Chaining `Join(q, EmbR₁)`, `Join(result, EmbR₂)`,
... across relation types would support multi-hop relational inference — traversing
concept-to-concept links to reach entity predictions not directly encoded in any
single head. The algebra is implemented (`Project`, `Join`); the query path that
uses it is not. See `embed.md` § `Project` and `Expand` for details.

**2. The learning rule is not implemented.** The FLBAM / IFAM construction rule
`W = Residuate(Y, X)` (Belohlavek 2000, eq. 2; Sussner & Valle 2006) builds a
relation matrix from stored `(input, output)` pattern pairs such that all patterns
are stable attractors. In Lambert's algebra, this is a single `Residuate` call.
Implementing it would allow the model to accumulate new knowledge — adding entities
or relation instances incrementally — and to synthesise `R` values for unseen
combinations. The operation exists; the layer that calls it with training pairs and
writes back into the embeddings does not. See `embed.md` § `The learning rule` for
details.

### The generation gap as a Kan extension problem

The two structural absences identified above have a unified categorical diagnosis: Lambert
implements the **right Kan** direction (Residuate, universal, retrieval) but not the
**left Kan** direction (Join-forward, existential, generation).

The concept lattice Mφ is closed under both operations — its completeness as a V-category
(Shen & Tang 2021, Theorem 6.2) guarantees that both Kan extensions exist and remain
within the lattice. The algebra is sound for both directions. What is absent is the query
path that uses the left Kan direction.

**`EmbR` not wired in** is the left Kan path through concept space. Calling `Join(q, EmbR)`
is a left Kan extension step: it asks "given concept q, what concepts are existentially
reachable via this relation?" Chaining it gives multi-hop reachability. The `Project` and
`Expand` operations implement the encoding and decoding steps around this chain; the
chain itself — the sequential Join through EmbR matrices — is what is missing from the
query path.

**The learning rule** (`W = Residuate(Y, X)`) is the right Kan extension applied to
pattern pairs: it finds the greatest weight matrix consistent with all stored
(input, output) pairs simultaneously. Writing it back into R is the construction step
that moves the model from read-only to writable.

The CQL literature frames this precisely: a schema migration that cannot express left
Kan extensions cannot generate new instances — it can only restrict existing ones. A
Lambert model that only uses Residuate is in the same position: it can find the tightest
concept above a query, but it cannot project forward to entities not observed during
construction.

**Reference:** Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases.
*Theory and Applications of Categories*, 32(16), 547–619.  cite{schultz2017}

**Reference:** Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via
quantale-enriched two-variable adjunctions. *Applied Categorical Structures*, 29, 823–858.
cite{shen2021}
