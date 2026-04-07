# Project Lambert: Close the Generation Gap
## Product Requirements Document

---

## 1. What & Why

**Problem:**
Lambert implements only half of its own algebra. Every query passes through Residuate
(the right Kan extension, universal / retrieval direction). The left Kan extension
(Join-forward through EmbR, existential / generation direction) is algebraically present
but unconnected. The learning rule that writes new knowledge back into the relation
matrix is similarly implemented but never called. The result: Lambert can recall any
concept it was constructed from, but cannot infer beyond what it directly observed, and
cannot accumulate new knowledge incrementally.

This gap is not a design choice — it is diagnosed explicitly in the research notes. Both
`embed.md` and `explorer.md` name it as the primary structural absence. `theory.md`
identifies it as a Kan extension problem and describes the exact fix. The algebra for
both directions exists and is correct. What is missing is the wiring.

**Desired Outcome:**
After this PRD:
1. Multi-hop relational inference works: `Join(q, EmbR_1)`, `Join(result, EmbR_2)`, ...
   chained across named relation types, producing entity-level predictions for entities
   not directly observed in any single training example.
2. Incremental learning works: calling `learn(input_patterns, output_patterns)` on a
   Lambert model updates the relation matrix R via `Residuate(Y, X)` (Belohlavek 2000
   eq. 2 / Sussner & Valle 2006), storing the new patterns as stable attractors without
   disrupting existing ones.
3. Dead code and implementation inconsistencies that obscure the algebra are removed:
   the `_concept_fixpoint` override in `CategoryExplorer` and the `learn=True` dead
   parameter on `explore`.

**Justification:**
The theory grounding work over the past sessions established the formal basis for both
operations. The Kan framing in `theory.md` (right Kan = Residuate, left Kan = Join)
makes the two directions symmetrical by construction. The FLBAM / IFAM learning rule is
a single `Residuate` call — the implementation already exists. The cost of closing these
gaps now is low; the cost of building further features (multi-hop knowledge graphs,
incremental knowledge updates, generation rather than recall) on top of the current
half-implemented architecture is high. This is the highest-leverage structural step
remaining before Lambert can be applied to real knowledge graph workloads.

---

## 2. Correctness Contract

**Audience:**
Research and engineering work that builds on Lambert. The immediate audience is the
codebase itself: other modules that need to answer questions Lambert cannot currently
answer (multi-hop queries, novel entity predictions, incremental updates). The secondary
audience is anyone reading the code who needs the implementation to match the theory
documentation.

**Failure Definition:**
- Multi-hop query returns incorrect results (entities that should not be reachable, or
  missing entities that should be reachable via the chain).
- Incremental learning overwrites existing attractors rather than adding new ones.
- Dead code removal breaks any existing test or import.

**Danger Definition:**
- A learning call that silently corrupts the relation matrix rather than extending it
  (e.g., taking a global minimum rather than the correct Residuate-based max construction).
- A multi-hop query that traverses EmbR matrices across incompatible heads (wrong
  dimensional alignment), producing numerically plausible but semantically wrong results.

**Risk Tolerance:**
Confident wrong answer is worse than refusal. For multi-hop queries, if the chain
type-checks dimensionally but the semantics are unclear (e.g., crossing incompatible
relation types), the query should raise a descriptive error rather than return a result.

---

## 3. Context Loaded

- `research/theory.md` (Query semantics, Kan framing): Establishes that `Π_R(q) =
  Residuate(R, q)` is the right Kan extension and `Σ_R(q) = Join(q, R)` is the left
  Kan extension. Multi-hop left Kan chains are described exactly. Both directions are
  dual; the concept lattice is closed under both (Shen & Tang 2021 Theorem 6.2).

- `research/lattice/embed.md` (Project, Expand, learning rule): Identifies `EmbR` as
  computed but unconnected. Describes the learning rule `W = Residuate(Y, X)` from
  Belohlavek (2000) eq. 2 and Sussner & Valle (2006). States explicitly that the algebra
  for both features is present; the calling layer is not.

- `research/lattice/explorer.md` (generation gap, dead code): Names the `_concept_fixpoint`
  override and the `learn=True` dead parameter. Explains why the override adds no value
  (the input state is already converged). Describes the generation gap as a Kan extension
  problem and gives the CQL framing: a system that only uses the right Kan direction
  cannot generate new instances.

- `research/lattice/attention.md` (correction principle): The Residuate correction in
  `_step` keeps fixpoints within what the embedding can support. The same constraint
  applies in the multi-hop path — each hop must project back through the embedding before
  the next hop begins, or the intermediate state leaves the concept lattice.

- `research/core/tensor.md` (EmbR, Tucker projection, CQL adjoint triple): `Project`
  compresses a relation into concept space as `emb.T ∘ R ∘ emb`. The Tucker core
  `EmbR: (k, k)` is a concept-to-concept relation. `Join(q, EmbR)` is a left Kan step
  in concept space. The CQL table shows the three operations are Δ_F, Σ_F (Join), Π_F
  (Residuate) — every well-typed query is a composition of these.

- Shen & Tang (2021) `isbell-adjunctions-kan-adjunctions-quantales.pdf`: Confirms that
  Mφ = Fix(φ↓φ↑) is a complete V-category (Theorem 6.2), so both Kan extensions remain
  in the lattice. Proposition 5.3 identifies the Kan adjunctions as exactly Lambert's
  Join and Residuate pair.

- Belohlavek (2000) `fuzzy-logic-bidirectional-associative-mem.pdf`: Algorithm eq. 2
  gives the FLBAM weight construction rule directly. The construction is `I_ij =
  ∨_p A^p(g_i) ⊗ B^p(m_j)` — which in Lambert's algebra is `R = Join(A.T, B)` across
  all stored pattern pairs (taking the max over patterns, using the Gödel residuum for
  each pair). Theorem 6 proves perfect recall if the training set forms a consistent
  conceptual structure.

- Domingos (2025) `tensor-logic.pdf`: The Tucker decomposition reduces rank-3 relational
  tensors to a core and three factor matrices. Multi-hop queries in tensor logic are
  chains of einsums. The concept-space chain `Join(q, EmbR_1), Join(..., EmbR_2), ...`
  is the max-min semiring version of this.

- `tests/test_core.py` (13 tests, all passing): Establishes the baseline test coverage
  that must not regress.

- `model.py` / `query.py`: `EmbR` is stored in `self.heads[name]['EmbR']` after `run()`.
  `Query` currently only uses `mha.retrieve` (right Kan path). The multi-hop and
  learning paths will be added to `Query` and a new `Learner` class respectively.

---

## 4. Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Multi-hop query reachability | Not possible | 2-hop and 3-hop chains return correct entity sets on synthetic data | `pytest tests/test_multihop.py` |
| Incremental learning: attractor preservation | Not possible | Existing concepts survive a learn() call with new patterns | `pytest tests/test_learn.py` — verify old query results unchanged |
| Incremental learning: new attractor storage | Not possible | New patterns are retrievable after learn() | `pytest tests/test_learn.py` — verify new query returns new entities |
| Dead code removed | 2 items | 0 | `grep -n '_concept_fixpoint' lattice/explorer.py` returns no override; `grep -n 'learn=' lattice/explorer.py` returns no parameter |
| Existing tests | 13 passing | 13 passing | `python -m pytest tests/test_core.py -v` exits 0 |
| Citation check | 5 ok | 5+ ok | `python tools/check_citations.py` exits 0 |

---

## 5. User Stories

GIVEN a Lambert model with two named relation heads `EmbR_collab` and `EmbR_authored`,
WHEN `q.multihop(entity="Alice", chain=["collab", "authored"])` is called,
THEN the result contains entities that Alice's collaborators have authored work on,
  and the result would be empty if no such chain exists in the embedding.

GIVEN a Lambert model trained on an initial set of entity-relation triples,
WHEN `model.learn(new_entity_rows, new_attribute_cols)` is called with additional data,
THEN subsequent queries for new entities return results,
  AND queries for previously known entities return the same results as before.

GIVEN a researcher reading `lattice/explorer.py`,
WHEN they look at the `_concept_fixpoint` and `explore` method signatures,
THEN neither contains dead code or misleading overrides that contradict the research
  documentation.

---

## 6. Acceptance Criteria

- [ ] `Join(q, EmbR)` can be called on a single EmbR head, returning a (k,) concept-space
  vector representing one relational hop from query q.
- [ ] `multihop(q, chain_of_head_names)` chains multiple EmbR matrices sequentially and
  projects the result back to entity space, returning a ranked entity list.
- [ ] A 2-hop chain on synthetic data (3 entities A, B, C; A-related-to-B via R1,
  B-related-to-C via R2) returns C as reachable from A.
- [ ] `learn(X, Y)` calls `Residuate(Y, X)` to construct a weight matrix and merges it
  into R via elementwise max (Belohlavek 2000 eq. 2: join over stored patterns).
- [ ] After `learn(X_new, Y_new)`, `query(entities=[new_entity])` returns results.
- [ ] After `learn(X_new, Y_new)`, `query(entities=[old_entity])` returns identical
  results to before the learn call.
- [ ] `_concept_fixpoint` override is removed from `CategoryExplorer`.
- [ ] `learn=True` parameter is removed from `explore` in `CategoryExplorer`.
- [ ] All 13 existing tests in `tests/test_core.py` continue to pass.
- [ ] New tests `tests/test_multihop.py` and `tests/test_learn.py` exist and pass.
- [ ] `python tools/check_citations.py` exits 0.

---

## 7. Non-Goals

- **GPU / batch acceleration** — EmbR operations will run on CPU NumPy. Performance
  optimisation is deferred; correctness first. (Why: the sparse semiring implementation
  is not trivially GPU-parallelisable and this is a research codebase.)
- **Schema migration / CQL full implementation** — Lambert is not becoming a general
  CQL engine. The Σ/Δ/Π framing is the theoretical lens, not the API surface.
- **Polyadic / triadic FCA** — Bazin et al. (2024) extends FCA to arity > 2. This is
  deferred. (Why: requires a separate tensor representation; current data model is
  binary relational.)
- **Automatic chain discovery** — the multi-hop API requires the caller to specify which
  head names to chain in which order. Automated chain search (e.g., beam search over
  EmbR compositions) is deferred. (Why: requires a search budget and a scoring function
  not yet defined.)
- **LLM labelling integration** — the `legacy/language.py` Labeler is not touched by
  this PRD. (Why: it is decoupled from the main pipeline and its interface needs
  rethinking separately.)
- **Gradient-based training** — the learning rule here is the closed-form algebraic
  construction (Residuate), not backpropagation. Smooth-temperature gradient training
  is a separate future direction. (Why: the algebraic learning rule is the theoretically
  grounded path; smooth training would require a separate loss formulation.)

---

## 8. Technical Constraints

- **Stack:** Python 3.x, NumPy. No new dependencies.
- **Architecture:** Layered stack: `core/` → `lattice/` → `model.py` / `query.py`.
  New methods live in the appropriate layer:
  - Multi-hop traversal: `Query` class in `query.py` (new `multihop` method).
  - EmbR chain step: `Embed` class in `lattice/embed.py` (new `hop` method using
    existing `Join` and `Project`/`Expand`).
  - Learning rule: new `Learner` class in `lattice/embed.py` or a new
    `lattice/learn.py` (to be decided in Sprint 1, based on coupling analysis).
  - Dead code removal: `lattice/explorer.py` only.
- **Test location:** `tests/test_multihop.py` and `tests/test_learn.py`.
- **Import convention:** `from lattice.embed import Embed, Learner` (or similar — exact
  names decided in Sprint 1).
- **Temp parameter:** All new operations accept a `temp` parameter and delegate to
  existing `Join`/`Residuate`; no new smooth-approximation logic.
- **EmbR shape:** `(k, k)` where k = number of representative concepts. All EmbR matrices
  for a given model share the same k (they are all projected from the same embedding
  basis). This is the dimensional contract that makes chaining type-safe.

---

## 9. Architecture Decisions

| Decision | Reversal Cost | Alternatives Considered | Rationale |
|----------|--------------|------------------------|-----------|
| Multi-hop lives in `Query`, not `Lambert` | Low | Add to `Lambert.run()` | `Query` already owns the retrieval API; multi-hop is a query operation, not a pipeline configuration |
| Learning rule as separate `Learner` class | Medium | Method on `Lambert` | `Lambert.run()` is construction-time; learning is post-construction. Separating concerns keeps `run()` idempotent and construction deterministic |
| Merge learned weights via elementwise max (`np.maximum`) | Low | Replace R, or use weighted average | Belohlavek (2000) eq. 2 is explicitly a join (∨) over stored patterns. Max preserves existing attractors while adding new ones — exactly the right semantic |
| `hop` method on `Embed` rather than inline in `Query` | Low | Inline in `Query.multihop` | Keeps the core operation close to the algebra layer; `Query` orchestrates, `Embed` computes |
| Remove `_concept_fixpoint` override, restore parent | Low | Keep override with a comment | The override is actively misleading — it re-runs MHA on already-converged states. The parent implementation (alternating Residuate) is correct and has the Belohlavek 2-step guarantee |

---

## 10. Security Boundaries

- **Auth model:** None — this is a local Python library with no network surface.
- **Trust boundaries:** Inputs to `learn(X, Y)` are caller-supplied NumPy arrays.
  Dimension validation is required (X.shape[1] must match R.shape[0]; Y.shape[1] must
  match R.shape[1]). An incorrect shape should raise a descriptive `ValueError`, not
  silently produce a wrong result.
- **Data sensitivity:** N/A — no PII, credentials, or tokens.
- **Multi-tenancy:** N/A.

---

## 11. Data Model

The existing data model is unchanged. The relevant structures are:

**EmbR (Tucker core):** `ndarray, shape (k, k)` — stored in
`self.heads[head_name]['EmbR']` after `Lambert.run()`. One per head.

**Relation matrix R:** `ndarray, shape (n_entities, n_attributes)` — the primary data
structure. The learning rule updates this in place (or returns a new R, to be decided).

**Access patterns for new operations:**
1. Multi-hop: read `EmbR` for each head in the chain; read `emb` for final projection.
   Read-only access during a query; no mutation.
2. Learning: read existing R; write new R (via `np.maximum` merge). The updated R must
   be reflected in subsequent `ConceptEmbed` and `Attention` calls — this requires either
   a re-run of the embedding step or an incremental update. Sprint 2 will determine which
   is correct for the first implementation.

---

## 12. Shared Contracts

**EmbR shape contract:** All EmbR matrices produced by `ConceptEmbed` on the same
Lambert model instance have shape `(k, k)` where k = `emb.shape[1]` for that head.
Multi-hop chaining requires the same k across all chained heads. If heads have different
k values, chaining is ill-typed and must raise a `ValueError`.

**Hop step interface:**
```python
def hop(self, q: np.ndarray, EmbR: np.ndarray, temp: float) -> np.ndarray:
    """One left-Kan step: Join(q, EmbR) in concept space."""
    # q: (k,) concept-space vector
    # EmbR: (k, k) concept-to-concept relation
    # returns: (k,) concept-space vector
```

**Learning rule interface:**
```python
def learn(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Construct weight update from pattern pairs and merge into R.
    X: (n_patterns, n_entities) — input patterns (entity-space)
    Y: (n_patterns, n_attributes) — output patterns (attribute-space)
    Returns updated R: (n_entities, n_attributes)
    """
```

**Multi-hop query interface:**
```python
def multihop(self, entity: str, chain: list[str], top_k: int = 10) -> QueryResult:
    """Multi-hop relational inference via chained left-Kan extensions.
    entity: starting entity name
    chain: ordered list of head names to traverse
    """
```

---

## 13. Architecture Invariant Registry

| Concept | Owner | Format/Values | Verify Command |
|---------|-------|---------------|----------------|
| EmbR shape | `lattice/embed.py :: ConceptEmbed` | `(k, k)` where k = `emb.shape[1]` for the head | `python -c "import sys; sys.path.insert(0,'.'); from model import Lambert; ..."` — verified by test |
| Hop input/output type | `lattice/embed.py :: hop` | Input and output are both `(k,)` concept-space vectors; never entity-space vectors | `pytest tests/test_multihop.py::test_hop_shape` exits 0 |
| Learn merge is max-based | `lattice/embed.py :: Learner` | `R_new = np.maximum(R_old, Residuate(Y, X))` — never replace, never average | `pytest tests/test_learn.py::test_learn_preserves_old` exits 0 |
| Entity-space projection | `lattice/embed.py :: Expand` | Multi-hop final result is always projected back through `emb` before ranking | `pytest tests/test_multihop.py::test_multihop_entity_space` exits 0 |
| Dead code absent | `lattice/explorer.py` | No `_concept_fixpoint` override; no `learn` parameter on `explore` | `python -c "import ast; ..."` — verified by test assertion |

---

## 14. Open Questions

- [ ] Should `learn()` trigger a re-run of `ConceptEmbed` automatically (updating `emb`
  and `EmbR` to reflect new R), or should it update R only and leave re-embedding as a
  separate explicit call? The fully automatic path is more user-friendly but changes the
  semantics of `emb` mid-session. Recommendation: Sprint 2 implements R-update only and
  documents explicit re-embedding as required for queries to reflect new knowledge.
- [ ] The multi-hop correction principle: should each hop step apply a Residuate
  correction (as `Attention._step` does) to keep intermediate states within the concept
  lattice? The theory says yes (Shen & Tang, completeness of Mφ), but this doubles the
  operations per hop and may be overkill for a first implementation. Sprint 3 implements
  without correction first; a follow-up can add it if needed.
- [ ] Is `learn=True` the only dead parameter on `explore`, or are there others? Sprint 1
  should grep for any other unreferenced parameters before removing them.

---

## 15. Uncertainty Policy

When uncertain: Flag in a code comment and document in Agent Notes.
When theory documentation conflicts with existing code behaviour: prefer theory
documentation — the research notes are authoritative; the code is incomplete.
When a dimension mismatch is detected at runtime: raise `ValueError` with a message
that includes the shapes and the invariant that was violated.

---

## 16. Verification

**Deterministic:**
- `python -m pytest tests/test_core.py -v` — 13/13 must pass throughout all sprints
- `python -m pytest tests/test_multihop.py -v` — new tests, must pass after Sprint 3
- `python -m pytest tests/test_learn.py -v` — new tests, must pass after Sprint 2
- `python tools/check_citations.py` — must exit 0 after any research file edits

**Manual (post-sprint review):**
- Reviewer should confirm that the multi-hop path through `EmbR` matches the algebraic
  description in `theory.md` line by line.
- Reviewer should confirm that `learn()` merge uses `np.maximum` and not replacement or
  averaging.
- Reviewer should confirm that `explorer.py` contains no `_concept_fixpoint` override
  and no `learn` parameter after Sprint 1.

---

## 17. Sprint Decomposition

| Sprint | Title | Depends On | Batch | Model | Parallel With |
|--------|-------|-----------|-------|-------|---------------|
| 1 | Dead code removal | None | 1 | sonnet | — |
| 2 | Learning rule | Sprint 1 | 2 | sonnet | Sprint 3 |
| 3 | Multi-hop traversal | Sprint 1 | 2 | sonnet | Sprint 2 |
| 4 | Integration tests | Sprints 2 & 3 | 3 | sonnet | — |
| 5 | Research notes update | Sprint 4 | 4 | sonnet | — |

**Parallel note:** Sprints 2 and 3 touch disjoint file sets:
- Sprint 2 touches `lattice/embed.py` (learning rule addition) and creates
  `tests/test_learn.py`.
- Sprint 3 touches `query.py` (multihop addition) and creates `tests/test_multihop.py`,
  plus adds the `hop` method to `lattice/embed.py`.

**Conflict:** Both Sprint 2 and Sprint 3 touch `lattice/embed.py`. Therefore they
MUST be sequential, not parallel. Revised plan:

| Sprint | Title | Depends On | Batch | Model | Parallel With |
|--------|-------|-----------|-------|-------|---------------|
| 1 | Dead code removal | None | 1 | sonnet | — |
| 2 | Learning rule (`embed.py` + tests) | Sprint 1 | 2 | sonnet | — |
| 3 | Multi-hop traversal (`embed.py` + `query.py` + tests) | Sprint 2 | 3 | sonnet | — |
| 4 | Integration tests + regression | Sprint 3 | 4 | sonnet | — |
| 5 | Research notes update | Sprint 4 | 5 | sonnet | — |

### Sprint 1: Dead Code Removal → `sprints/01-dead-code-removal.md`

**Objective:** Remove the `_concept_fixpoint` override and `learn=True` dead parameter
from `CategoryExplorer`, restoring the parent implementation.
**Estimated effort:** S
**Dependencies:** None

**File Boundaries:**
- `files_to_create`: none
- `files_to_modify`: `lattice/explorer.py`
- `files_read_only`: `research/lattice/explorer.md`, `lattice/embed.py`, `tests/test_core.py`
- `shared_contracts`: none

### Sprint 2: Learning Rule → `sprints/02-learning-rule.md`

**Objective:** Implement `Learner` — the algebraic learning rule `W = Residuate(Y, X)`
merged into R via `np.maximum` — and tests verifying attractor preservation and creation.
**Estimated effort:** M
**Dependencies:** Sprint 1

**File Boundaries:**
- `files_to_create`: `tests/test_learn.py`
- `files_to_modify`: `lattice/embed.py` (add `Learner` class or `learn` method on `Embed`)
- `files_read_only`: `research/lattice/embed.md`, `research/lattice/explorer.md`,
  `core/tensor.py`, `model.py`, `tests/test_core.py`
- `shared_contracts`: `learn()` interface from Section 12

### Sprint 3: Multi-Hop Traversal → `sprints/03-multihop-traversal.md`

**Objective:** Implement `hop(q, EmbR, temp)` on `Embed` and `multihop(entity, chain)`
on `Query`, wiring EmbR into the query path for the first time.
**Estimated effort:** M
**Dependencies:** Sprint 2 (EmbR shape contract and embed.py state finalised)

**File Boundaries:**
- `files_to_create`: `tests/test_multihop.py`
- `files_to_modify`: `lattice/embed.py` (add `hop` method), `query.py` (add `multihop`)
- `files_read_only`: `research/theory.md`, `research/lattice/embed.md`,
  `research/core/tensor.md`, `model.py`, `tests/test_learn.py`
- `shared_contracts`: `hop` interface and `multihop` interface from Section 12

### Sprint 4: Integration Tests + Regression → `sprints/04-integration-tests.md`

**Objective:** Write end-to-end tests confirming that learn + multihop work together on
a synthetic knowledge graph, and that the full `Lambert.run()` + `Query` pipeline
still functions correctly.
**Estimated effort:** M
**Dependencies:** Sprints 2 & 3

**File Boundaries:**
- `files_to_create`: `tests/test_integration.py`
- `files_to_modify`: none
- `files_read_only`: `model.py`, `query.py`, `lattice/embed.py`, `lattice/explorer.py`,
  `tests/test_core.py`, `tests/test_learn.py`, `tests/test_multihop.py`
- `shared_contracts`: all interfaces from Section 12

### Sprint 5: Research Notes Update → `sprints/05-research-notes-update.md`

**Objective:** Update `research/lattice/explorer.md` and `research/lattice/embed.md`
to reflect that the dead code has been removed and both the learning rule and multi-hop
query path are now implemented. Update `research/theory.md` to mark the generation gap
as closed.
**Estimated effort:** S
**Dependencies:** Sprint 4

**File Boundaries:**
- `files_to_create`: none
- `files_to_modify`: `research/lattice/explorer.md`, `research/lattice/embed.md`,
  `research/theory.md`
- `files_read_only`: `tests/test_learn.py`, `tests/test_multihop.py`,
  `tests/test_integration.py`, `lattice/embed.py`, `query.py`
- `shared_contracts`: none

---

## 18. Execution Log

[Filled during execution — tracked in progress.json]

---

## 19. Learnings

[Filled after all sprints complete — /compound step output]
