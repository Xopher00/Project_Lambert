# Sprint 2: Learning Rule

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 2 of 5
- **Depends on:** Sprint 1 (dead code removed; explorer.py clean)
- **Batch:** 2 (sequential)
- **Model:** sonnet
- **Estimated effort:** M

## Objective

Implement the algebraic learning rule `W = Residuate(Y, X)` as a `Learner` class (or
method on `Embed`) in `lattice/embed.py`, and write tests verifying that new patterns
become stable attractors while existing attractors are preserved.

## File Boundaries

### Creates (new files)

- `tests/test_learn.py` — unit tests for the learning rule

### Modifies (can touch)

- `lattice/embed.py` — add Learner class or learn method on Embed (decision in Agent Notes).

### Read-Only (reference but do NOT modify)

- `research/lattice/embed.md` — authoritative description of the learning rule
  (§"The learning rule")
- `research/lattice/explorer.md` — §"The generation gap" for framing
- `core/tensor.py` — `Residuate` implementation; must use it directly
- `model.py` — to understand where `heads[name]['EmbR']` and `heads[name]['emb']`
  are stored, in case re-embedding is eventually needed
- `tests/test_core.py` — must not regress

### Shared Contracts

- `learn()` interface from PRD Section 12:
  ```python
  def learn(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
      """
      X: (n_patterns, n_attributes) — output patterns (attribute-space row vectors)
      Y: (n_patterns, n_entities)   — input patterns (entity-space row vectors)
      Returns updated R: (n_entities, n_attributes)
      Merge: R_new = np.maximum(R_old, Residuate(Y, X))
      Argument order follows W = Y ⊗ₙ Xᵀ = Residuate(Y, X) (Belohlavek 2000 eq. 2).
      """
  ```

### Consumed Invariants

- Learn merge is max-based: `R_new = np.maximum(R_old, Residuate(Y, X))`.
  Never replace R, never average.
- Input validation: Y.shape[1] must match R.shape[0] (n_entities);
  X.shape[1] must match R.shape[1] (n_attributes). Raise `ValueError` with a
  descriptive message that includes the expected and actual shapes on mismatch.

## Tasks

- [ ] Read `research/lattice/embed.md` §"The learning rule" carefully. The construction
  is: given stored pattern pairs `{(A^p, B^p)}`, the weight matrix is
  `I_ij = ∨_p A^p(g_i) ⊗ B^p(m_j)` where `⊗` is the Gödel residuum. In Lambert's
  terms this is: for each pattern pair (x_row, y_row), compute `Residuate(y_row, x_row)`
  to get a contribution matrix, then take `np.maximum` across all pattern pairs, then
  `np.maximum` with the existing R to merge.
- [ ] Decide: method on `Embed` vs. standalone `Learner` class. Document decision in
  Agent Notes with reasoning.
- [ ] Implement the learning rule in `lattice/embed.py`. Requirements:
  - Call `self.Residuate(Y, X, temp=0)` (exact, T=0 for the construction step).
    Arguments: Y (n_patterns, n_entities), X (n_patterns, n_attributes) — follows
    W = Y ⊗ₙ Xᵀ = Residuate(Y, X). The result has shape (n_entities, n_attributes)
    matching R.
  - Merge result into R via `np.maximum(R, delta_R)`.
  - Validate shapes before computation: Y.shape[1] must equal R.shape[0] (n_entities);
    X.shape[1] must equal R.shape[1] (n_attributes). Raise `ValueError` with a message
    that includes the expected shapes and the actual shapes on mismatch.
  - Do not update `emb` or `EmbR` automatically — R update only. Document this
    limitation in a docstring: "Call ConceptEmbed on updated R to refresh embeddings."
- [ ] Write `tests/test_learn.py` with the following test cases:
  - `test_learn_new_attractor`: construct a small Lambert model on 3 entities, call
    `learn` with a new entity pattern (4th entity), then verify that querying for the
    new entity returns a result.
  - `test_learn_preserves_old`: after `learn`, verify that querying for a pre-existing
    entity returns the same result as before the learn call.
  - `test_learn_merge_is_max`: directly verify that the merged R equals
    `np.maximum(R_before, delta_R)` for a synthetic case.
  - `test_learn_shape_validation`: verify that passing incompatible X or Y shapes
    raises `ValueError`.
- [ ] Run `python -m pytest tests/test_core.py tests/test_learn.py -v` and confirm
  all pass.
- [ ] Run `python tools/check_citations.py` and confirm 0 errors.

## Acceptance Criteria

- [ ] `Embed` (or `Learner`) has a `learn(X, Y)` method that implements
  `R_new = np.maximum(R_old, Residuate(Y, X))`.
- [ ] `learn(X, Y)` raises `ValueError` if X or Y shapes are incompatible with R.
- [ ] `python -m pytest tests/test_learn.py -v` exits 0 (all 4 tests pass).
- [ ] `python -m pytest tests/test_core.py -v` exits 0 (13/13, no regression).
- [ ] `python tools/check_citations.py` exits 0.

## Verification

- [ ] `python -m pytest tests/test_core.py tests/test_learn.py -v` exits 0
- [ ] `python tools/check_citations.py` exits 0
- [ ] `python -c "from lattice.embed import Embed; print('ok')"` exits 0

## Context

From `research/lattice/embed.md` §"The learning rule":

> ```
> W = Y ⊗ₙ Xᵀ  =  Residuate(Y, X)
> ```
>
> This builds the weight matrix directly from a set of input/output pattern pairs
> (X, Y). In Lambert's terms: given a set of `(entity_vector, concept_vector)` pairs,
> `Residuate(Y, X)` constructs the relation matrix `R` that stores all of them as
> stable attractors.
>
> The algebra for this is already present (`Residuate` is implemented and correct).
> What is absent is the layer that calls it with training pairs and writes the result
> back into the embeddings.

The merge is `np.maximum` because Belohlavek (2000) eq. 2 is a join (∨) over stored
pattern pairs — the weight matrix is the supremum of all per-pattern contributions.
Replacing R would lose existing attractors. Averaging would violate the algebraic
semantics (the construction rule is not defined for averages).

From `research/lattice/embed.md`:
> The operation exists; the layer that calls it with training pairs and writes back
> into the embeddings does not.

This sprint provides that layer.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
