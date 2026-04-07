# INVARIANTS — generation-gap PRD

Cross-cutting contracts for the generation-gap feature. Each invariant is
machine-verifiable after the relevant sprint completes.

---

## EmbR Shape Contract

- **Owner:** `lattice/embed.py :: ConceptEmbed`
- **Preconditions:** Caller must only chain EmbR matrices produced by the same
  `Lambert` model instance (same embedding basis, same k).
- **Postconditions:** Every EmbR produced by `ConceptEmbed` has shape `(k, k)`
  where k = `emb.shape[1]` for that head.
- **Invariants:** All EmbR matrices in a multi-hop chain must have the same shape.
  Chaining EmbR matrices from different models or different k values is an error.
- **Verify:** `python -m pytest tests/test_multihop.py::test_hop_shape -v` exits 0
- **Fix:** If shapes differ, inspect `heads[name]['EmbR'].shape` for each head in the
  chain; re-run `Lambert.run()` if the model state is inconsistent.

---

## Hop Input/Output Type

- **Owner:** `lattice/embed.py :: hop`
- **Preconditions:** Input q must be a `(k,)` concept-space vector (not entity-space).
  EmbR must be `(k, k)`.
- **Postconditions:** Output is a `(k,)` concept-space vector.
- **Invariants:** The intermediate state at each hop step is always in concept space.
  Entity-space projection only occurs at the final step via `Join(result, emb.T)`.
- **Verify:** `python -m pytest tests/test_multihop.py::test_hop_shape -v` exits 0
- **Fix:** If entity-space vectors are leaking into hop chains, check that the query
  seed is `emb[entity_idx, :]` (concept-space row), not `R[entity_idx, :]` (raw row).

---

## Learn Merge is Max-Based

- **Owner:** `lattice/embed.py :: Learner` (or `Embed.learn`)
- **Preconditions:** Caller supplies Y: (n_patterns, n_entities) and
  X: (n_patterns, n_attributes) compatible with R's shape. Argument order follows
  W = Y ⊗ₙ Xᵀ = Residuate(Y, X) (Belohlavek 2000 eq. 2; ⊗ is the Gödel residuum,
  not min-composition — this is Residuate, not Join).
- **Postconditions:** `R_new = np.maximum(R_old, Residuate(Y, X))`.
  Existing attractors are preserved (R_new >= R_old element-wise).
- **Invariants:** The merge operation is always elementwise max. Never replace R
  entirely; never average. This is Belohlavek (2000) eq. 2: the weight matrix is
  the join (supremum) over stored pattern contributions.
- **Verify:** `python -m pytest tests/test_learn.py::test_learn_merge_is_max -v` exits 0
- **Fix:** If old attractors are being lost, confirm the merge is `np.maximum` and
  not assignment (`R = delta_R`). If shapes are wrong, confirm Y columns = n_entities
  and X columns = n_attributes (not swapped).

---

## EmbR Square Constraint

- **Owner:** `lattice/embed.py :: ConceptEmbed`
- **Preconditions:** `ConceptEmbed` is called on a relation matrix R with shape
  `(n_entities, n_attributes)` and an embedding `emb: (n_entities, k)`.
- **Postconditions:** `EmbR = emb.T ∘ R ∘ emb` has shape `(k, n_attributes_projected)`.
  EmbR is square `(k, k)` only when `n_attributes == n_entities`. For rectangular R,
  the current `Project` implementation may produce a non-square result.
- **Invariants:** Sprint 3 must validate `EmbR.shape[0] == EmbR.shape[1]` for every
  head in a multihop chain before performing any hop. If any EmbR is non-square, raise
  `ValueError` identifying the head name and actual shape.
- **Verify:** `python -m pytest tests/test_multihop.py::test_multihop_shape_mismatch -v` exits 0
- **Fix:** If EmbR is non-square, the relation matrix R for that head has
  `n_attributes ≠ n_entities`. Either use a square relation matrix, or implement a
  generalised projection that handles rectangular R (out of scope for this PRD).

---

## Entity-Space Projection at Final Hop

- **Owner:** `query.py :: Query.multihop`
- **Preconditions:** The multi-hop chain produces a concept-space result vector.
  The model's `emb` matrix must be populated.
- **Postconditions:** The final result of `multihop` is entity scores obtained by
  projecting through `emb`: `entity_scores = Join(result[np.newaxis,:], emb.T)[0]`.
- **Invariants:** `QueryResult.entities` contains entity names from `entity_labels`,
  not concept indices. The projection step is mandatory and must not be skipped.
- **Verify:** `python -m pytest tests/test_multihop.py::test_multihop_entity_space -v` exits 0
- **Fix:** If results contain concept indices instead of entity names, check that
  `_rank` is applied to `entity_scores` (shape n_entities), not to the concept vector.

---

## Dead Code Absent

- **Owner:** `lattice/explorer.py`
- **Preconditions:** Sprint 1 must be complete.
- **Postconditions:** `CategoryExplorer` does not define a `_concept_fixpoint` method
  (uses parent `Embed._concept_fixpoint`). `explore` method has no `learn` parameter.
- **Invariants:** The `_concept_fixpoint` override and `learn` parameter must not be
  re-introduced in future edits.
- **Verify:** `python -c "import ast; src=open('lattice/explorer.py').read(); tree=ast.parse(src); names=[n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]; assert '_concept_fixpoint' not in names or True; print('ok')"` — more precisely: `grep -c 'def _concept_fixpoint' lattice/explorer.py` must return 0.
- **Fix:** If the override reappears, remove it. The parent implementation is correct.

---

## QueryResult Mode Vocabulary

- **Owner:** `query.py :: Query` (and Sprint 3's `multihop` implementation)
- **Preconditions:** A `QueryResult` is returned from any Query method.
- **Postconditions:** `QueryResult.mode` is one of the declared vocabulary strings.
- **Invariants:** The set of valid mode strings is: `'forward'`, `'intersection'`,
  `'backward'`, `'multihop'`. The string `'multihop'` is defined by Sprint 3 and
  consumed by Sprint 4's integration tests. It must not change (no underscores,
  no suffix). Any new query mode must be declared here before use.
- **Verify:** `grep -E "mode='multihop'" query.py` returns at least one match (after
  Sprint 3 is complete).
- **Fix:** If Sprint 4's `assert result.mode == 'multihop'` fails, confirm that
  Sprint 3's `multihop` method sets `mode='multihop'` exactly (check for typos such
  as `'multi_hop'` or `'multihop_chain'`).

---

## Citation Integrity

- **Owner:** `tools/check_citations.py`
- **Preconditions:** All cited keys in research .md files must exist in `bibliography.md`.
- **Postconditions:** `check_citations.py` exits 0.
- **Invariants:** Applies to all sprints that touch research .md files (Sprint 5).
- **Verify:** `python tools/check_citations.py` exits 0
- **Fix:** Add missing citation keys to `bibliography.md` or correct the cite{} key
  in the research file.
