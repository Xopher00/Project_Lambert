# Sprint 4: Integration Tests + Regression

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 4 of 5
- **Depends on:** Sprint 3 (learn + multihop both implemented)
- **Batch:** 4 (sequential)
- **Model:** sonnet
- **Estimated effort:** M

## Objective

Write integration tests that confirm the implemented pipeline works end-to-end using
a real `Lambert.run()` call and the `Query` interface. Fix `test_embed.py` which has a
broken import. Do NOT attempt to wire `Learner` into `Lambert` — that is a structural
design change deferred to a later PRD. Integration tests here mean: real model, real
query, real multihop — all three verified to function together. The `learn` path is
tested at the `Learner` unit level (already done in Sprint 2); integration of `Learner`
with `Lambert` requires re-embedding and is out of scope.

## File Boundaries

### Creates (new files)

- `tests/test_integration.py` — integration tests using real Lambert instances

### Modifies (can touch)

- `tests/test_embed.py` — fix the broken `from tests.test_learn import` to use a
  relative import or direct import that does not require `tests` to be a package

### Read-Only (reference but do NOT modify)

- `model.py` — `Lambert` class, `run()` interface, `_explore`, `_get_embeddings`
- `query.py` — `Query` class, `__call__` and `multihop` interfaces
- `lattice/embed.py` — `Embed`, `Learner`, `hop` interfaces
- `lattice/explorer.py` — post-Sprint-1 clean state
- `tests/test_core.py`, `tests/test_learn.py`, `tests/test_multihop.py` — must not regress

### Shared Contracts

- `Lambert.run(relations=..., n_entities=...)` → populates `heads`, `concept_space`,
  `explorer`, `query` on the model instance
- `model.query(entities=...)` → `QueryResult` with `mode='forward'` or `'intersection'`
- `model.query.multihop(entity, chain)` → `QueryResult` with `mode='multihop'`
- `QueryResult.mode` values: `'forward'`, `'intersection'`, `'backward'`, `'multihop'`
- `Learner(R).learn(X, Y)` → updated `R` ndarray; does NOT affect a Lambert instance

### Consumed Invariants

- EmbR shape invariant: `heads[name]['EmbR'].shape == (k, k)` after run().
- `model.query` is a `Query` instance (assigned by `_explore`) after `run()`.
- `multihop` result has `mode == 'multihop'` (exact string).

## Key Facts (must read before writing tests)

### What Learner is and is not

`Learner` (in `lattice/embed.py`) is a standalone class that wraps a bare relation
matrix `R` and updates it via `np.maximum(R, Residuate(Y, X))`. It has no connection
to a trained `Lambert` instance. After `Learner.learn()`, the updated `R` could be
passed to `ConceptEmbed` to get new `emb`/`EmbR`, but then a new Lambert model would
need to be constructed from scratch — the existing `model.query` instance would be
stale. This round-trip is a future concern; do NOT attempt it in this sprint.

The integration tests for `Learner` are already in `tests/test_learn.py`. This sprint
does NOT add more `Learner` tests. The integration we care about here is:
`Lambert.run()` → `model.query()` → `model.query.multihop()` working together.

### What multihop returns on a real Lambert model

After `Lambert.run(relations=..., n_entities=...)`, each head in `model.heads` has an
`emb` of shape `(n_entities, k)` and an `EmbR` of shape `(k, k)`. For sparse
entity-to-entity relation matrices (which is what the synthetic graph uses), `k` will
equal `n_entities` (ConceptEmbed selects all columns) and `EmbR` may be mostly Bottom
values. This is correct behaviour — a sparse relation simply has few concept-space paths.

Do NOT assert that specific entities appear in multihop results when using a sparse
entity-to-entity matrix. The correct assertion is structural: `QueryResult` returned,
`mode == 'multihop'`, entities (if any) drawn from `entity_labels`, no exception raised.

### How to build a relation matrix that actually works for multihop

Use dense relation matrices with binary attribute columns (not entity-to-entity). An
entity-to-attribute relation is dense enough for `ConceptEmbed` to select representative
concepts and for `EmbR` to have non-Bottom values.

Example: 4 entities × 6 boolean attributes (skills/traits). This gives a well-formed
`(4, 6)` R matrix where ConceptEmbed selects a proper subset of columns, EmbR has
structure, and multihop can propagate.

Alternatively: test reachability with the mock-model approach (already in test_multihop),
which guarantees controlled EmbR values. The integration test's purpose is that `Lambert.run()`
+ `Query` compose without error, not that specific entity-level multihop reachability holds
on synthetic data with arbitrary algebra compression.

### The test_embed.py import bug

`tests/test_embed.py` has `from tests.test_learn import (...)` which fails because
`tests/` has no `__init__.py`, so Python does not treat it as a package. Fix: change
to `from test_learn import (...)` (relative, since pytest adds `tests/` to sys.path
via `pythonpath = ["."]` in pyproject.toml — actually it adds the project root, and
pytest's `rootdir` and collection machinery add testpaths to sys.path). The safest fix:
`from tests.test_learn import` → `import importlib, sys; ...` is overly complex.
The correct minimal fix: simply import directly since pytest adds the testpaths dir.
Use `from test_learn import (...)` without the `tests.` prefix.

## Tasks

- [ ] Fix `tests/test_embed.py`: change `from tests.test_learn import` to
  `from test_learn import`. Confirm `python -m pytest tests/test_embed.py -v` exits 0.

- [ ] Read `model.py` `_get_embeddings` and `_explore` to understand what attributes
  are available on the model instance after `run()`. Specifically confirm:
  - `model.heads[name]` has `'emb'`, `'EmbR'`, `'rep_cols'`, `'feature_labels'`
  - `model.query` is a `Query` instance
  - `model.concept_space` has `'emb'`, `'categories'`, `'feature_map'`

- [ ] Design a synthetic Lambert fixture with at least 4 entities and 4 attributes
  (entity-to-attribute, not entity-to-entity) that produces a well-formed embedding.
  Use a dense enough R that `ConceptEmbed` selects at least 2 representative concepts.
  Document the fixture inline. Use the same fixture across all integration tests.

- [ ] Write `test_lambert_run_populates_state`: after `Lambert(...).run(relations=...,
  n_entities=...)`, assert that `model.heads` is a non-empty dict, each head has `'emb'`
  and `'EmbR'` keys, and `model.query` is a `Query` instance. This confirms the full
  pipeline initialises without error.

- [ ] Write `test_lambert_forward_query_returns_result`: using the synthetic fixture,
  call `model.query(entities=entity_labels[0])` and assert the result is a `QueryResult`
  instance with `mode` in `('forward', 'intersection')`. Do not assert specific entities
  in the result — the concept lattice is data-dependent. Assert only that the call
  completes without error and returns the correct type.

- [ ] Write `test_multihop_on_real_model`: after `run()`, call
  `model.query.multihop(entity_labels[0], [first_head_name])` (single-head, one hop).
  Assert: returns `QueryResult`, `mode == 'multihop'`, all entities in result are in
  `entity_labels`. Do NOT assert which entities are returned — that is algebra-dependent.

- [ ] Write `test_multihop_chain_on_real_model`: after `run()` on a model with at least
  2 heads, call `model.query.multihop(entity_labels[0], [head1, head2])` (two-hop chain).
  Assert: returns `QueryResult`, `mode == 'multihop'`, no exception raised.
  (This is a smoke test — chain execution without error is the goal.)

- [ ] Write `test_full_pipeline_smoke`: call `run()`, then `model.query()`, then
  `model.query.multihop()`, in sequence. Assert no exceptions. This is the regression
  guard: if any plumbing breaks between sprints, this fails.

- [ ] Run `python -m pytest tests/ -v` and confirm all tests pass (including the
  newly fixed test_embed.py).

- [ ] Run `python tools/check_citations.py` and confirm 0 errors.

## Acceptance Criteria

- [ ] `tests/test_embed.py` imports without error: `python -m pytest tests/test_embed.py -v`
  exits 0.
- [ ] `tests/test_integration.py` exists and contains at least 5 tests.
- [ ] `test_lambert_run_populates_state` passes: heads, EmbR, query all present after run().
- [ ] `test_multihop_on_real_model` passes: QueryResult returned, mode='multihop', no error.
- [ ] `test_full_pipeline_smoke` passes: run + query + multihop all complete without error.
- [ ] `python -m pytest tests/ -v` exits 0 (all tests in all test files pass).
- [ ] `python tools/check_citations.py` exits 0.

## Verification

- [ ] `python -m pytest tests/ -v` exits 0
- [ ] `python tools/check_citations.py` exits 0

## Context

### Concrete fixture pattern (entity-to-attribute)

```python
import numpy as np
from model import Lambert
from query import Query, QueryResult

def _build_real_model():
    """
    4 entities × 6 binary attributes. Dense enough for ConceptEmbed to
    select multiple representative concepts per head.

    Two heads: 'traits' (cols 0-2) and 'roles' (cols 3-5).
    """
    entities = ['Alice', 'Bob', 'Carol', 'Dave']
    n = len(entities)

    R_traits = np.array([
        [0.9, 0.2, 0.0],  # Alice: trait_0 strong, trait_1 weak
        [0.1, 0.8, 0.3],  # Bob:   trait_1 strong
        [0.5, 0.5, 0.9],  # Carol: trait_2 strong
        [0.0, 0.7, 0.6],  # Dave:  trait_1 and trait_2
    ], dtype=float)

    R_roles = np.array([
        [0.8, 0.1, 0.0],  # Alice: role_0
        [0.0, 0.9, 0.2],  # Bob:   role_1
        [0.3, 0.3, 0.8],  # Carol: role_2
        [0.6, 0.0, 0.4],  # Dave:  role_0 and role_2
    ], dtype=float)

    relations = {
        'traits': (R_traits, ['trait_0', 'trait_1', 'trait_2']),
        'roles':  (R_roles,  ['role_0',  'role_1',  'role_2']),
    }
    model = Lambert(entity_labels=entities, embed_temp=1.0, attn_temp=1.0)
    model.run(relations=relations, n_entities=n)
    return model, entities
```

### Why not test learn → query round-trip here

`Learner.learn(X, Y)` updates a bare R matrix. To make those changes visible in
`model.query`, one would need to: (1) update `model.heads[name]['emb']` and
`model.heads[name]['EmbR']` by re-running `ConceptEmbed` on the updated R, then
(2) rebuild the `MultiHeadAttention` and `CategoryExplorer`, then (3) reassign
`model.query`. This is not a test concern — it is an API design concern. The correct
fix is for `Lambert` to expose a `refit(head_name, new_R)` method that does this
sequence. That is a design change tracked as an open question in the PRD (Section 14)
and deferred beyond this sprint.

The `Learner` unit tests in `test_learn.py` already verify the algebraic correctness
of the learning rule. This sprint's integration test verifies that `Lambert.run()` and
`Query` compose correctly, which is the gap that was not covered.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
