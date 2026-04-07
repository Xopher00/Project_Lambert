# Sprint 4: Integration Tests + Regression

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 4 of 5
- **Depends on:** Sprint 3 (learn + multihop both implemented)
- **Batch:** 4 (sequential)
- **Model:** sonnet
- **Estimated effort:** M

## Objective

Write an end-to-end integration test that exercises `Lambert.run()`, `learn()`, and
`multihop()` together on a synthetic knowledge graph, verifying that all three work
in combination and that the full pipeline regresses cleanly.

## File Boundaries

### Creates (new files)

- `tests/test_integration.py` — integration tests

### Modifies (can touch)

(none — this sprint only adds tests)

### Read-Only (reference but do NOT modify)

- `model.py` — `Lambert` class, `run()` interface
- `query.py` — `Query` class, `__call__` and `multihop` interfaces
- `lattice/embed.py` — `learn` interface
- `lattice/explorer.py` — post-Sprint-1 clean state
- `tests/test_core.py`, `tests/test_learn.py`, `tests/test_multihop.py` — for
  patterns and synthetic data construction

### Shared Contracts

- `Lambert.run()` → populates `heads`, `concept_space`, `explorer`, `query`
- `model.query(entities=[...])` → `QueryResult`
- `model.query.multihop(entity, chain)` → `QueryResult`
- `learn(X, Y)` interface from Sprint 2
- `QueryResult.mode` values: `'forward'`, `'intersection'`, `'backward'`, `'multihop'`

### Consumed Invariants

- All five INVARIANTS.md invariants (EmbR shape, hop type, learn merge, entity-space
  projection, dead code absent) should be verifiable from the integration test passing.

## Tasks

- [ ] Design a synthetic knowledge graph with the following structure (document inline):
  - 4 entities: Alice, Bob, Carol, Dave
  - 2 relation types: `works_with` and `reports_to`
  - `works_with`: Alice-Bob (0.9), Bob-Carol (0.8)
  - `reports_to`: Alice-Dave (0.9), Carol-Dave (0.7)
  - Designed so that the chain `works_with → reports_to` makes Dave reachable from
    Alice via Bob (Alice works_with Bob, Bob reports_to nobody directly, but
    Carol reports_to Dave and Alice works_with Bob works_with Carol — 2 hops via
    works_with, then 1 via reports_to). Adjust the graph as needed to get a clean
    testable 2-hop result.
- [ ] Write `test_lambert_run_and_query`: call `Lambert(entity_labels=[...]).run(...)`
  on the synthetic graph, verify `concept_space` is populated, verify a forward query
  returns results.
- [ ] Write `test_learn_then_query`: after `run()`, call `learn()` with a new pattern
  for a 5th entity "Eve" (who works_with Alice). Then query for Eve and confirm results
  are returned. Then query for Alice and confirm results match pre-learn results.
- [ ] Write `test_multihop_end_to_end`: after `run()`, call
  `model.query.multihop("Alice", ["works_with", "reports_to"])` and confirm Dave
  appears in the result (Dave is reachable from Alice via the 2-hop chain).
- [ ] Write `test_full_pipeline_regression`: run the full pipeline (run + query + learn
  + multihop) in sequence and confirm no exceptions are raised. This is a smoke test
  that exercises all new code paths together.
- [ ] Run `python -m pytest tests/ -v` (all test files) and confirm everything passes.
- [ ] Run `python tools/check_citations.py` and confirm 0 errors.

## Acceptance Criteria

- [ ] `tests/test_integration.py` exists and contains at least 4 tests.
- [ ] `test_multihop_end_to_end` passes: Dave is reachable from Alice via the 2-hop chain.
- [ ] `test_learn_then_query` passes: Eve is queryable after learn; Alice results unchanged.
- [ ] `python -m pytest tests/ -v` exits 0 (all tests in all test files pass).
- [ ] `python tools/check_citations.py` exits 0.

## Verification

- [ ] `python -m pytest tests/ -v` exits 0
- [ ] `python tools/check_citations.py` exits 0

## Context

The synthetic knowledge graph should be constructed as NumPy matrices passed to
`Lambert.run(relations=...)`. Each head gets a matrix and a list of feature labels.

Example structure:
```python
import numpy as np
entities = ["Alice", "Bob", "Carol", "Dave"]
n = len(entities)

R_works = np.zeros((n, n))
R_works[0, 1] = 0.9   # Alice works_with Bob
R_works[1, 2] = 0.8   # Bob works_with Carol

R_reports = np.zeros((n, n))
R_reports[0, 3] = 0.9  # Alice reports_to Dave
R_reports[2, 3] = 0.7  # Carol reports_to Dave

relations = {
    'works_with': (R_works, entities),
    'reports_to': (R_reports, entities),
}
model = Lambert(entity_labels=entities)
model.run(relations=relations, n_entities=n)
```

For the 2-hop test, the chain `["works_with", "reports_to"]` should make Dave
reachable from Alice via: Alice → (works_with) → Bob → (works_with) → Carol →
(reports_to) → Dave. The exact reachability depends on how EmbR compresses the
concept space — adjust the test assertion to check that Dave's score is above eps
rather than checking exact values.

For `learn()`, construct X and Y as the new entity's relation vectors:
```python
# Eve works_with Alice at strength 0.85
X_eve = np.zeros((1, n))  # entity-space input
X_eve[0, 0] = 0.85        # Alice index
Y_eve = np.zeros((1, n))  # attribute-space output (same space for entity-entity R)
# adjust shapes as required by the actual learn() interface
```
The exact interface will be confirmed from Sprint 2's implementation.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
