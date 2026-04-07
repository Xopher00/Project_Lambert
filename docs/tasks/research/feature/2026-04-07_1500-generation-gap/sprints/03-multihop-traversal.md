# Sprint 3: Multi-Hop Traversal

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 3 of 5
- **Depends on:** Sprint 2 (embed.py finalised; EmbR shape contract confirmed)
- **Batch:** 3 (sequential)
- **Model:** sonnet
- **Estimated effort:** M

## Objective

Implement `hop(q, EmbR, temp)` on `Embed` and `multihop(entity, chain, top_k)` on
`Query`, wiring EmbR into the query path for the first time and enabling multi-hop
relational inference.

## File Boundaries

### Creates (new files)

- `tests/test_multihop.py` — unit tests for the hop and multihop operations

### Modifies (can touch)

- `lattice/embed.py` — add `hop` method
- `query.py` — add `multihop` method to the `Query` class

### Read-Only (reference but do NOT modify)

- `research/theory.md` — §"The adjoint triple and multi-hop queries" for the exact
  algebraic chain; §"The Kan framing" for the semantic interpretation
- `research/lattice/embed.md` — §"Project and Expand" for EmbR shape and usage
- `research/core/tensor.md` — CQL adjoint triple table; Join semantics
- `model.py` — where `heads[name]['EmbR']` and `heads[name]['emb']` are stored
- `tests/test_core.py` — must not regress
- `tests/test_learn.py` — must not regress

### Shared Contracts

- `hop` interface from PRD Section 12:
  ```python
  def hop(self, q: np.ndarray, EmbR: np.ndarray, temp: float) -> np.ndarray:
      """One left-Kan step: Join(q[np.newaxis,:], EmbR)[0] in concept space.
      q: (k,) concept-space vector
      EmbR: (k, k) concept-to-concept relation
      returns: (k,) concept-space vector
      """
  ```
- `multihop` interface from PRD Section 12:
  ```python
  def multihop(self, entity: str, chain: list, top_k: int = 10) -> QueryResult:
      """
      entity: starting entity name
      chain: ordered list of head names (strings) to traverse
      top_k: max results to return
      """
  ```
- EmbR shape contract: all EmbR matrices chained must have the same k. Raise
  `ValueError` if shapes differ.
- EmbR square constraint: each EmbR in the chain must satisfy `EmbR.shape[0] ==
  EmbR.shape[1]`. Raise `ValueError` if not, including the head name and actual shape.
- `QueryResult.mode` for multihop queries is exactly the string `'multihop'` (no
  underscores between 'multi' and 'hop', no trailing suffix). This string is consumed
  by Sprint 4's integration tests and must not be changed without updating both sprints.

### Consumed Invariants

- Hop input/output type: input and output are both `(k,)` concept-space vectors.
  The intermediate state at each hop step must never be in entity space.
- Entity-space projection: the final result of `multihop` is projected back through
  `emb` to entity space before ranking. `Join(result_concept_vec, emb.T)` for the
  final step (or equivalently project via Expand / the embedding's column space).

## Tasks

- [ ] Read `research/theory.md` §"The adjoint triple and multi-hop queries" and
  §"The Kan framing" to confirm the exact algebraic steps:
  ```
  q₁ = Join(q,  EmbR₁)     # one hop forward (left Kan along R₁)
  q₂ = Join(q₁, EmbR₂)     # second hop forward (left Kan along R₂)
  A  = Join(q₂, emb.T)     # project back to entity space
  ```
- [ ] Implement `hop(q, EmbR, temp)` on `Embed` in `lattice/embed.py`:
  - Reshape q to `(1, k)`, call `self.Join(q[np.newaxis,:], EmbR, temp=temp)`,
    return `result[0]` as `(k,)`.
  - Validate: `q.shape == (EmbR.shape[0],)` and `EmbR.shape[0] == EmbR.shape[1]`.
    Raise `ValueError` on mismatch.
- [ ] Implement `multihop(entity, chain, top_k)` on `Query` in `query.py`:
  - Validate that all EmbR matrices in the chain have the same shape AND that each is
    square (`shape[0] == shape[1]`). Raise `ValueError` before any hop if not, with a
    message that includes the head names and their shapes.
  - Resolve entity to an index; get its concept-space seed from the FIRST head's emb:
    `q = heads[chain[0]]['emb'][entity_idx, :]` (shape `(k,)`). Do NOT use
    `R[entity_idx, :]` (that is entity-space, not concept-space).
  - For each head name in `chain`, look up `heads[name]['EmbR']` and call
    `embed.hop(q, EmbR, temp=model.attn_temp)`. The `embed` instance can be
    constructed from `Embed()` (stateless operations) or obtained from the model.
  - After the final hop, project back to entity space using the first head's emb:
    `emb = heads[chain[0]]['emb']`
    `entity_scores = embed.Join(result[np.newaxis,:], emb.T, temp=model.attn_temp)[0]`.
  - Rank using `_rank`, build provenance using available intents (may be empty for a
    pure multi-hop — document this in a docstring), return a `QueryResult` with
    `mode='multihop'` (exact string — consumed by Sprint 4 integration tests).
- [ ] Write `tests/test_multihop.py` with the following test cases:
  - `test_hop_shape`: `hop(q, EmbR, temp=0)` returns a vector of shape `(k,)`.
  - `test_hop_reachability`: on a synthetic 3-concept EmbR with one nonzero path
    from concept 0 to concept 2, `hop(e0, EmbR, temp=0)` returns a vector with
    nonzero value at concept 2.
  - `test_multihop_entity_space`: `multihop` returns a `QueryResult` with
    `mode='multihop'`; result entities are drawn from `entity_labels`.
  - `test_multihop_chain_2hop`: build a minimal synthetic Lambert model with 3
    entities A, B, C and two head relations R1 (A→B) and R2 (B→C). Call
    `multihop("A", ["head_0", "head_1"])` and confirm C appears in results.
  - `test_multihop_shape_mismatch`: passing head names with different EmbR shapes
    raises `ValueError`.
- [ ] Run `python -m pytest tests/test_core.py tests/test_learn.py tests/test_multihop.py -v`
  and confirm all pass.
- [ ] Run `python tools/check_citations.py` and confirm 0 errors.

## Acceptance Criteria

- [ ] `Embed.hop(q, EmbR, temp)` is implemented and returns `(k,)` vector.
- [ ] `Query.multihop(entity, chain, top_k)` is implemented and returns `QueryResult`
  with `mode='multihop'`.
- [ ] `test_multihop_chain_2hop` passes: C is reachable from A via [R1, R2].
- [ ] Mismatched EmbR shapes raise `ValueError`.
- [ ] `python -m pytest tests/test_core.py tests/test_learn.py tests/test_multihop.py -v`
  exits 0.
- [ ] `python tools/check_citations.py` exits 0.

## Verification

- [ ] `python -m pytest tests/test_core.py tests/test_learn.py tests/test_multihop.py -v`
  exits 0
- [ ] `python tools/check_citations.py` exits 0
- [ ] `python -c "from query import Query; print('ok')"` exits 0

## Context

The algebraic chain from `research/theory.md`:

```
Π_R(q) = Residuate(R, q)      # right Kan: universal, retrieval
Σ_R(q) = Join(q, R)           # left Kan:  existential, generation
```

Multi-hop left Kan chain (from `research/theory.md` §"The adjoint triple"):

```
q₁ = Join(q,  EmbR₁)     # one hop forward (left Kan along R₁)
q₂ = Join(q₁, EmbR₂)     # second hop forward (left Kan along R₂)
A  = Join(q₂, emb.T)     # project back to entity space
```

EmbR is a (k, k) Tucker core computed by `ConceptEmbed` and stored in
`self.heads[name]['EmbR']`. It is a concept-to-concept relation: `EmbR[i,j]` is the
degree to which concept j is reachable from concept i via this relation.

The final projection `Join(q, emb.T)` maps the concept-space result back to entity
space. `emb: (n_entities, k)`, so `emb.T: (k, n_entities)`. The output is
`(1, n_entities)` — entity-level scores.

From `research/lattice/embed.md` §"Project and Expand":

> A query in concept space — `q: (k,)` — traversed through `EmbR` via `Join(q, EmbR)`
> produces a new concept-space vector representing one relational hop. Chaining multiple
> `EmbR` matrices (one per relation type) would support multi-hop inference.

The note on provenance: multi-hop queries do not naturally produce per-head intents
(there is no MHA.retrieve call). The `intents` field of the returned `QueryResult`
can be an empty dict `{}` for the first implementation. Document this in the docstring.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
