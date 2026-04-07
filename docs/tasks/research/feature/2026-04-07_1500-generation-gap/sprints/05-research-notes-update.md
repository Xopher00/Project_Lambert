# Sprint 5: Research Notes Update

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 5 of 5
- **Depends on:** Sprint 4 (all tests passing; integration confirmed)
- **Batch:** 5 (sequential)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Update the research documentation to reflect what was actually implemented across
Sprints 1–4. Be accurate: describe what exists, how it works, and what its current
limitations are. Do NOT overstate. Do NOT claim the generation gap is "fully closed"
— the learning-rule-to-model round-trip (re-embedding after `Learner.learn()`) is
not yet wired, and that is documented as an open question. Mark what IS done: dead
code removed, `hop` and `multihop` implemented, `Learner` algebraic rule implemented.

## File Boundaries

### Creates (new files)

(none)

### Modifies (can touch)

- `research/lattice/explorer.md` — update §"Redundancies in the current implementation"
  to past tense; Sprint 1 removed `_concept_fixpoint` override and `learn=True` param.
- `research/lattice/embed.md` — update §"Project and Expand" to describe implemented
  `hop` and `Query.multihop`; update §"The learning rule" to describe `Learner.learn()`
  as implemented, with its limitation (R-only update, re-embedding required).
- `research/theory.md` — update generation gap discussion to reflect current status:
  left Kan path (`hop`/`multihop`) is implemented; learning rule is implemented at the
  algebraic layer; the round-trip from `Learner.learn()` to a queryable Lambert model
  is the remaining open problem.

### Read-Only (reference but do NOT modify)

- `tests/test_learn.py` — to confirm accurate description of what was implemented
- `tests/test_multihop.py` — to confirm accurate description of what was implemented
- `tests/test_integration.py` — to reference integration test names in docs
- `lattice/embed.py` — to confirm exact method/class names (Embed.hop, Learner.learn)
- `query.py` — to confirm exact method name (Query.multihop)
- `lattice/explorer.py` — to confirm dead code is absent

### Shared Contracts

(none — documentation only)

### Consumed Invariants

- `python tools/check_citations.py` must exit 0 after all edits.
- No new bibliography entries required (all cited papers already integrated).

## Tasks

- [ ] Read `lattice/explorer.py` to confirm `_concept_fixpoint` override is absent and
  `learn=True` parameter is absent from `explore`. Use this as the source of truth for
  what was removed.

- [ ] Read `lattice/embed.py` to confirm exact signatures for `Embed.hop` and
  `Learner.learn`. Use these verbatim in documentation updates.

- [ ] Read `query.py` to confirm exact signature for `Query.multihop`.

- [ ] Read `tests/test_integration.py` to know which integration tests exist, so docs
  can reference them accurately.

- [ ] In `research/lattice/explorer.md`:
  - Locate §"Redundancies in the current implementation" (or equivalent section about
    `_concept_fixpoint` and `learn=True`).
  - Rewrite to past tense: state that these were removed in Sprint 1 of the
    generation-gap PRD. Keep the explanation of WHY the override was wrong — it is
    theoretically instructive. Retain the section; do not delete it.

- [ ] In `research/lattice/embed.md`:
  - Locate the paragraph stating EmbR is "not wired into any query path" (or equivalent).
    Replace with: describe `Embed.hop(q, EmbR, temp)` and `Query.multihop(entity, chain)`
    as the wiring. Include the actual method signatures from the code. Describe the
    semantics: one left-Kan step per hop, final projection back to entity space via emb.T.
  - Locate the paragraph about the learning rule stating "the layer that calls it does not
    exist" (or equivalent). Replace with: describe `Learner(R).learn(X, Y)` as the
    implementation. State explicitly: `R_new = np.maximum(R_old, Residuate(Y, X))` (the
    Belohlavek 2000 eq. 2 construction). State the current limitation: `Learner.learn()`
    updates R only; callers must re-run `ConceptEmbed` on the updated R to refresh `emb`
    and `EmbR`, and reconstruct the Lambert model to make new knowledge queryable. This
    round-trip is the remaining open problem.

- [ ] In `research/theory.md`:
  - Locate "Lambert currently implements only the right Kan path" (or equivalent phrase).
    Update to: both directions are now implemented. Right Kan = `mha.retrieve` (unchanged).
    Left Kan = `Embed.hop` + `Query.multihop` (new). Learning rule = `Learner.learn()`
    (new, algebraic layer only).
  - Locate the generation gap discussion. Add a status note: the left Kan path and
    algebraic learning rule are implemented. The remaining gap is the re-embedding
    step: after `Learner.learn()`, a new `ConceptEmbed` + Lambert rebuild is required
    before new knowledge is queryable via `model.query`. This step is not yet automated.
  - Do NOT claim the gap is fully closed. Use precise language about what is done.

- [ ] Run `python tools/check_citations.py` and confirm 0 errors.

- [ ] Proofread: method names in docs must exactly match names in code. If a name
  changed during implementation, use the actual implemented name.

## Acceptance Criteria

- [ ] `research/lattice/explorer.md` no longer describes `_concept_fixpoint` override
  or `learn=True` as present features; uses past tense and explains why they were removed.
- [ ] `research/lattice/embed.md` §"Project and Expand" (or equivalent) describes
  `Embed.hop` and `Query.multihop` as implemented with their actual signatures.
- [ ] `research/lattice/embed.md` §"The learning rule" describes `Learner.learn()` as
  implemented; includes the `np.maximum` merge formula; states the re-embedding limitation.
- [ ] `research/theory.md` no longer contains the phrase "Lambert currently implements
  only the right Kan path" (or any equivalent present-tense claim that the left Kan is
  absent). The generation gap discussion accurately reflects current status.
- [ ] `python tools/check_citations.py` exits 0.
- [ ] `python -m pytest tests/ -v` exits 0 (docs-only sprint; no code was touched).

## Verification

- [ ] `python tools/check_citations.py` exits 0
- [ ] `python -m pytest tests/ -v` exits 0 (regression check — only docs were changed)
- [ ] `grep -rn "only the right Kan" research/` returns 0 matches

## Context

### Accurate status of each feature after Sprints 1–4

| Feature | Status | Where implemented | Limitation |
|---------|--------|-------------------|-----------|
| Dead code removal (`_concept_fixpoint` override, `learn=True`) | Done | `lattice/explorer.py` Sprint 1 | None |
| Algebraic learning rule (`Learner.learn`) | Done | `lattice/embed.py` Sprint 2 | R-only update; re-embedding required for query |
| Left Kan hop (`Embed.hop`) | Done | `lattice/embed.py` Sprint 3 | None |
| Multi-hop query (`Query.multihop`) | Done | `query.py` Sprint 3 | Provenance empty (no MHA.retrieve call) |
| `Learner` ↔ `Lambert` round-trip | Not done | Open question in PRD §14 | Requires `refit()` API or equivalent |
| Integration test (run + query + multihop) | Done | `tests/test_integration.py` Sprint 4 | — |

### Key phrases to update in each file

**`research/lattice/embed.md`:**
- "EmbR is currently not wired into any query path" → describe `hop` and `multihop`
- "The operation exists; the layer that calls it ... does not" → describe `Learner.learn()`

**`research/lattice/explorer.md`:**
- Present-tense descriptions of `_concept_fixpoint` override → past tense + Sprint 1 reference
- Present-tense descriptions of `learn=True` dead parameter → past tense + Sprint 1 reference

**`research/theory.md`:**
- "Lambert currently implements only the right Kan path" → both paths implemented; learning
  rule implemented at algebra layer; re-embedding round-trip is the remaining gap

When writing: be precise, not promotional. The research notes are the authoritative
theoretical record. "The round-trip from Learner.learn() to a queryable Lambert model
is not yet automated" is accurate and useful. "The generation gap is closed" is false.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
