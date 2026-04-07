# Sprint 5: Research Notes Update

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 5 of 5
- **Depends on:** Sprint 4 (all tests passing; implementation complete)
- **Batch:** 5 (sequential)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Update the research documentation to reflect that the dead code has been removed, the
learning rule is implemented, and the multi-hop query path is live. Mark the generation
gap as closed in `theory.md`. Do not add new theory — only update status and forward
references.

## File Boundaries

### Creates (new files)

(none)

### Modifies (can touch)

- `research/lattice/explorer.md` — remove or update the §"Redundancies in the current
  implementation" section to reflect Sprint 1 completion.
- `research/lattice/embed.md` — update §"Project and Expand" and §"The learning rule"
  to reflect that both features are now implemented; replace "not wired" / "absent"
  language with accurate status.
- `research/theory.md` — update §"The Kan framing" and §"The adjoint triple and
  multi-hop queries" to note that the left Kan path and learning rule are now
  implemented; update status language in the generation gap discussion.

### Read-Only (reference but do NOT modify)

- `tests/test_learn.py` — to confirm accurate description of what was implemented
- `tests/test_multihop.py` — to confirm accurate description of what was implemented
- `tests/test_integration.py` — to reference integration test names in docs
- `lattice/embed.py` — to confirm exact method names (for accurate documentation)
- `query.py` — to confirm exact method names

### Shared Contracts

(none — documentation only)

### Consumed Invariants

- `python tools/check_citations.py` must exit 0 after all edits.
- No new bibliography entries required for this sprint (all cited papers were already
  integrated in prior sessions).

## Tasks

- [ ] Read the test files (test_learn.py, test_multihop.py, test_integration.py) to
  understand exactly what was implemented before editing the docs.
- [ ] In `research/lattice/explorer.md`:
  - Update §"Redundancies in the current implementation" to past tense:
    "These redundancies have been removed in Sprint 1 of the generation-gap PRD."
    Keep the explanation of why the override was wrong — it is still valuable theory.
  - Do NOT remove the section; it documents an important design lesson.
- [ ] In `research/lattice/embed.md`:
  - §"Project and Expand": replace the paragraph starting "EmbR is currently not
    wired into any query path" with a paragraph describing the new `hop` method and
    `Query.multihop`. Include the method signature from the actual implementation.
  - §"The learning rule": replace "The operation exists; the layer that calls it ...
    does not" with a description of the new `learn()` implementation. Reference the
    Belohlavek (2000) eq. 2 construction and the np.maximum merge. Note the limitation
    that re-embedding is required after learn for queries to reflect new knowledge.
- [ ] In `research/theory.md`:
  - §"The Kan framing": replace "Lambert currently implements only the right Kan path"
    with a description of both paths now being implemented.
  - §"The adjoint triple and multi-hop queries": replace future-tense language with
    present-tense ("the chain is implemented as `Query.multihop`").
  - Add a brief status note at the top of the §"Query semantics" section or at the
    end of the generation gap discussion: "As of [date], both Kan directions are
    implemented. The generation gap described below has been closed."
- [ ] Run `python tools/check_citations.py` and confirm 0 errors.
- [ ] Proofread all three files for consistency: the descriptions should match the
  actual method names and signatures in the code.

## Acceptance Criteria

- [ ] `research/lattice/explorer.md` §"Redundancies" uses past tense and references
  the completed sprint.
- [ ] `research/lattice/embed.md` §"Project and Expand" describes `hop` and `multihop`
  as implemented features, not future directions.
- [ ] `research/lattice/embed.md` §"The learning rule" describes `learn()` as
  implemented, with the np.maximum merge documented and the re-embedding caveat noted.
- [ ] `research/theory.md` no longer contains the phrase "Lambert currently implements
  only the right Kan path" (or equivalent future-tense claim about the left Kan being
  absent).
- [ ] `python tools/check_citations.py` exits 0.
- [ ] `python -m pytest tests/ -v` exits 0 (unchanged — no code was touched).

## Verification

- [ ] `python tools/check_citations.py` exits 0
- [ ] `python -m pytest tests/ -v` exits 0 (regression check — docs only were changed)
- [ ] `grep -n "currently implements only the right Kan" research/theory.md` returns 0

## Context

The key phrases to find and update in each file:

**`research/lattice/embed.md`:**
- "EmbR is currently **not wired into any query path**" — update
- "The operation exists; the layer that calls it ... **does not**" — update

**`research/lattice/explorer.md`:**
- "These are not blocking issues" / "make phase 2 harder to reason about" — update
  to past tense since the override is now removed

**`research/theory.md`:**
- "Lambert currently implements **only the right Kan path**" — update
- "The missing generation capability is the left Kan path" — update
- "The algebra is present; the query path is not" — update

When writing updated text: be accurate, not promotional. If the implementation has
known limitations (e.g., re-embedding required after learn), document them clearly.
The research notes are the authoritative theoretical record — they should remain
honest about what the system does and does not do.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
