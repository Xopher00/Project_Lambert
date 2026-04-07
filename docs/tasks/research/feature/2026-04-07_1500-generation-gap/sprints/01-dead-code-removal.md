# Sprint 1: Dead Code Removal

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 1 of 5
- **Depends on:** None
- **Batch:** 1 (sequential)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Remove the `_concept_fixpoint` override and the `learn=True` dead parameter from
`CategoryExplorer`, restoring the correct parent `Embed._concept_fixpoint` for phase 2
lattice closure.

## File Boundaries

### Creates (new files)

(none)

### Modifies (can touch)

- `lattice/explorer.py` — remove the _concept_fixpoint override method and the dead learn parameter from explore.

### Read-Only (reference but do NOT modify)

- `research/lattice/explorer.md` — authoritative explanation of why the override is
  wrong and what the correct phase 2 behaviour is.
- `lattice/embed.py` — parent `_concept_fixpoint` that the override replaces; must
  understand its signature to confirm removal is safe.
- `tests/test_core.py` — must not regress.

### Shared Contracts

(none — this sprint touches no interfaces consumed by other sprints)

### Consumed Invariants

- Dead code absent invariant: after this sprint, `grep -n '_concept_fixpoint'
  lattice/explorer.py` must return nothing (no override); `grep -n 'learn='
  lattice/explorer.py` must return nothing (no parameter).

## Tasks

- [ ] Read `research/lattice/explorer.md` §"Redundancies in the current implementation"
  to confirm the exact methods and parameters to remove.
- [ ] In `lattice/explorer.py`: delete the `_concept_fixpoint` method override entirely.
  The parent `Embed._concept_fixpoint` (alternating Residuate steps) is the correct
  implementation and will be used automatically after removal.
- [ ] In `lattice/explorer.py`: remove `learn=True` (or `learn=False`) from the `explore`
  method signature. Remove any reference to `learn` inside the method body if any exists.
  If it is only in the signature and never read, this is a single-line change.
- [ ] Grep for any other dead parameters or unreferenced local variables introduced in the
  same function while you have the file open. Document findings in Agent Notes. Do NOT
  fix anything outside the two named items unless they are trivially one-line removals
  with no semantic risk.
- [ ] Run `python -m pytest tests/test_core.py -v` and confirm 13/13 pass.
- [ ] Run `python tools/check_citations.py` and confirm it exits 0.
- [ ] Run a quick smoke test: `python -c "from lattice.explorer import CategoryExplorer;
  print('ok')"` to confirm the import still works.

## Acceptance Criteria

- [ ] `lattice/explorer.py` contains no `_concept_fixpoint` method defined at the
  `CategoryExplorer` class level (the class no longer overrides the parent method).
- [ ] `explore` in `CategoryExplorer` has no `learn` parameter in its signature.
- [ ] `python -m pytest tests/test_core.py -v` exits 0 (13/13).
- [ ] `python -c "from lattice.explorer import CategoryExplorer; print('ok')"` exits 0.
- [ ] `python tools/check_citations.py` exits 0.

## Verification

- [ ] `python -m pytest tests/test_core.py -v` exits 0
- [ ] `python -c "from lattice.explorer import CategoryExplorer; print('ok')"` exits 0
- [ ] `python tools/check_citations.py` exits 0
- [ ] `grep -c '_concept_fixpoint' lattice/explorer.py` returns 0

## Context

From `research/lattice/explorer.md`:

> **`_concept_fixpoint` override.** `CategoryExplorer` overrides `_concept_fixpoint`
> from `Embed`, replacing the parent's alternating `Residuate` steps with a call to
> `mha.retrieve`. The override is only ever called by `ConceptEmbed` inside
> `explore_lattice` (phase 2). At that point it receives columns of `emb_new` as
> seeds — but each column of `emb_new` is already a converged MHA extent vector from
> phase 1. Re-running `mha.retrieve` on an already-converged state returns the same
> state. The override adds no new concepts or categories; no entity ever enumerated in
> phase 1 is reclassified.
>
> **`learn=True` parameter on `explore`.** The parameter is declared in the method
> signature but never read inside the method body. It is dead.

The parent `Embed._concept_fixpoint` (alternating `Residuate` steps) has the
Belohlavek (2000) Theorem 1 two-step convergence guarantee. The override loses that
guarantee while doing the same work more slowly. Restoring the parent is strictly
better.

## Agent Notes (filled during execution)

- Assigned to: [Agent ID / session]
- Started: [timestamp]
- Completed: [timestamp]
- Decisions made: []
- Assumptions: []
- Issues found: []
