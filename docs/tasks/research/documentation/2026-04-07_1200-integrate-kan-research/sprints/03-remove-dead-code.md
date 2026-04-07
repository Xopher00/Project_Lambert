# Sprint 3: Remove Dead Code from CategoryExplorer

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 3 of 3
- **Depends on:** Sprint 2
- **Batch:** 2 (sequential after Sprint 2)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Remove two documented dead-code items from `lattice/explorer.py`: the `_concept_fixpoint` method override (which re-runs already-converged MHA state and loses the Belohlavek two-step guarantee) and the `learn=True` parameter from `explore` (which is declared but never read).

## File Boundaries

### Creates (new files)

- none

### Modifies (can touch)

- `lattice/explorer.py` — remove `_concept_fixpoint` method and `learn` parameter only

### Read-Only (reference but do NOT modify)

- `lattice/embed.py` — parent class; `_concept_fixpoint` implementation here is the one that will be used after removal
- `research/lattice/explorer.md` — "Redundancies in the current implementation" section; defines exactly what to remove
- `research/lattice/embed.md` — `_concept_fixpoint` section; confirms the parent implementation is algebraically correct

### Shared Contracts

- none specific to this sprint

### Consumed Invariants

- `Lambert.run()` is importable — `python -c "from model import Lambert"` exits 0
- `CategoryExplorer` importable — `python -c "from lattice import CategoryExplorer"` exits 0

## Tasks

- [ ] Read `lattice/explorer.py` in full before making any changes — understand the full class structure
- [ ] Read `lattice/embed.py` `_concept_fixpoint` implementation to confirm the parent is the correct operation (alternating Residuate on `R_active` and `R_active.T`)
- [ ] Verify Sprint 2 completed: `test ! -d /home/scanbot/ua_tensors/provenance` exits 0
- [ ] In `lattice/explorer.py`, locate the `_concept_fixpoint` method override on `CategoryExplorer` (the one that calls `self.mha.retrieve` instead of using Residuate)
- [ ] Delete the entire `_concept_fixpoint` method from `CategoryExplorer` — do not modify any other method
- [ ] In `lattice/explorer.py`, locate the `explore` method signature containing the `learn=True` parameter
- [ ] Remove `learn=True` (and `learn` if it appears as a local variable) from `explore` — do not change any other part of the method body (the method body never uses `learn` per the research note)
- [ ] Run `python -c "from lattice import CategoryExplorer; print('ok')"` — must print "ok"
- [ ] Run `python -c "from model import Lambert; print('ok')"` — must print "ok"
- [ ] Run the smoke test below to confirm the pipeline still produces categories

### Smoke test (run after changes)

```python
import numpy as np
from model import Lambert

# Minimal 4x4 relation: entities 0-3, attributes 0-3
R = np.array([
    [1.0, 0.8, 0.0, 0.0],
    [0.9, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.7],
    [0.0, 0.0, 0.8, 1.0],
], dtype=float)
vocab = ['a', 'b', 'c', 'd']
labels = ['e0', 'e1', 'e2', 'e3']

m = Lambert(entity_labels=labels)
m.run(R=R, vocab=vocab)
n_cats = len(m.concept_space['categories'])
print(f"categories: {n_cats}")
assert n_cats >= 1, "Expected at least 1 category"
print("ok")
```

This must print "ok".

## Acceptance Criteria

- [ ] `grep -n "_concept_fixpoint" lattice/explorer.py` returns nothing
- [ ] `grep -n "learn=" lattice/explorer.py` returns nothing (parameter gone)
- [ ] `python -c "from lattice import CategoryExplorer"` exits 0
- [ ] Smoke test exits 0 and prints "ok"

## Verification

- [ ] `grep -c "_concept_fixpoint" lattice/explorer.py` returns 0
- [ ] `python -c "from lattice import CategoryExplorer; print('ok')"` prints "ok"
- [ ] Smoke test passes

## Context

### What to remove and why (from `research/lattice/explorer.md`)

**`_concept_fixpoint` override:**
> `CategoryExplorer` overrides `_concept_fixpoint` from `Embed`, replacing the parent's alternating `Residuate` steps with a call to `mha.retrieve`. The override is only ever called by `ConceptEmbed` inside `explore_lattice` (phase 2). At that point it receives columns of `emb_new` as seeds — but each column of `emb_new` is already a converged MHA extent vector from phase 1. Re-running `mha.retrieve` on an already-converged state returns the same state. The override adds no new concepts or categories; no entity ever enumerated in phase 1 is reclassified.
>
> The parent `Embed._concept_fixpoint` (alternating `Residuate` on `emb_new`) is the algebraically correct operation for phase 2: it finds formal concepts of the category-extent matrix, which is second-order FCA on a well-defined rectangular context. Belohlavek (2000) Theorem 1 guarantees convergence in two steps for that operation. The override loses that guarantee and does the same work more slowly.

**`learn=True` parameter:**
> The parameter is declared in the method signature but never read inside the method body. It is dead.

The research note explicitly says: "These are not blocking issues — the pipeline produces correct results despite them." This means removal is safe: correct results are produced by the parent class implementation, and the pipeline is not relying on the override's behavior.

### Scope boundary

Only these two items are to be removed. Do not:
- Modify the `explore_lattice` method
- Modify the first-order category discovery logic in `explore`
- Modify any method in `Embed` or parent classes
- Change the `CategoryExplorer.__init__` constructor

## Agent Notes (filled during execution)

- Assigned to: —
- Started: —
- Completed: —
- Decisions made: —
- Assumptions: —
- Issues found: —
