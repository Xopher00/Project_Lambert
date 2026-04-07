# Sprint 2: Retire Provenance to Legacy

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 2 of 3
- **Depends on:** None
- **Batch:** 1 (parallel with Sprint 1)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Move the `provenance/` package to `legacy/provenance/`, update the package docstring to explain the retirement, and verify the active codebase has no broken imports.

## File Boundaries

### Creates (new files)

- `legacy/provenance/__init__.py`
- `legacy/provenance/provenance.py`
- `legacy/provenance/audit.py`
- `legacy/provenance/tree.py`

### Modifies (can touch)

- (none — all provenance files are created fresh in legacy/; docstring update happens within the create step)

### Read-Only (reference but do NOT modify)

- `model.py` — verify no provenance imports
- `query.py` — verify no provenance imports
- `lattice/explorer.py` — verify no provenance imports
- `lattice/embed.py` — verify no provenance imports
- `legacy/language.py` — reference pattern for how retired modules are documented
- `research/lattice/explorer.md` — "What the explorer does not do" section, for what supersedes provenance

### Shared Contracts

- Legacy module pattern: docstring explains retirement context and what replaces the module; class/function definitions preserved intact; module remains importable from its new location

### Consumed Invariants

- Active codebase has no provenance imports — verify before and after with grep
- `Lambert.run()` is importable — `python -c "from model import Lambert"` exits 0

## Tasks

- [ ] Run `grep -r "from provenance" . --include="*.py" | grep -v "^./provenance/" | grep -v "^./legacy/" | grep -v "__pycache__"` and confirm it returns nothing
- [ ] Run `grep -r "import provenance" . --include="*.py" | grep -v "^./provenance/" | grep -v "^./legacy/" | grep -v "__pycache__"` and confirm it returns nothing
- [ ] Create `legacy/provenance/` directory
- [ ] Copy `provenance/provenance.py` → `legacy/provenance/provenance.py` (preserve content exactly)
- [ ] Copy `provenance/audit.py` → `legacy/provenance/audit.py` (preserve content exactly)
- [ ] Copy `provenance/tree.py` → `legacy/provenance/tree.py` (preserve content exactly)
- [ ] Copy `provenance/__init__.py` → `legacy/provenance/__init__.py`, then update the docstring to replace the "Under review" explanation with a clear retirement notice (see Context below)
- [ ] Remove the `provenance/` directory from the project root
- [ ] Run `python -c "from model import Lambert; print('ok')"` — must print "ok"
- [ ] Run `python -c "from lattice import CategoryExplorer; print('ok')"` — must print "ok"
- [ ] Run `python -c "from lattice.embed import Embed; print('ok')"` — must print "ok"

## Acceptance Criteria

- [ ] `test ! -d /home/scanbot/ua_tensors/provenance` exits 0 (directory does not exist)
- [ ] `ls /home/scanbot/ua_tensors/legacy/provenance/` lists: `__init__.py`, `provenance.py`, `audit.py`, `tree.py`
- [ ] `python -c "from model import Lambert"` exits 0
- [ ] `python -c "from lattice import CategoryExplorer"` exits 0
- [ ] `legacy/provenance/__init__.py` docstring does not contain the phrase "Under review"
- [ ] `legacy/provenance/__init__.py` docstring contains the word "retired" or "retirement"

## Verification

- [ ] `test ! -d provenance` exits 0
- [ ] `python -c "from model import Lambert; print('ok')"` prints "ok"
- [ ] `python -c "from lattice import CategoryExplorer; print('ok')"` prints "ok"

## Context

### Retirement docstring for `legacy/provenance/__init__.py`

Replace the existing docstring with text along these lines (adapt to the voice of the existing codebase):

```
Provenance package. Proof extraction and formatting over relational tensors.
Retired to legacy. The active codebase no longer uses this package.

The witness-based proof tree approach documented here has been superseded by
lattice traversal paths through the concept hierarchy constructed by
CategoryExplorer. Where provenance reconstructed proof trees by tracking
intermediate Join witnesses, the concept lattice provides the same information
structurally: formal concept intents are the reason set for a query result,
and lattice navigation (meet, join, containment) gives the reasoning path.

See research/lattice/explorer.md "What the explorer does not do" for the
current theoretical framing of provenance in the lattice model.

Modules retained here as historical record:
- tree: structural recursion utilities (Tree, Tree.Pair, fold, map, zip)
- audit: proof formatting (extract_path, format_branch, format_proof)
- provenance: proof extraction (Provenance, Witnesses, Prove, Query)

The theory and implementation are sound. The architecture decision to retire
this layer was made when the lattice embedding became the primary provenance
mechanism and explicit witness storage became redundant.
```

### Reference pattern

Read `legacy/language.py` to see how a retired module is documented — the class is preserved intact, the module-level docstring explains what it was, why it is not in use, and what the open questions are.

## Agent Notes (filled during execution)

- Assigned to: —
- Started: —
- Completed: —
- Decisions made: —
- Assumptions: —
- Issues found: —
