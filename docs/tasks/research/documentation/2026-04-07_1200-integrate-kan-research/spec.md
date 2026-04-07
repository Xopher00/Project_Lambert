# Lambert Research Integration: Kan Extensions, CQL Framing, and Provenance Legacy Migration

## 1. What & Why

**Problem:** Three bodies of work are sitting in staging limbo and not yet in the canonical research notes: (1) the Kan extension / CQL framing drafted in `research/plan/kan.md` and `research/plan/bibliography_additions.md`; (2) the `provenance/` package, which is explicitly flagged "under review" and may become redundant as the lattice layer matures; and (3) a known redundancy in `CategoryExplorer` (`_concept_fixpoint` override and dead `learn=True` parameter) documented in `research/lattice/explorer.md` but not yet fixed in code.

**Desired Outcome:**
1. The Kan/CQL additions are merged into the live research files (`core/tensor.md`, `theory.md`, `lattice/explorer.md`, `bibliography.md`). The `research/plan/` staging area is cleared.
2. The `provenance/` package is moved to `legacy/` with a clear migration note explaining what it does, why it was retired, and what replaces it — consistent with the "lattice traversal paths supersede explicit witness-based proof trees" direction stated in both `tensor.md` and `provenance/__init__.py`.
3. The two dead-code issues in `CategoryExplorer` are removed: the `_concept_fixpoint` override is deleted (restoring the parent's algebraically correct implementation), and the `learn=True` parameter is removed from `explore`.

**Justification:** Each piece is small and independent; the research notes are the primary deliverable of this project; and the `provenance/` "under review" flag has been in place long enough that a clear decision (retire to legacy) is more useful than continued ambiguity. None of these changes touch the active inference pipeline.

---

## 2. Correctness Contract

**Audience:** The primary user is the project researcher (Xopher) reviewing and extending the theoretical grounding. Secondary audience is anyone building on Lambert who reads the research notes to understand the algebraic structure.

**Failure Definition:** Research notes that contradict each other, or that cite the plan/ drafts as live sources, are useless. Code changes that break `Lambert.run()`, `Query.__call__`, or any test that currently passes are failures.

**Danger Definition:** Silently breaking the `CategoryExplorer` pipeline (e.g. removing the `_concept_fixpoint` override but not verifying that phase 2 still finds the same concepts) would be harmful — wrong results with no error signal.

**Risk Tolerance:** For documentation changes, a wrong-but-documented assumption is preferable to silence. For code changes (provenance move, dead code removal), correctness is non-negotiable: verify the pipeline produces identical results before and after.

---

## 3. Context Loaded

- `research/plan/kan.md`: Three complete, insertion-ready additions targeting `core/tensor.md` (after Shen & Tang paragraph), `theory.md` (extend query semantics section), `lattice/explorer.md` (extend generation gap section). All references use `cite{key}` format consistent with existing bibliography.md keys.
- `research/plan/bibliography_additions.md`: Three new entries (Kan 1958, Schultz et al. 2017, Fong & Spivak 2019) with annotation notes for the narrative style used in `bibliography.md`. Target: "Category theory" subsection, after existing Shen & Tang (2021) entry.
- `research/lattice/explorer.md` "Redundancies" section: Documents exactly two issues — (a) `_concept_fixpoint` override in `CategoryExplorer` re-runs MHA retrieval on already-converged states, losing the Belohlavek two-step guarantee; (b) `learn=True` parameter in `explore` is never read. States "not blocking issues".
- `research/core/tensor.md` "Witness tracking" section: States "witness tracking is currently under review. As the concept lattice layer matures, lattice traversal paths may supersede explicit witness-based proof trees."
- `provenance/__init__.py`: Package-level docstring says "Under review. As lattice traversal matures in CategoryExplorer, explicit witness-based proof trees may become redundant."
- `provenance/provenance.py` class docstring: "Under review" comment identical to above.
- `legacy/language.py`: Existing example of how a retired module looks — docstring explains it is not in use and why, class/functions preserved intact.
- `model.py`: `Lambert.run()` does not import or call anything from `provenance/`. The `Query` class (`query.py`) similarly has no provenance import. The pipeline is clean of provenance dependencies.
- `tests/` (in archive, gitignored): `provenance_test.py` exists — it will move to `legacy/` alongside the code to preserve the test as historical record.

---

## 4. Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| `research/plan/` staged items | 2 files pending | 0 pending | `ls research/plan/` shows only `index.md` (or directory removed) |
| Research files citing plan/ | 0 (plan/ is staging) | 0 | Grep for `plan/` references in research/ md files |
| `provenance/` in active codebase | Yes (3 files) | No (moved to `legacy/`) | `ls provenance/` returns nothing; `ls legacy/` includes provenance files |
| Dead `_concept_fixpoint` override | Present in `lattice/explorer.py` | Removed | `grep -n "_concept_fixpoint" lattice/explorer.py` returns nothing |
| Dead `learn=True` parameter | Present in `lattice/explorer.py` | Removed | `grep -n "learn=" lattice/explorer.py` returns nothing |
| `Lambert.run()` still works | Passes | Passes | Run existing integration test / notebook smoke test |
| `CategoryExplorer.explore_lattice` results unchanged | Baseline | Identical | Run before/after on a small dataset and compare concept counts |

---

## 5. User Stories

GIVEN the Kan extension research has been reviewed and approved  
WHEN the user reads `research/core/tensor.md`  
THEN they see the Kan/CQL framing immediately after the Shen & Tang paragraph, with correct cite keys

GIVEN the Kan/CQL additions are merged  
WHEN the user reads `research/theory.md`  
THEN the query semantics section includes the right Kan / left Kan framing, the adjoint triple table, and the multi-hop query derivation

GIVEN the provenance package has been retired  
WHEN the user looks in `legacy/`  
THEN they find `provenance.py`, `audit.py`, `tree.py`, and `__init__.py` with a clear docstring explaining why they were moved, preserved intact for reference

GIVEN the dead code is removed from `CategoryExplorer`  
WHEN the user reads `lattice/explorer.py`  
THEN there is no `_concept_fixpoint` method override and no `learn` parameter on `explore`

GIVEN the dead code has been removed  
WHEN `Lambert.run()` executes on any dataset  
THEN it produces the same number and content of categories as before the change

---

## 6. Acceptance Criteria

- [ ] `research/core/tensor.md` contains the Kan extension paragraph (after the Shen & Tang paragraph) and the CQL adjoint triple table, with `cite{schultz2017}` and `cite{kan1958}` keys
- [ ] `research/theory.md` query semantics section includes the Kan framing subsection (right Kan / left Kan definitions, the Π/Σ duality, the multi-hop derivation, and the left Kan gap note)
- [ ] `research/lattice/explorer.md` generation gap section includes the Kan extension diagnosis (right Kan implemented, left Kan missing, CQL migration analogy)
- [ ] `research/bibliography.md` Category theory subsection contains entries for Kan (1958), Schultz et al. (2017), and Fong & Spivak (2019) with narrative annotations matching the existing style
- [ ] `research/plan/index.md` shows no pending items (both entries removed after merge)
- [ ] `legacy/` contains `provenance/provenance.py`, `provenance/audit.py`, `provenance/tree.py`, `provenance/__init__.py` — all moved intact, not copied
- [ ] `legacy/provenance/__init__.py` docstring explains why this package was retired and what replaces it (lattice traversal paths through the concept hierarchy)
- [ ] `provenance/` directory no longer exists at the project root
- [ ] `lattice/explorer.py` has no `_concept_fixpoint` method override
- [ ] `lattice/explorer.py` `explore` method has no `learn` parameter
- [ ] `python -c "from model import Lambert; print('ok')"` succeeds
- [ ] Running `CategoryExplorer.explore_lattice` on a small test dataset produces identical category count before and after the dead code removal

---

## 7. Non-Goals

- **Do not implement the left Kan query path (EmbR wiring).** The research documents this gap precisely; implementing it is a separate project with its own scope. This PRD only integrates the theoretical documentation.
- **Do not implement the learning rule.** Same reasoning — documented gap, not this sprint.
- **Do not modify `lattice/explorer.py`'s first-order category discovery logic.** Only the two documented dead-code items are removed. Phase 1 (`explore`) and Phase 2 (`explore_lattice`) logic beyond those items is untouched.
- **Do not update `research/core/tensor.md`'s witness tracking section.** The retirement of provenance is captured in the legacy move; the tensor.md "under review" note can remain as-is — it is accurate.
- **Do not update `query.py` or `model.py`.** Neither imports from `provenance/`; no changes needed.
- **Do not modify the `Labeler` class in `legacy/language.py`.** It is already in legacy; out of scope.
- **Do not run any notebook.** Verification is via Python import checks and a small programmatic test.

---

## 8. Technical Constraints

- **Stack:** Python, NumPy. No new dependencies.
- **Architecture:** Flat file moves for provenance retirement (no symlinks, no re-imports). Research notes are Markdown; insertions follow exact target locations specified in `research/plan/kan.md`.
- **Legacy pattern:** Follow `legacy/language.py` as the reference pattern — module stays importable, class/functions intact, docstring explains the retirement.
- **Import safety:** After the move, `from provenance.provenance import Provenance` must still work (imports from `legacy/provenance/` via a re-export shim OR the import simply breaks cleanly — given that nothing in the active codebase imports provenance, either is acceptable. Prefer the clean break: move the files, do not add a shim.)
- **Research note style:** Cite keys use `cite{key}` format (not bibtex-style `\cite`). Narrative annotations in bibliography.md are one paragraph per entry. Bold-name lead sentence style matches existing entries.

---

## 9. Architecture Decisions

| Decision | Reversal Cost | Alternatives Considered | Rationale |
|----------|--------------|------------------------|-----------|
| Move provenance to `legacy/provenance/` as a subdirectory | Low | (a) Delete entirely; (b) Leave in place with "deprecated" marker | Move preserves the code and tests as historical record, consistent with `legacy/language.py` pattern. Delete loses context. Leaving in place perpetuates the ambiguity the "under review" comment was meant to resolve. |
| Remove `_concept_fixpoint` override without adding a replacement | Low | Add a comment explaining why the override was removed | The parent class implementation is the algebraically correct one (Belohlavek two-step guarantee). No replacement needed — the parent handles it. A comment in the commit message is sufficient. |
| No import shim after provenance move | Low | Add `provenance/__init__.py` that re-exports from `legacy/` | Nothing in the active codebase imports from `provenance/`. A shim adds maintenance burden for zero benefit. |
| Insert Kan additions verbatim from `research/plan/kan.md` | Low | Rewrite / summarise | The drafts are already reviewed and insertion-ready. Verbatim insertion eliminates the risk of transcription errors. Minor style adjustments (cite key format) are the only edits needed. |

---

## 10. Security Boundaries

Not applicable. This is documentation and internal refactoring with no auth, network, or user-input surface.

---

## 11. Data Model

Not applicable. No schema changes.

---

## 12. Shared Contracts

**Cite key format:** All new bibliography references use `<!-- [keyname] -->` HTML comment format (matching the existing bibliography.md style) and `cite{keyname}` inline in research .md files. New keys introduced: `kan1958`, `schultz2017`, `fong2019`.

**Legacy module pattern:** Retired modules placed in `legacy/<name>/` as a subdirectory. `__init__.py` updated to explain retirement context and what supersedes the module. Class/function definitions preserved intact.

---

## 13. Architecture Invariant Registry

| Concept | Owner | Format/Values | Verify Command |
|---------|-------|---------------|----------------|
| Cite key consistency | `research/bibliography.md` | Keys defined in bibliography.md HTML comments must match `cite{key}` usage in research .md files | `python tools/check_citations.py` (existing pre-commit hook) |
| Active codebase has no provenance imports | `legacy/provenance/` | `provenance/` must not exist at project root after Sprint 2 | `test ! -d provenance` |
| `Lambert.run()` is importable | `model.py` | `from model import Lambert` exits 0 | `python -c "from model import Lambert; print('ok')"` |

---

## 14. Open Questions

- [ ] Should `research/plan/` directory be removed entirely after the merge, or should `index.md` be retained as an empty staging area for future drafts? (Recommendation: retain `index.md` with an empty pending list — it is a useful pattern.)
- [ ] The `tensor.md` "Witness tracking" section says provenance "is currently under review." After the retirement, should this be updated to "has been retired to legacy/"? (Recommendation: yes, as a minor follow-up edit in Sprint 1.)

---

## 15. Uncertainty Policy

When uncertain about insertion location in a research note: insert at the exact location specified in `research/plan/kan.md` (explicit target locations are given for each addition). Do not rewrite surrounding text.

When uncertain about whether a code change is safe: verify with a small programmatic test (`python -c "..."`) before and after. If results differ, stop and report.

---

## 16. Verification

- **Deterministic:**
  - `python tools/check_citations.py` — cite key consistency (existing hook)
  - `python -c "from model import Lambert; print('ok')"` — Lambert importable
  - `python -c "from lattice import CategoryExplorer; print('ok')"` — CategoryExplorer importable
  - `test ! -d /home/scanbot/ua_tensors/provenance` — provenance removed from root
  - `ls /home/scanbot/ua_tensors/legacy/provenance/` — provenance exists in legacy
- **Manual:**
  - Reviewer reads each Kan/CQL insertion in context to verify it flows naturally from the surrounding text
  - Reviewer confirms bibliography entries match the existing annotation style (bold name, one-paragraph narrative)

---

## 17. Sprint Decomposition

| Sprint | Title | Depends On | Batch | Model | Parallel With |
|--------|-------|-----------|-------|-------|---------------|
| 1 | Integrate Kan/CQL research notes | None | 1 | sonnet | Sprint 2 |
| 2 | Retire provenance to legacy | None | 1 | sonnet | Sprint 1 |
| 3 | Remove dead code from CategoryExplorer | Sprint 2 | 2 | sonnet | — |

Sprint 1 and Sprint 2 touch entirely different files (research .md files vs. Python source). They can run in parallel. Sprint 3 depends on Sprint 2 completing (to confirm `provenance/` is gone and the import graph is clean) but is otherwise independent.

### Sprint 1: Integrate Kan/CQL research notes → `sprints/01-integrate-kan-research.md`

**Objective:** Merge the staged Kan extension / CQL framing drafts into the live research notes and update `research/plan/index.md` to reflect completion.

**Estimated effort:** S

**File Boundaries:**
- `files_to_create`: none
- `files_to_modify`: `research/core/tensor.md`, `research/theory.md`, `research/lattice/explorer.md`, `research/bibliography.md`, `research/plan/index.md`
- `files_read_only`: `research/plan/kan.md`, `research/plan/bibliography_additions.md`
- `shared_contracts`: cite key format (`kan1958`, `schultz2017`, `fong2019`)

**Tasks:**
- [ ] Insert Kan extension paragraph into `research/core/tensor.md` after the Shen & Tang paragraph in the Residuate section
- [ ] Insert CQL adjoint triple table into `research/core/tensor.md` after the Kan paragraph
- [ ] Extend query semantics section in `research/theory.md` with the Kan framing subsection
- [ ] Extend generation gap section in `research/lattice/explorer.md` with the Kan diagnosis
- [ ] Add Kan (1958), Schultz et al. (2017), Fong & Spivak (2019) entries to `research/bibliography.md` Category theory subsection with annotations
- [ ] Update `research/plan/index.md` to remove both pending entries (or mark them merged)
- [ ] Update `research/core/tensor.md` "Witness tracking" section to note provenance has been retired to `legacy/` (minor follow-up)

**Acceptance Criteria:**
- [ ] `python tools/check_citations.py` exits 0 (all cite keys valid)
- [ ] All three new bibliography keys appear in bibliography.md
- [ ] `research/plan/index.md` shows no pending items

### Sprint 2: Retire provenance to legacy → `sprints/02-retire-provenance.md`

**Objective:** Move the `provenance/` package to `legacy/provenance/`, update the package docstring to explain retirement, and verify the active codebase has no broken imports.

**Estimated effort:** S

**File Boundaries:**
- `files_to_create`: `legacy/provenance/__init__.py`, `legacy/provenance/provenance.py`, `legacy/provenance/audit.py`, `legacy/provenance/tree.py`
- `files_to_modify`: none (files are moved, not edited — except for the `__init__.py` docstring update)
- `files_read_only`: `model.py`, `query.py`, `lattice/explorer.py` (confirm no imports)
- `shared_contracts`: Legacy module pattern (see Section 12)

**Tasks:**
- [ ] Confirm `grep -r "from provenance" . --include="*.py" | grep -v "^./provenance/" | grep -v "^./legacy/"` returns nothing
- [ ] Move `provenance/provenance.py`, `provenance/audit.py`, `provenance/tree.py`, `provenance/__init__.py` to `legacy/provenance/`
- [ ] Update `legacy/provenance/__init__.py` docstring: replace "Under review" with explanation that the module has been retired; note that lattice traversal paths through the concept hierarchy supersede explicit witness-based proof trees; reference `lattice/explorer.md` generation gap section
- [ ] Remove `provenance/` directory from project root
- [ ] Run `python -c "from model import Lambert; print('ok')"` — must succeed
- [ ] Run `python -c "from lattice import CategoryExplorer; print('ok')"` — must succeed

**Acceptance Criteria:**
- [ ] `test ! -d /home/scanbot/ua_tensors/provenance` exits 0
- [ ] `ls /home/scanbot/ua_tensors/legacy/provenance/` lists all four files
- [ ] `python -c "from model import Lambert"` exits 0
- [ ] `python -c "from lattice import CategoryExplorer"` exits 0

### Sprint 3: Remove dead code from CategoryExplorer → `sprints/03-remove-dead-code.md`

**Objective:** Delete the `_concept_fixpoint` override and the `learn` parameter from `CategoryExplorer` in `lattice/explorer.py`, restoring the parent class's algebraically correct implementation.

**Estimated effort:** S

**File Boundaries:**
- `files_to_create`: none
- `files_to_modify`: `lattice/explorer.py`
- `files_read_only`: `lattice/embed.py` (parent class reference), `research/lattice/explorer.md` (redundancies section), `research/lattice/embed.md` (`_concept_fixpoint` section)
- `shared_contracts`: none

**Tasks:**
- [ ] Read `lattice/explorer.py` fully before making any changes
- [ ] Read `lattice/embed.py` `_concept_fixpoint` implementation to confirm parent is correct
- [ ] Remove the `_concept_fixpoint` method from `CategoryExplorer` class
- [ ] Remove `learn=True` parameter from the `explore` method signature
- [ ] Run `python -c "from lattice import CategoryExplorer; print('ok')"` — must succeed
- [ ] Run a small programmatic smoke test: construct a minimal 4-entity relation matrix, run `Lambert.run()`, confirm categories are found (non-empty)

**Acceptance Criteria:**
- [ ] `grep -n "_concept_fixpoint" lattice/explorer.py` returns nothing
- [ ] `grep -n "learn=" lattice/explorer.py` returns nothing
- [ ] `python -c "from lattice import CategoryExplorer"` exits 0
- [ ] Smoke test: Lambert produces at least 1 category on a 4x4 relation matrix

---

## 18. Execution Log

[Filled during execution]

---

## 19. Learnings

[Filled after all sprints complete]
