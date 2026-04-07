# Lambert Research Integration: Kan Extensions, CQL Framing, and Provenance Legacy Migration

## 1. What & Why

**Problem:** Three bodies of work are sitting in staging limbo and not yet in the canonical research notes: (1) the Kan extension / CQL framing drafted in `research/plan/kan.md` and `research/plan/bibliography_additions.md`; (2) the `provenance/` package, which is explicitly flagged "under review" and may become redundant as the lattice layer matures; and (3) a known redundancy in `CategoryExplorer` (`_concept_fixpoint` override and dead `learn=True` parameter) documented in `research/lattice/explorer.md` but not yet fixed in code.

**Desired Outcome:**
1. The Kan/CQL additions are merged into the live research files (`core/tensor.md`, `theory.md`, `lattice/explorer.md`, `bibliography.md`). The `research/plan/` staging area is cleared.
2. The `provenance/` package is moved to `legacy/` with a clear migration note explaining what it does, why it was retired, and what replaces it — consistent with the "lattice traversal paths supersede explicit witness-based proof trees" direction stated in both `tensor.md` and `provenance/__init__.py`.
3. The two dead-code issues in `CategoryExplorer` are removed: the `_concept_fixpoint` override is deleted (restoring the parent's algebraically correct implementation), and the `learn=True` parameter is removed from `explore`.

**Justification:** Each piece is small and independent; the research notes are the primary deliverable of this project; and the `provenance/` "under review" flag has been in place long enough that a clear decision (retire to legacy) is more useful than continued ambiguity. None of these changes touch the active inference pipeline.

---

## 1b. Revision Note (2026-04-07)

Sprints 1 and 2 are complete. Sprint 3 is pending. Prior to execution, a research scan revealed significant problems in `tests/` that were not in scope when the PRD was written. The directory contains:

- Stale prototype source copies (`tests/attention.py`, `tests/explorer.py`, `tests/model.py`, `tests/train.py`, `tests/language.py`) that duplicate the pre-refactor package layout and will never run against the current canonical imports.
- Three Python test files (`tests/test_transformer.py`, `tests/test_tr_closure.py`, `tests/test_encoding.py`) that import `from transformer import Transformer` — a class that no longer exists at that path. These are broken.
- `tests/provenance_test.py` that imports `from provenance import Provenance` — broken since Sprint 2 retired provenance to `legacy/`.
- `tests/bible_test.py` that imports from the stale flat layout and pulls remote data; not a unit test.
- No pytest infrastructure: no `pyproject.toml`, no `conftest.py`, no `pytest.ini`.
- Notebooks that are exploratory and not tests.

Two new sprints are added:
- **Sprint 4: Triage and clean up `tests/`** — decide which files to keep vs. delete, remove stale prototype copies, move broken test files to `legacy/` or delete them, and establish minimal pytest infrastructure (`pyproject.toml` with `[tool.pytest.ini_options]`).
- **Sprint 5: Core operation unit tests** — write real pytest tests for `Join`, `Residuate`, `Closure`, and `SoftMax` against the current canonical API (`core.tensor`, `core.activations`).

Sprint 3 (dead code removal from CategoryExplorer) carries forward unchanged.

Sprint ordering: Sprint 4 is independent of Sprint 3 — they touch different files. They can run in parallel (batch 2). Sprint 5 depends on Sprint 4 (needs the infrastructure in place) and can run after Sprint 3 as well (batch 3).

---

## 2. Correctness Contract

**Audience:** The primary user is the project researcher (Xopher) reviewing and extending the theoretical grounding. Secondary audience is anyone building on Lambert who reads the research notes to understand the algebraic structure.

**Failure Definition:** Research notes that contradict each other, or that cite the plan/ drafts as live sources, are useless. Code changes that break `Lambert.run()`, `Query.__call__`, or any test that currently passes are failures. Tests that import from the wrong module path and silently fail to run are as bad as no tests.

**Danger Definition:** Silently breaking the `CategoryExplorer` pipeline (e.g. removing the `_concept_fixpoint` override but not verifying that phase 2 still finds the same concepts) would be harmful — wrong results with no error signal. Deleting test files that are broken-but-salvageable (i.e. whose logic is sound and only the import path is wrong) would lose correctness coverage.

**Risk Tolerance:** For documentation changes, a wrong-but-documented assumption is preferable to silence. For code changes (dead code removal, test triage), correctness is non-negotiable: verify the pipeline produces identical results before and after. For test infrastructure, a minimal working setup that actually runs is far more valuable than a comprehensive setup that cannot be invoked.

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
- `tests/` (in archive, gitignored): `provenance_test.py` was archived to `archive/tests-legacy-scripts.tar.gz` by the tests-reorganization PRD. It stays there — no further action needed.

**Additional context loaded for Sprints 4–5 (revision 2026-04-07):**
- `tests/` contains 6 stale prototype source copies: `attention.py`, `explorer.py`, `model.py`, `train.py`, `language.py` — all duplicate pre-refactor flat-layout code. The canonical versions are in `lattice/` and `model.py` at the root. These copies will never run against the current package; they should be deleted.
- `tests/test_transformer.py` and `tests/test_tr_closure.py` both `import from transformer import Transformer`. `transformer.py` no longer exists — that class was merged into the lattice layer. The test logic itself is testing `Closure` and `TransformerBlock` behavior. With the right imports (`from core.tensor import Tensor`, `from lattice.embed import Embed`) the logic could be salvaged, but the current API no longer has `TransformerBlock`. These are delete candidates.
- `tests/test_encoding.py` imports `from train import Train` and uses `gensim` to load GloVe embeddings plus `nltk.corpus.wordnet` — requires downloading large corpora. Not a unit test; test logic is an integration demo. Delete candidate.
- `tests/bible_test.py` loads remote data from GitHub. Not a unit test. Delete candidate.
- `tests/bench_fb15k.py` is a benchmark that loads the FB15k dataset. Delete candidate (or move to a `benchmarks/` directory if ever needed).
- `tests/provenance_test.py`: was archived to `archive/tests-legacy-scripts.tar.gz` by the tests-reorganization PRD. It stays archived. No move needed.
- `tests/attention.py` (`class Attention(Embed)`): a standalone exploratory version, not a test. Delete candidate.
- Notebooks (`ua-tests.ipynb`, `bible.ipynb`, `activations.ipynb`, etc.): exploratory; keep as-is (gitignored). Not the target of Sprint 4.
- Remaining after triage: `tests/` should be an empty directory ready for pytest. The new unit tests go here as `tests/test_core.py`.
- No `pyproject.toml` exists at the project root. A minimal one is needed with `[tool.pytest.ini_options]` pointing `testpaths = ["tests"]` and `pythonpath = ["."]` (so that `from core.tensor import Tensor` resolves from the project root).
- Canonical ops to test: `Tensor.Join`, `Tensor.Residuate`, `Tensor.Closure`, `Activations.SoftMax`. All live in `core/tensor.py` and `core/activations.py`. Both classes are importable via `from core.tensor import Tensor` and `from core.activations import Activations`.

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
| Stale prototype copies in `tests/` | 6 files | 0 files | `ls tests/*.py` returns only new canonical test files |
| Broken test files (wrong import path) | 4 files | 0 files | All `tests/*.py` are importable without error |
| `tests/provenance_test.py` | Archived to `archive/tests-legacy-scripts.tar.gz` (done by tests-reorganization PRD) | Final state | no further action needed |
| pytest infrastructure | None | `pyproject.toml` present with testpaths | `python -m pytest tests/ --collect-only` exits 0 |
| Core op unit tests | 0 | `tests/test_core.py` with ≥12 passing tests | `python -m pytest tests/test_core.py` exits 0 |

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

GIVEN `tests/` has been triaged  
WHEN the user runs `ls tests/*.py`  
THEN they see only canonical test files using the current package API, and no stale prototype copies

GIVEN pytest infrastructure is in place  
WHEN the user runs `python -m pytest tests/ --collect-only`  
THEN pytest finds the test files and lists the test cases without import errors

GIVEN the core unit tests are written  
WHEN the user runs `python -m pytest tests/test_core.py -v`  
THEN all tests for Join, Residuate, Closure, and SoftMax pass on small synthetic matrices with known correct answers

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
- [ ] `tests/` contains no stale prototype source copies (`attention.py`, `explorer.py`, `model.py`, `train.py`, `language.py` are gone)
- [ ] `tests/` contains no broken test files importing `from transformer import Transformer` or `from provenance import Provenance`
- [ ] `pyproject.toml` exists at the project root with `[tool.pytest.ini_options]` setting `testpaths = ["tests"]` and `pythonpath = ["."]`
- [ ] `python -m pytest tests/ --collect-only` exits 0 and finds at least 12 test items
- [ ] `python -m pytest tests/test_core.py` exits 0 with all tests passing
- [ ] `tests/test_core.py` covers: `Join` (identity, composition, sparsity), `Residuate` (adjunction law), `Closure` (convergence on DAG), `SoftMax` (sums to 1, temp=0 selects argmax)

---

## 7. Non-Goals

- **Do not implement the left Kan query path (EmbR wiring).** The research documents this gap precisely; implementing it is a separate project with its own scope. This PRD only integrates the theoretical documentation.
- **Do not implement the learning rule.** Same reasoning — documented gap, not this sprint.
- **Do not modify `lattice/explorer.py`'s first-order category discovery logic.** Only the two documented dead-code items are removed. Phase 1 (`explore`) and Phase 2 (`explore_lattice`) logic beyond those items is untouched.
- **Do not update `research/core/tensor.md`'s witness tracking section.** The retirement of provenance is captured in the legacy move; the tensor.md "under review" note can remain as-is — it is accurate.
- **Do not update `query.py` or `model.py`.** Neither imports from `provenance/`; no changes needed.
- **Do not modify the `Labeler` class in `legacy/language.py`.** It is already in legacy; out of scope.
- **Do not run any notebook.** Verification is via Python import checks and a small programmatic test.
- **Do not add a full CI/CD pipeline or GitHub Actions workflow.** A local `pyproject.toml` + pytest is sufficient.
- **Do not add linting, type checking (mypy), or coverage reporting.** Those are future improvements. Just get pytest running cleanly.
- **Do not write tests for the `lattice/` layer** (attention, explorer). Those are more complex and require a running Lambert instance. Out of scope for Sprint 5; start with pure algebraic operations in `core/`.
- **Do not write tests for `model.py` or `query.py`** in Sprint 5. An end-to-end Lambert integration test is a separate effort.
- **Do not salvage `test_transformer.py` or `test_tr_closure.py`.** The `TransformerBlock` API they test no longer exists. Delete them rather than patching.
- **Do not keep or move `bench_fb15k.py`** in Sprint 4. It requires the FB15k dataset (~1GB) and is not a test. Delete it.
- **Do not touch the gitignored notebooks** during Sprint 4. They stay as exploratory scratch space.
- **Do not add dependencies** beyond what is already installed (`numpy`, `scipy` are the only imports in `core/`).

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
| Canonical test import path | `core/` package | All test files import from `core.*` or `lattice.*`, never from flat names like `from tensor import` or `from transformer import` | `grep -r "^from tensor\|^import tensor\|^from transformer\|^import transformer\|^from algebra\|^import algebra" tests/ --include="*.py"` returns nothing |
| pytest infrastructure present | `pyproject.toml` | `[tool.pytest.ini_options]` block with `testpaths` and `pythonpath` | `python -m pytest tests/ --collect-only` exits 0 |

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
- **New (Sprint 4):**
  - `ls tests/*.py` returns nothing (all stale .py files already archived by tests-reorganization PRD)
  - `python -m pytest tests/ --collect-only` exits 0
- **New (Sprint 5):**
  - `python -m pytest tests/test_core.py -v` exits 0 with ≥13 tests passing
  - `python -m pytest tests/` exits 0 (full suite)

---

## 17. Sprint Decomposition

| Sprint | Title | Depends On | Batch | Status | Model | Parallel With |
|--------|-------|-----------|-------|--------|-------|---------------|
| 1 | Integrate Kan/CQL research notes | None | 1 | complete | sonnet | Sprint 2 |
| 2 | Retire provenance to legacy | None | 1 | complete | sonnet | Sprint 1 |
| 3 | Remove dead code from CategoryExplorer | Sprint 2 | 2 | not_started | sonnet | Sprint 4 |
| 4 | Triage and clean up `tests/` | Sprint 2 | 2 | not_started | sonnet | Sprint 3 |
| 5 | Core operation unit tests | Sprint 4 | 3 | not_started | sonnet | — |

Sprint 1 and Sprint 2 are complete. Sprint 3 and Sprint 4 are in batch 2 — they touch entirely different files and can run in parallel. Sprint 5 depends on Sprint 4 (needs the infrastructure Sprint 4 creates) and runs in batch 3.

Maximum 5 sprints — this decomposition is at the limit. If test scope needs to expand beyond `core/`, that is a separate PRD.

### Sprint 1: Integrate Kan/CQL research notes → `sprints/01-integrate-kan-research.md`

**Status: COMPLETE**

### Sprint 2: Retire provenance to legacy → `sprints/02-retire-provenance.md`

**Status: COMPLETE**

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

### Sprint 4: Triage and clean up `tests/` → `sprints/04-triage-tests.md`

**Objective:** Establish minimal pytest infrastructure by creating `pyproject.toml` at the project root so that `python -m pytest tests/` can be invoked cleanly.

**Context:** The bulk of the originally-scoped work (deleting stale `.py` files, cleaning `.gitignore`) was completed by the tests-reorganization PRD before this sprint ran. `tests/` currently contains no `.py` files. `tests/provenance_test.py` was archived to `archive/tests-legacy-scripts.tar.gz` — it stays there; no move to `legacy/provenance/` is needed. The only remaining work is creating `pyproject.toml`.

**Estimated effort:** S

**File Boundaries:**
- `files_to_create`: `pyproject.toml` (at project root)
- `files_to_modify`: none
- `files_read_only`: `legacy/provenance/__init__.py`
- `shared_contracts`: none

**Tasks:**
- [ ] Confirm `tests/` contains no stale `.py` prototype copies (expected — already archived)
- [ ] Confirm `legacy/provenance/` exists with its four source files
- [ ] Create `pyproject.toml` at the project root with `[tool.pytest.ini_options]` setting `testpaths = ["tests"]` and `pythonpath = ["."]`
- [ ] Run `python -m pytest tests/ --collect-only` — must exit 0

**Acceptance Criteria:**
- [ ] `pyproject.toml` exists with `testpaths` and `pythonpath` configured
- [ ] `python -m pytest tests/ --collect-only` exits 0

### Sprint 5: Core operation unit tests → `sprints/05-core-unit-tests.md`

**Objective:** Write `tests/test_core.py` with at least 12 passing pytest tests covering the four core operations: `Tensor.Join`, `Tensor.Residuate`, `Tensor.Closure`, and `Activations.SoftMax`. All tests use small synthetic matrices with known correct answers derivable by hand.

**Estimated effort:** S

**File Boundaries:**
- `files_to_create`: `tests/test_core.py`
- `files_to_modify`: none
- `files_read_only`: `core/tensor.py`, `core/activations.py`, `core/algebra.py`, `core/fixpoint.py`, `pyproject.toml`
- `shared_contracts`: canonical import paths (`from core.tensor import Tensor`, `from core.activations import Activations`)

**Tasks:**
- [ ] Read `core/tensor.py` fully — understand Join, Residuate, Closure signatures and semantics
- [ ] Read `core/activations.py` fully — understand SoftMax signature and semantics
- [ ] Write `tests/test_core.py` with the test cases listed below
- [ ] Run `python -m pytest tests/test_core.py -v` — all tests must pass
- [ ] Run `python -m pytest tests/ --collect-only` — confirms all tests are collected

**Test cases required:**

*Join (relational composition ∨(y: A[x,y] ∧ B[y,z]))*
1. `test_join_identity` — `Join(I, R) ≈ R` and `Join(R, I) ≈ R` for a simple 3x3 relation at low temp
2. `test_join_chain` — two-hop composition: `A→B`, `B→C` gives nonzero `A→C` entry via Join
3. `test_join_zero_input` — `Join(zeros, R)` returns a matrix of Bottom values (no path through zero rows)
4. `test_join_shape` — `Join(A, B)` returns shape `(n, p)` when A is `(n, m)` and B is `(m, p)`

*Residuate (adjoint of Join: greatest B such that A∘B ≤ C)*
5. `test_residuate_adjunction` — for any A, C: `Join(A, Residuate(A, C)) ≤ C` (adjunction inequality holds)
6. `test_residuate_identity` — `Residuate(I, C) ≈ C` when composition with identity leaves C unchanged
7. `test_residuate_shape` — returns shape `(m, p)` when A is `(n, m)` and C is `(n, p)`

*Closure (iterated Join to fixpoint)*
8. `test_closure_dag` — on a 4-node chain DAG (`0→1→2→3`), Closure finds all transitive paths (entry `[0,3]` becomes nonzero)
9. `test_closure_idempotent` — `Closure(Closure(R)) ≈ Closure(R)` (closure of closure is the closure)
10. `test_closure_superset` — `Closure(R)[i,j] ≥ R[i,j]` for all i, j (closure only adds, never removes)

*SoftMax (temperature-controlled distribution)*
11. `test_softmax_sums_to_one` — `SoftMax(x, temp=1.0, axis=0).sum(axis=0) ≈ 1` for a 1D input
12. `test_softmax_temp_zero_argmax` — at `temp=0`, `SoftMax` returns the hard argmax (single 1.0, rest 0)
13. `test_softmax_high_temp_uniform` — at very high temp, `SoftMax` approaches uniform distribution

**Acceptance Criteria:**
- [ ] `tests/test_core.py` exists and imports cleanly from `core.tensor` and `core.activations`
- [ ] `python -m pytest tests/test_core.py -v` exits 0 with all ≥13 tests passing
- [ ] Every test uses only `numpy` and the `core` package — no external corpora, no network, no large files
- [ ] No test takes more than 1 second to run

---

## 18. Execution Log

[Filled during execution]

---

## 19. Learnings

[Filled after all sprints complete]
