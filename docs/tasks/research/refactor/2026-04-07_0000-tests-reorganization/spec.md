# tests/ Reorganization

## What & Why

**Problem:** `tests/` is a junk drawer accumulated over multiple restructuring phases. It mixes old source duplicates, stale integration scripts importing from a dead flat-path stack, notebooks using old imports, and self-contained research explorations — with no organizing principle. The `.gitignore` has 30+ explicit `tests/*` suppressions as a symptom: the directory grew by accumulation, never by design.

**Desired Outcome:** `tests/` contains only notebooks that are either (a) actively useful on the current package paths (`core.*`, `lattice.*`) or (b) self-contained research explorations with no internal dependencies. All dead source duplicates are archived. All stale `.py` scripts are archived. Old-path notebooks with legitimate experiments get their import lines updated. The `.gitignore` suppressions for `tests/` are replaced by a clean allowlist.

## Correctness Contract

**Audience:** The project maintainer (Xopher) and any future collaborator reading `tests/`. They use this directory to find runnable, trustworthy exploratory work.

**Failure:** The directory still contains files that silently fail to import, or the cleanup accidentally destroys a notebook with unreproducible experimental results.

**Danger:** Deleting or overwriting content that exists nowhere else in the repo. Every Category A / Category B file must be verified against a canonical counterpart or archived before removal.

## Context Loaded

- `legacy/`: contains `train.py`, `language.py`, `provenance/` — the canonical home for retired infrastructure
- `lattice/`: contains `attention.py`, `explorer.py` — the canonical, more-complete successors to the same-named files in `tests/`
- `top-level model.py` / `query.py`: import from `core.*` — the canonical new-path style
- `archive/`: already exists; contains `.tar.gz` bundles of old versions — correct location for archiving entire groups of files
- `.gitignore`: currently suppresses 30+ individual `tests/*` entries; will be replaced by a clean per-category allowlist after reorganization
- Transformer: confirmed dropped — the lattice/legacy stack replaced it; no `transformer` module exists in `core/`, `lattice/`, or `legacy/`

## Acceptance Criteria

### Category A — Archive source duplicates

- [x] `tests/train.py` is archived (moved to `archive/` tarball or removed after tarball created) and no longer present in `tests/`
- [x] `tests/attention.py` is archived and no longer present in `tests/`
- [x] `tests/explorer.py` is archived and no longer present in `tests/`
- [x] `tests/model.py` is archived and no longer present in `tests/`

### Category B — Archive stale scripts

- [x] `tests/language.py` is archived and no longer present in `tests/`
- [x] `tests/provenance_test.py` is archived and no longer present in `tests/`
- [x] `tests/bible_test.py` is archived and no longer present in `tests/`
- [x] `tests/test_encoding.py` is archived and no longer present in `tests/`
- [x] `tests/test_transformer.py` is archived and no longer present in `tests/`
- [x] `tests/test_tr_closure.py` is archived and no longer present in `tests/`
- [x] `tests/bench_fb15k.py` is archived and no longer present in `tests/`
- [x] `tests/bible.ipynb` is archived and no longer present in `tests/`
- [x] `tests/distill.ipynb` is archived and no longer present in `tests/`

### Category C — Notebooks: update imports

- [x] `tests/ua-tests.ipynb` has all flat-path imports (`from algebra`, `from embed`, `from train`, etc.) replaced with `core.*` / `lattice.*` equivalents (Train flagged with FIXME — no equivalent exists)
- [x] `tests/fixpoint.ipynb` has all flat-path imports updated (hopfield flagged with FIXME — no equivalent exists)
- [x] `tests/reasoning_embedding_space.ipynb` has all flat-path imports updated (Train flagged with FIXME)
- [x] `tests/tokenize.ipynb` has all flat-path imports updated
- [x] `tests/training_loop.ipynb` has all flat-path imports updated

### Category C — Notebooks: keep as-is (already correct paths)

- [x] `tests/join.ipynb`, `tests/closure.ipynb`, `tests/benchmarks.ipynb`, `tests/activations.ipynb`, `tests/kan_extensions.ipynb`, `tests/lambert_vs_hydra.ipynb`, `tests/residuate.ipynb`, `tests/semiring_einsum.ipynb` are untouched

### Category C — Notebooks: self-contained explorations (keep as-is)

- [x] `tests/countries.ipynb`, `tests/medical_kg.ipynb`, `tests/primekg.ipynb`, `tests/software_dependencies.ipynb`, `tests/tool_graph.ipynb`, `tests/lattices.ipynb` are untouched

### Housekeeping

- [x] `tests/hydra.ipynb` and `tests/lambert_vs_hydra.ipynb` are added to `.gitignore` (they remain on disk; tracked state is user's call — current decision: gitignore them)
- [x] `.gitignore` `tests/` section is updated: removed the 30+ per-file suppressions, replaced with data-file entries + 2 gitignored notebooks
- [x] `archive/` contains a new tarball `tests-legacy-scripts.tar.gz` (or equivalent) holding all Category A + B files before deletion
- [x] No file is deleted without first existing in the archive tarball

## Non-Goals / Boundaries

- No pytest infrastructure or test runner setup — this PRD is purely organizational cleanup
- No changes to `core/`, `lattice/`, `legacy/`, `model.py`, `query.py`, or any file outside `tests/` and `.gitignore` (and `archive/` for the new tarball)
- No new notebooks created
- `distill_data.json`, `distill_embeddings_*.npz`, `glove_cache.pkl`, `kg.csv`, `pypi_ai_training_data.pkl` — data files; archive alongside notebooks that depend on them, or gitignore in place; do NOT delete without verifying they are reproducible
- No changes to `legacy/language.py` or `legacy/train.py` — those are the canonical copies; `tests/` versions are the duplicates

## If Uncertain

When in doubt about whether a notebook import maps cleanly to a `core.*` / `lattice.*` equivalent: flag the specific import in a comment within the notebook cell (e.g., `# FIXME: old import — no clear equivalent found`), do NOT silently delete the cell, and document in the Learnings section at the bottom of this spec.

## Tradeoff Resolution

When archive-vs-delete conflicts with speed: prefer archive. Every file removed from `tests/` must first exist in the archive tarball. Safety beats cleanliness.

When old-path import vs. archive conflicts for notebooks: if a clean mapping to `core.*` / `lattice.*` exists, update. If no mapping exists (e.g., `transformer`), archive instead of leaving a broken notebook.

## Verification

- [x] `ls tests/*.py` returns nothing (all `.py` files gone from `tests/`)
- [x] `tar -tzf archive/tests-legacy-scripts.tar.gz | wc -l` returns 13 (Category A: 4 + Category B: 9)
- [x] For each updated notebook: imports updated to `core.*` / `lattice.*`; unmappable imports (Train, hopfield) flagged with `# FIXME` comments
- [x] `git status tests/` shows only untracked notebooks — no stale tracked entries
- [x] `.gitignore` no longer contains per-file `tests/*` entries for archived files

## Implementation

### Step 1 — Create archive tarball

- [x] `cd /home/scanbot/ua_tensors && tar -czf archive/tests-legacy-scripts.tar.gz tests/train.py tests/attention.py tests/explorer.py tests/model.py tests/language.py tests/provenance_test.py tests/bible_test.py tests/test_encoding.py tests/test_transformer.py tests/test_tr_closure.py tests/bench_fb15k.py tests/bible.ipynb tests/distill.ipynb`
- [x] Verify tarball contents with `tar -tzf archive/tests-legacy-scripts.tar.gz` — 13 files confirmed

### Step 2 — Remove Category A + B files from tests/

- [x] Deleted all 13 files from `tests/`

### Step 3 — Update imports in Category C notebooks

- [x] Updated all 5 notebooks: flat imports → `core.*` / `lattice.*`
- [x] `training_loop.ipynb` kept (does not import `Train`); `ua-tests.ipynb` and `reasoning_embedding_space.ipynb` import `Train` flagged with FIXME; `fixpoint.ipynb` imports `hopfield.Memory` flagged with FIXME

### Step 4 — Update .gitignore

- [x] Removed the 30+ per-file `tests/*` suppressions
- [x] Added clean replacement entries: data files + `tests/hydra.ipynb` + `tests/lambert_vs_hydra.ipynb`

### Step 5 — Final check

- [x] `ls tests/` matches expected keep-list (no .py files, correct notebooks)
- [x] `git status tests/` shows only intentional untracked files

## Learnings (filled after completion)

**Unmappable flat imports:**
- `from train import Train` — `Train` exists only in `legacy/train.py` and the now-archived `tests/train.py`. No equivalent in `core.*` or `lattice.*`. Flagged with `# FIXME` in `ua-tests.ipynb` (cell 0) and `reasoning_embedding_space.ipynb` (cell 0).
- `from hopfield import Memory` — `hopfield.py` source file does not exist; only a stale `__pycache__/hopfield.cpython-311.pyc` was present at the project root. No equivalent anywhere in `core.*` or `lattice.*`. Flagged with `# FIXME` in `fixpoint.ipynb` (cell 10).

**Notebooks kept (not archived):**
- All 5 Category C update-target notebooks were updated in place rather than archived. `training_loop.ipynb` does not import `Train` directly — it imports from `model` (Lambert), which is the current top-level path.

**Import mapping used:**
- `from algebra import` → `from core.algebra import`
- `from embed import` → `from lattice.embed import`
- `from attention import` → `from lattice.attention import`
- `from fixpoint import` → `from core.fixpoint import`
- `from explorer import` → `from lattice.explorer import`
- `from model import Lambert` — already correct (top-level), no change
- `from train import Train` — no mapping; FIXME added
- `from hopfield import Memory` — no mapping; FIXME added

**Surprises:** None. The tarball step was clean. `training_loop.ipynb` turned out not to use `Train` at all — it was already using `Lambert` from `model`.
