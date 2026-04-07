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

- [ ] `tests/train.py` is archived (moved to `archive/` tarball or removed after tarball created) and no longer present in `tests/`
- [ ] `tests/attention.py` is archived and no longer present in `tests/`
- [ ] `tests/explorer.py` is archived and no longer present in `tests/`
- [ ] `tests/model.py` is archived and no longer present in `tests/`

### Category B — Archive stale scripts

- [ ] `tests/language.py` is archived and no longer present in `tests/`
- [ ] `tests/provenance_test.py` is archived and no longer present in `tests/`
- [ ] `tests/bible_test.py` is archived and no longer present in `tests/`
- [ ] `tests/test_encoding.py` is archived and no longer present in `tests/`
- [ ] `tests/test_transformer.py` is archived and no longer present in `tests/`
- [ ] `tests/test_tr_closure.py` is archived and no longer present in `tests/`
- [ ] `tests/bench_fb15k.py` is archived and no longer present in `tests/`
- [ ] `tests/bible.ipynb` is archived and no longer present in `tests/`
- [ ] `tests/distill.ipynb` is archived and no longer present in `tests/`

### Category C — Notebooks: update imports

- [ ] `tests/ua-tests.ipynb` has all flat-path imports (`from algebra`, `from embed`, `from train`, etc.) replaced with `core.*` / `lattice.*` equivalents
- [ ] `tests/fixpoint.ipynb` has all flat-path imports updated
- [ ] `tests/reasoning_embedding_space.ipynb` has all flat-path imports updated
- [ ] `tests/tokenize.ipynb` has all flat-path imports updated
- [ ] `tests/training_loop.ipynb` has all flat-path imports updated (Train → core or lattice equivalent; if no equivalent exists, notebook is archived instead with a note)

### Category C — Notebooks: keep as-is (already correct paths)

- [ ] `tests/join.ipynb`, `tests/closure.ipynb`, `tests/benchmarks.ipynb`, `tests/activations.ipynb`, `tests/kan_extensions.ipynb`, `tests/lambert_vs_hydra.ipynb`, `tests/residuate.ipynb`, `tests/semiring_einsum.ipynb` are untouched

### Category C — Notebooks: self-contained explorations (keep as-is)

- [ ] `tests/countries.ipynb`, `tests/medical_kg.ipynb`, `tests/primekg.ipynb`, `tests/software_dependencies.ipynb`, `tests/tool_graph.ipynb`, `tests/lattices.ipynb` are untouched

### Housekeeping

- [ ] `tests/hydra.ipynb` and `tests/lambert_vs_hydra.ipynb` are added to `.gitignore` (they remain on disk; tracked state is user's call — current decision: gitignore them)
- [ ] `.gitignore` `tests/` section is updated: remove the 30+ per-file suppressions, replace with category-level entries reflecting the new clean state
- [ ] `archive/` contains a new tarball `tests-legacy-scripts.tar.gz` (or equivalent) holding all Category A + B files before deletion
- [ ] No file is deleted without first existing in the archive tarball

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

- [ ] `ls tests/*.py` returns nothing (all `.py` files gone from `tests/`)
- [ ] `tar -tzf archive/tests-legacy-scripts.tar.gz | wc -l` returns the expected file count (Category A + B files)
- [ ] For each updated notebook: open in Jupyter / run `jupyter nbconvert --to script` and confirm no `ImportError` on the import cells (or confirm updated imports are syntactically valid)
- [ ] `git status tests/` shows only tracked, intentional files — no untracked surprises
- [ ] `.gitignore` no longer contains per-file `tests/*` entries for archived files

## Implementation

### Step 1 — Create archive tarball

- [ ] `cd /home/scanbot/ua_tensors && tar -czf archive/tests-legacy-scripts.tar.gz tests/train.py tests/attention.py tests/explorer.py tests/model.py tests/language.py tests/provenance_test.py tests/bible_test.py tests/test_encoding.py tests/test_transformer.py tests/test_tr_closure.py tests/bench_fb15k.py tests/bible.ipynb tests/distill.ipynb`
- [ ] Verify tarball contents with `tar -tzf archive/tests-legacy-scripts.tar.gz`

### Step 2 — Remove Category A + B files from tests/

- [ ] Delete all 13 files listed in Step 1 from `tests/`

### Step 3 — Update imports in Category C notebooks

- [ ] For each of the 5 notebooks (ua-tests, fixpoint, reasoning_embedding_space, tokenize, training_loop): read each cell, identify flat-path imports, map to `core.*` / `lattice.*`, apply edits
- [ ] If `training_loop.ipynb` depends on `Train` from the old flat stack and no equivalent exists in `core.*` / `lattice.*`, archive it instead (add to tarball before deleting)

### Step 4 — Update .gitignore

- [ ] Remove the 30+ per-file `tests/*` suppressions
- [ ] Add clean replacement entries:
  - `tests/hydra.ipynb` (gitignored per decision above)
  - `tests/lambert_vs_hydra.ipynb`
  - Any data files that remain on disk but should not be tracked

### Step 5 — Final check

- [ ] `ls tests/` matches the expected keep-list
- [ ] `git status` is clean or shows only intentional new/modified files
- [ ] Commit: `chore: reorganize tests/ — archive stale scripts, update notebook imports, clean gitignore`

## Learnings (filled after completion)

[What flat-path imports had no clean mapping, which notebooks were archived instead of updated, any surprises found during the tarball step]
