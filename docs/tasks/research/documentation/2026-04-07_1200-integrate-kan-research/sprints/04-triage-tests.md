# Sprint 4: Triage and Clean Up tests/

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 4 of 5
- **Depends on:** Sprint 2
- **Batch:** 2 (parallel with Sprint 3)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Establish minimal pytest infrastructure by creating `pyproject.toml` at the project root, so that `python -m pytest tests/` can be invoked cleanly.

## Context Note

The stale `.py` files that were originally in `tests/` were already removed by the tests-reorganization PRD (archived to `archive/tests-legacy-scripts.tar.gz`). `tests/` currently contains only notebooks and data files — no `.py` files at all.

`tests/provenance_test.py` was also archived in that same effort. It stays archived. The spec.md acceptance criterion requiring it at `legacy/provenance/provenance_test.py` is superseded — it is in the archive tarball and that is the correct final state.

The `.gitignore` was already cleaned up by the tests-reorganization PRD (replaced with an allowlist). No `.gitignore` changes are needed.

The only remaining work for this sprint is creating `pyproject.toml`.

## File Boundaries

### Creates (new files)

- `pyproject.toml` — at project root, minimal pytest configuration

### Modifies (can touch)

- none

### Read-Only (reference but do NOT modify)

- `legacy/provenance/__init__.py` — confirm legacy provenance exists (Sprint 2 postcondition)
- `.gitignore` — read-only; confirm it is already clean (no test files listed)

### Shared Contracts

- none

### Consumed Invariants

- `Lambert.run()` is importable — `python -c "from model import Lambert"` exits 0
- Active codebase has no provenance imports — `test ! -d /home/scanbot/ua_tensors/provenance` exits 0

## Tasks

- [ ] Confirm `tests/` contains no stale `.py` prototype copies: `ls tests/*.py` returns nothing (expected — already archived)
- [ ] Confirm `legacy/provenance/` exists with its four files (`__init__.py`, `audit.py`, `provenance.py`, `tree.py`)
- [ ] Create `pyproject.toml` at the project root with `[tool.pytest.ini_options]` block setting `testpaths = ["tests"]` and `pythonpath = ["."]`
- [ ] Run `python -m pytest tests/ --collect-only` — must exit 0 (zero tests collected is acceptable at this stage; no import errors is the requirement)

## Acceptance Criteria

- [ ] `pyproject.toml` exists at the project root with `[tool.pytest.ini_options]` containing `testpaths` and `pythonpath`
- [ ] `python -m pytest tests/ --collect-only` exits 0

## Verification

- [ ] `python -m pytest tests/ --collect-only` exits 0
- [ ] `python -c "from model import Lambert; print('ok')"` prints "ok"

## Agent Notes (filled during execution)

- Assigned to: —
- Started: —
- Completed: —
- Decisions made: —
- Assumptions: —
- Issues found: —
