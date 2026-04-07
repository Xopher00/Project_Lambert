# Sprint 5: Core Operation Unit Tests

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 5 of 5
- **Depends on:** Sprint 4
- **Batch:** 3 (sequential — needs pyproject.toml from Sprint 4)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Write `tests/test_core.py` with at least 13 passing pytest tests covering `Tensor.Join`, `Tensor.Residuate`, `Tensor.Closure`, and `Activations.SoftMax` using small synthetic matrices with known correct answers.

## File Boundaries

### Creates (new files)

- `tests/test_core.py` — pytest test file for core algebraic operations

### Modifies (can touch)

- none

### Read-Only (reference but do NOT modify)

- `core/tensor.py` — source for `Tensor.Join`, `Tensor.Residuate`, `Tensor.Closure` signatures
- `core/activations.py` — source for `Activations.SoftMax` signature
- `core/algebra.py` — reference for underlying algebra types
- `core/fixpoint.py` — reference for fixpoint semantics used by Closure
- `pyproject.toml` — confirms `pythonpath = ["."]` is set (so `from core.tensor import Tensor` resolves)

### Shared Contracts

- Canonical import paths: `from core.tensor import Tensor`, `from core.activations import Activations`

### Consumed Invariants

- pytest infrastructure present — `python -m pytest tests/ --collect-only` exits 0
- Canonical test import path — all test files import from `core.*` or `lattice.*`

## Tasks

- [ ] Read `core/tensor.py` in full — understand `Join`, `Residuate`, `Closure` signatures and semantics
- [ ] Read `core/activations.py` in full — understand `SoftMax` signature and semantics
- [ ] Verify `python -m pytest tests/ --collect-only` exits 0 (Sprint 4 postcondition)
- [ ] Write `tests/test_core.py` with the 13 test cases listed below
- [ ] Run `python -m pytest tests/test_core.py -v` — all tests must pass
- [ ] Run `python -m pytest tests/ --collect-only` — confirms all tests are collected without import errors

## Required Test Cases

### Join (relational composition ∨(y: A[x,y] ∧ B[y,z]))

1. `test_join_identity` — `Join(I, R) ≈ R` and `Join(R, I) ≈ R` for a simple 3×3 relation
2. `test_join_chain` — two-hop composition: `A→B`, `B→C` gives nonzero `A→C` entry
3. `test_join_zero_input` — `Join(zeros, R)` returns bottom values (no path through zero rows)
4. `test_join_shape` — `Join(A, B)` returns shape `(n, p)` when A is `(n, m)` and B is `(m, p)`

### Residuate (adjoint of Join: greatest B such that A∘B ≤ C)

5. `test_residuate_adjunction` — for any A, C: `Join(A, Residuate(A, C)) ≤ C` holds element-wise
6. `test_residuate_identity` — `Residuate(I, C) ≈ C` when composition with identity is transparent
7. `test_residuate_shape` — returns shape `(m, p)` when A is `(n, m)` and C is `(n, p)`

### Closure (iterated Join to fixpoint)

8. `test_closure_dag` — on a 4-node chain DAG (`0→1→2→3`), Closure finds all transitive paths (entry `[0,3]` becomes nonzero)
9. `test_closure_idempotent` — `Closure(Closure(R)) ≈ Closure(R)`
10. `test_closure_superset` — `Closure(R)[i,j] ≥ R[i,j]` for all i, j

### SoftMax (temperature-controlled distribution)

11. `test_softmax_sums_to_one` — `SoftMax(x, temp=1.0).sum() ≈ 1` for a 1D input
12. `test_softmax_temp_zero_argmax` — at `temp=0` (or very low), `SoftMax` returns the hard argmax (single 1.0, rest ≈ 0)
13. `test_softmax_high_temp_uniform` — at very high temp, `SoftMax` approaches a uniform distribution

## Acceptance Criteria

- [ ] `tests/test_core.py` exists and imports cleanly from `core.tensor` and `core.activations`
- [ ] `python -m pytest tests/test_core.py -v` exits 0 with all ≥13 tests passing
- [ ] Every test uses only `numpy` and the `core` package — no external corpora, no network, no large files
- [ ] No test takes more than 1 second to run

## Verification

- [ ] `python -m pytest tests/test_core.py -v` exits 0
- [ ] `python -m pytest tests/` exits 0 (full suite)
- [ ] `python -c "from core.tensor import Tensor; from core.activations import Activations; print('ok')"` prints "ok"

## Context

All four operations live in `core/`. The imports are:
- `from core.tensor import Tensor` — provides `Join`, `Residuate`, `Closure`
- `from core.activations import Activations` — provides `SoftMax`

Tests must use small synthetic matrices (3×3 or 4×4) with values derivable by hand so that expected outputs can be hard-coded without floating-point ambiguity. Use `numpy.testing.assert_allclose` with a reasonable tolerance (e.g. `atol=1e-6`).

The `pyproject.toml` from Sprint 4 sets `pythonpath = ["."]`, so imports from the project root resolve correctly. Do not add `sys.path` manipulation inside the test file.

## Agent Notes (filled during execution)

- Assigned to: —
- Started: —
- Completed: —
- Decisions made: —
- Assumptions: —
- Issues found: —
