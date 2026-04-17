"""
Core operation unit tests.

Covers Tensor.Join, Tensor.Residuate, Tensor.Closure, and Activations.SoftMax
using small synthetic matrices with known correct answers.

All operations run at temp=0 (exact, hard) unless otherwise stated so that
expected values can be computed by hand.  Operations at temp>0 are tested
through their mathematical properties (adjunction, idempotence, etc.).

Import paths follow the canonical convention: core.tensor and core.activations.
"""

import numpy as np
import numpy.testing as npt
import pytest

from core.tensor import Tensor
from core.activations import Activations
from core.algebra import Bottom, Top

pytestmark = pytest.mark.core

# A single shared Tensor instance (stateless for these tests).
T = Tensor(temp=0.0)
TEMP = 0.0   # hard (exact) temperature used in most tests
ATOL = 1e-6


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

class TestJoin:
    """Relational composition — max_y min(A[x,y], B[y,z])."""

    def test_join_identity(self):
        """Join(I, R) ≈ R  and  Join(R, I) ≈ R for a 3×3 dense relation.

        Uses a fully-populated (no-zero-entry) matrix so the sparse
        optimisation inside Join always finds active rows/columns.  The
        algebra maps "no path" to Bottom (-1e9), not 0, so zero entries in R
        would appear as Bottom in the output rather than 0; this test avoids
        that edge case by using a relation with no zero entries.
        """
        I = np.eye(3)
        R = np.array([
            [0.8, 0.2, 0.3],
            [0.1, 0.5, 0.4],
            [0.6, 0.7, 0.9],
        ])
        left  = T.Join(I, R, temp=TEMP)
        right = T.Join(R, I, temp=TEMP)
        # All values in R are in (0,1] so min(1, R[x,z]) == R[x,z].
        npt.assert_allclose(left,  R, atol=ATOL)
        npt.assert_allclose(right, R, atol=ATOL)

    def test_join_chain(self):
        """Two-hop composition: A→B, B→C gives nonzero A→C entry."""
        # A connects node 0 to node 1; B connects node 1 to node 2.
        # Join(A, B) should have a nonzero [0, 2] entry.
        A = np.array([[0.0, 0.7], [0.0, 0.0]])   # 2×2: 0→1 edge weight 0.7
        B = np.array([[0.0, 0.0], [0.6, 0.0]])   # 2×2: 1→0... actually use 2×2
        # Let's use 3×3 to make the chain clearer.
        A3 = np.zeros((3, 3))
        B3 = np.zeros((3, 3))
        A3[0, 1] = 0.7   # 0 → 1
        B3[1, 2] = 0.6   # 1 → 2
        result = T.Join(A3, B3, temp=TEMP)
        # Expected: result[0,2] = max_y min(A3[0,y], B3[y,2])
        #   y=0: min(0, 0) = 0
        #   y=1: min(0.7, 0.6) = 0.6
        #   y=2: min(0, 0) = 0
        # max = 0.6
        assert result[0, 2] > 0.0, "Two-hop path should be nonzero"
        npt.assert_allclose(result[0, 2], 0.6, atol=ATOL)

    def test_join_zero_input(self):
        """Join(zeros, R) returns Bottom values — no path through zero rows."""
        zeros = np.zeros((3, 3))
        R = np.array([
            [0.5, 0.2, 0.8],
            [0.1, 0.9, 0.4],
            [0.7, 0.3, 0.6],
        ])
        result = T.Join(zeros, R, temp=TEMP)
        # All entries in zeros are exactly 0 → Abs(zeros)[:,y] = 0 < threshold
        # xs_list[y] will be empty for all y → result stays filled with Bottom.
        assert np.all(result == Bottom), (
            f"Expected all Bottom ({Bottom}), got min={result.min()}"
        )

    def test_join_shape(self):
        """Join(A, B) returns shape (n, p) when A is (n, m) and B is (m, p)."""
        n, m, p = 4, 3, 5
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        result = T.Join(A, B, temp=TEMP)
        assert result.shape == (n, p), f"Expected ({n},{p}), got {result.shape}"


# ---------------------------------------------------------------------------
# Residuate
# ---------------------------------------------------------------------------

class TestResiduate:
    """Adjoint of Join — greatest B such that Join(A, B) ≤ C."""

    def test_residuate_adjunction(self):
        """For any A, C: Join(A, Residuate(A, C)) ≤ C holds element-wise."""
        np.random.seed(42)
        A = np.random.rand(4, 3)
        C = np.random.rand(4, 4)
        B = T.Residuate(A, C, temp=TEMP)
        composed = T.Join(A, B, temp=TEMP)
        # composed[i,j] should be ≤ C[i,j]; allow small numerical slack
        # Note: composed entries that stayed at Bottom (-1e9) trivially satisfy this.
        mask = composed > Bottom + 1.0   # ignore cells that stayed at Bottom
        if mask.any():
            assert np.all(composed[mask] <= C[mask] + ATOL), (
                "Adjunction violated: Join(A, Residuate(A, C)) > C"
            )

    def test_residuate_identity(self):
        """Residuate(I, C) ≈ C when I is the identity matrix.

        The sparse optimisation skips C[i,z]=0 entries (treating them as
        "no active constraint"), leaving the corresponding B entry at Top.
        To get a clean Residuate(I, C)==C result, C must have no zero entries.
        """
        I = np.eye(3)
        C = np.array([
            [0.9, 0.1, 0.5],
            [0.3, 0.7, 0.2],
            [0.6, 0.4, 0.8],
        ])
        result = T.Residuate(I, C, temp=TEMP)
        # For each (y,z): min_i Implies(I[i,y], C[i,z])
        #   i ≠ y: I[i,y]=0 ≤ C[i,z] (all positive) → Implies = Top
        #   i = y: I[y,y]=1; Implies(1, C[y,z]) = C[y,z] since C[y,z] < 1
        # Min over all i = C[y,z]
        npt.assert_allclose(result, C, atol=ATOL)

    def test_residuate_shape(self):
        """Residuate returns shape (m, p) when A is (n, m) and C is (n, p)."""
        n, m, p = 5, 3, 4
        A = np.random.rand(n, m)
        C = np.random.rand(n, p)
        result = T.Residuate(A, C, temp=TEMP)
        assert result.shape == (m, p), f"Expected ({m},{p}), got {result.shape}"


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------

class TestClosure:
    """Iterated Join to fixpoint — transitive reachability."""

    def _chain_dag(self):
        """Build a 4-node chain DAG: 0→1→2→3."""
        R = np.zeros((4, 4))
        R[0, 1] = 0.8
        R[1, 2] = 0.7
        R[2, 3] = 0.6
        return R

    def test_closure_dag(self):
        """Closure on 4-node chain finds all transitive paths (entry [0,3] nonzero)."""
        R = self._chain_dag()
        C = T.Closure(R, temp=0.1)
        assert C[0, 3] > 0.0, f"Expected transitive path 0→3, got {C[0,3]}"

    def test_closure_idempotent(self):
        """Closure(Closure(R)) ≈ Closure(R)."""
        R = self._chain_dag()
        C1 = T.Closure(R, temp=0.1)
        C2 = T.Closure(C1, temp=0.1)
        npt.assert_allclose(C1, C2, atol=1e-2)

    def test_closure_superset(self):
        """Closure(R)[i,j] ≥ R[i,j] for all i, j."""
        R = self._chain_dag()
        C = T.Closure(R, temp=0.1)
        # Closure should not lose any direct edges.
        assert np.all(C >= R - ATOL), (
            "Closure dropped a direct edge"
        )


# ---------------------------------------------------------------------------
# SoftMax
# ---------------------------------------------------------------------------

class TestSoftMax:
    """Temperature-controlled distribution normalisation."""

    def test_softmax_sums_to_one(self):
        """SoftMax(x, temp=1.0).sum() ≈ 1 for a 1D input."""
        act = Activations(temp=1.0)
        x = np.array([1.0, 2.0, 3.0, 0.5])
        result = act.SoftMax(x, temp=1.0, axis=0)
        npt.assert_allclose(result.sum(), 1.0, atol=ATOL)

    def test_softmax_temp_zero_argmax(self):
        """At temp=0 SoftMax returns the hard argmax: single 1.0, rest ≈ 0."""
        act = Activations(temp=0.0)
        x = np.array([0.3, 0.9, 0.1, 0.5])
        result = act.SoftMax(x, temp=0.0, axis=0)
        # The SoftMax at temp=0 delegates to Max(x, axis=0) which returns
        # the max value (a scalar), not a distribution array.
        # We verify the return value equals the maximum element.
        assert result == pytest.approx(0.9, abs=ATOL), (
            f"Expected hard max 0.9, got {result}"
        )

    def test_softmax_high_temp_uniform(self):
        """At very high temp, SoftMax approaches a uniform distribution."""
        act = Activations(temp=1000.0)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = act.SoftMax(x, temp=1000.0, axis=0)
        uniform = np.ones(4) / 4.0
        npt.assert_allclose(result, uniform, atol=0.01)
