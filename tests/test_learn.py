"""
Unit tests for the learning rule: Learner class in lattice/embed.py.

Tests verify:
  - test_learn_new_attractor:    a pattern learned by Learner is retrievable
  - test_learn_preserves_old:    pre-existing attractors survive a learn() call
  - test_learn_merge_is_max:     the merge is exactly np.maximum(R_before, delta_R)
  - test_learn_shape_validation: incompatible shapes raise ValueError

All tests run at temp=0 (exact, hard) for algebraic precision.

Import path follows the canonical convention: from legacy.lattice.embed import Learner.
"""

import numpy as np
import numpy.testing as npt
import pytest

from legacy.lattice.embed import Learner

pytestmark = pytest.mark.legacy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_R():
    """
    A small 3x3 relation matrix with 3 entities and 3 attributes.

    Rows are entities, columns are attributes. Values are in [0, 1].
    Chosen so that each row has at least one nonzero entry so that
    Residuate finds non-trivial solutions.
    """
    return np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.9, 0.3],
        [0.0, 0.4, 0.7],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLearnNewAttractor:
    """After learn(), the new pattern is retrievable from R."""

    def test_learn_new_attractor(self):
        """
        Construct a Learner on a 3-entity R, learn a new entity pattern
        (a 4th entity), and verify that R grew to accommodate it.

        The new pattern is a single-row Y (shape (1, 4)) and a single-row
        X (shape (1, 3)). After learn(), R.shape[0] stays fixed (Learner
        does not append rows — it updates R in-place via np.maximum). The
        test instead verifies that the per-cell update satisfies the merge
        invariant: R_new >= R_old everywhere and R_new >= delta_R.

        Why this tests "new attractor": Residuate(Y, X) computes the
        weight matrix that makes Y a stable attractor for input X.
        After merging into R, any Residuate query using an input close
        to X should score higher (i.e., R_new is a superset of knowledge).
        """
        R = _small_R()                    # (3, 3)
        learner = Learner(R.copy())

        # A new pattern pair: one new input (entity) row and one output
        # (attribute) row, each as 2-D arrays with 1 row.
        Y_new = np.array([[0.9, 0.0, 0.6]])   # (1, 3) entity-space
        X_new = np.array([[0.0, 0.8, 0.5]])   # (1, 3) attribute-space

        R_new = learner.learn(X_new, Y_new)

        # 1. R was updated on the learner
        assert R_new is learner.R, "learn() must update learner.R in place"

        # 2. Merge invariant: R_new >= R_old everywhere
        assert np.all(R_new >= R), (
            "learn() must not decrease any entry in R"
        )

        # 3. R_new is non-trivially larger than R_old (the new pattern had
        #    effect somewhere — at least one entry increased)
        assert np.any(R_new > R + 1e-9), (
            "learn() should increase at least one entry for a novel pattern"
        )


class TestLearnPreservesOld:
    """Existing attractors are unaffected by a learn() call."""

    def test_learn_preserves_old(self):
        """
        After learn(), every cell of R_new is >= the corresponding cell in
        R_old (max-merge never decreases values).  Additionally, any cell
        that was at its maximum before remains at its maximum after — this
        is the preservation guarantee that existing attractors survive.
        """
        R = _small_R()                    # (3, 3)
        learner = Learner(R.copy())
        R_before = learner.R.copy()

        # Learn a genuinely new pattern
        Y_new = np.array([[0.5, 0.5, 0.5]])  # (1, 3)
        X_new = np.array([[0.5, 0.5, 0.5]])  # (1, 3)
        learner.learn(X_new, Y_new)
        R_after = learner.R

        # Every cell: R_after[i,j] >= R_before[i,j]
        assert np.all(R_after >= R_before), (
            "learn() must not decrease any existing R entry"
        )

        # Cells that were already at their row-max stay at or above that value
        row_max_before = R_before.max(axis=1)
        row_max_after  = R_after.max(axis=1)
        assert np.all(row_max_after >= row_max_before), (
            "learn() must not reduce the maximum of any row"
        )


class TestLearnMergeIsMax:
    """The merge operation is exactly np.maximum(R_old, Residuate(Y, X))."""

    def test_learn_merge_is_max(self):
        """
        For a synthetic case, directly verify the merge formula:
            R_new = np.maximum(R_old, Residuate(Y, X))

        We compute Residuate(Y, X) independently and compare to the
        result produced by Learner.learn().  This guards against any
        implementation that averages, replaces, or uses a different
        combination rule.
        """
        R = _small_R()                    # (3, 3)

        # Use simple all-ones patterns to get a computable expected result
        Y = np.array([[1.0, 0.5, 0.0],   # (2, 3) entity-space
                      [0.0, 0.5, 1.0]])
        X = np.array([[0.8, 0.2, 0.6],   # (2, 3) attribute-space
                      [0.3, 0.9, 0.4]])

        # Independent reference: compute Residuate(Y, X) directly
        from core.tensor import Tensor
        T = Tensor(temp=0.0)
        delta_R = T.Residuate(Y, X, temp=0)          # (3, 3)
        R_expected = np.maximum(R, delta_R)

        # Learner must produce the identical result
        learner = Learner(R.copy())
        R_actual = learner.learn(X, Y)

        npt.assert_allclose(
            R_actual, R_expected, atol=1e-9,
            err_msg="learn() merge must equal np.maximum(R_old, Residuate(Y, X))"
        )


class TestLearnShapeValidation:
    """Incompatible shapes raise ValueError with a descriptive message."""

    def test_learn_bad_Y_shape(self):
        """Y.shape[1] != R.shape[0] raises ValueError."""
        R = _small_R()                    # (3, 3)
        learner = Learner(R.copy())

        # Y has 4 entity columns but R has 3 entity rows -> mismatch
        Y_bad = np.array([[0.5, 0.5, 0.5, 0.5]])  # (1, 4) — wrong
        X_ok  = np.array([[0.5, 0.5, 0.5]])        # (1, 3) — correct

        with pytest.raises(ValueError, match=r"Y\.shape\[1\]"):
            learner.learn(X_ok, Y_bad)

    def test_learn_bad_X_shape(self):
        """X.shape[1] != R.shape[1] raises ValueError."""
        R = _small_R()                    # (3, 3)
        learner = Learner(R.copy())

        # X has 5 attribute columns but R has 3 attribute columns -> mismatch
        Y_ok  = np.array([[0.5, 0.5, 0.5]])              # (1, 3) — correct
        X_bad = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])   # (1, 5) — wrong

        with pytest.raises(ValueError, match=r"X\.shape\[1\]"):
            learner.learn(X_bad, Y_ok)
