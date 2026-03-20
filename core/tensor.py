"""
Relational operations layer. Implements three core operations on matrices:

- Join: for each output cell (x, z), finds the best intermediate node y
  by taking the min of A[x,y] and B[y,z], then the max over all y.
- Residuate: given A and C, finds the greatest B such that A composed
  with B stays within C.
- Closure: iterates Join to fixpoint, computing transitive reachability
  across the relation.

All operations are sparse — only non-zero entries are visited — and
temperature-controlled via the smooth activations from the layer below.
"""

import numpy as np
from core.algebra import *
from core.activations import Activations
from core.fixpoint import FixpointIterator

class Tensor(Activations):
    """
    Relational operations built on top of Activations.

    Inherits all temperature-controlled activation functions and adds
    matrix-level operations: Join, Residuate, Closure, and ChainJoin.

    Also provides optional witness tracking: when enabled, every
    intermediate node y that connects x to z during a Join is recorded.
    This allows logical paths through the relation to be reconstructed
    later for auditing the model's internal reasoning and identifying
    where errors originate.

    # Under review — witness tracking may become redundant as new layers are added.
    """

    def _track_witnesses(self, xs, y, zs, contrib, th):
        """
        Record intermediate node y as a witness for each (x, z) pair it connects.

        For every (x, z) pair where y's contribution exceeds the threshold,
        stores the contribution score in self._witnesses[(x, z)][y]. Called
        during Join when tracking is enabled.

        Parameters
        ----------
        xs : array of int
            Row indices (x values) active for this y.
        y : int
            The intermediate node being evaluated.
        zs : array of int
            Column indices (z values) active for this y.
        contrib : ndarray
            Contribution scores, shape (len(xs), len(zs)).
        th : float
            Threshold below which a witness is not recorded.
        """
        if not self.tracking:
            return
        i, j = np.where(contrib > th)
        for ii, jj in zip(i, j):
            key = (xs[ii], zs[jj])
            if key not in self._witnesses:
                self._witnesses[key] = {}
            self._witnesses[key][y] = contrib[ii, jj]

    def _clear_witnesses(self):
        """Reset the witness store."""
        self._witnesses = {}

    # v (y: A[x,y] ∧ B[y,z])
    def Join(self, Tensor_A, Tensor_B, temp, threshold=1e-6):
        """
        Relational composition of two matrices.

        For each output cell (x, z), finds the best intermediate node y by
        taking the min of A[x, y] and B[y, z], then taking the max over all y:

            result[x, z] = max over y of min(A[x, y], B[y, z])

        Only non-zero entries are visited for efficiency. If witness tracking
        is enabled, records each y's contribution for later path reconstruction.

        Parameters
        ----------
        Tensor_A : ndarray, shape (n, m)
            Left relation matrix.
        Tensor_B : ndarray, shape (m, p)
            Right relation matrix.
        temp : float
            Temperature passed to SmoothMin and SmoothMax.
        threshold : float, optional
            Entries below this value are treated as zero. Default is 1e-6.

        Returns
        -------
        ndarray, shape (n, p)
            The composed relation matrix.
        """

        A, B = Tensor_A, Tensor_B
        n, m = A.shape
        _, p = B.shape
        result    = np.full((n, p), Bottom, dtype=float)
        absA, absB = Abs(A), Abs(B)
        xs_list = [np.flatnonzero(absA[:, y] > threshold) for y in range(m)]
        zs_list = [np.flatnonzero(absB[y, :] > threshold) for y in range(m)]

        # we first filter for non zero entries, then loop over intermediate nodes
        # in past versions we used numpy broadcasting to compare the entire matrices at once
        # this creates O(n3) complexity. the current setup effectively avoids this by only comparing subsections of 2d matrices
        for y in range(m):
            xs = xs_list[y]
            zs = zs_list[y]
            if not (len(xs) and len(zs)): continue
            ix = np.ix_(xs, zs)

            a_col = A[:, y]
            b_row = B[y, :]
            contrib = self.SmoothMin((a_col[xs, None], b_row[None, zs]), temp, axis=0)
            old = result[ix]
            result[ix] = self.SmoothMax((old, contrib), temp, axis=0)

            self._track_witnesses(xs, y, zs, contrib, threshold) # if witness tracking is enabled, save intermediate nodes y connecting x to z

        return result
    
    # ∧ (A[x,y]-¹ α C[x,z]) <= B[y,z]
    def Residuate(self, Tensor_A, Tensor_C, temp, threshold=1e-6):
        """
        The adjoint of Join. Given A and C, finds the greatest B such that
        Join(A, B) stays within C.

        For each cell (y, z), computes the tightest upper bound that every
        row i of A places on B[y, z], using the Implies operation from
        algebra.py:

            B[y, z] = min over i of Implies(A[i, y], C[i, z])

        If Join is relational composition forward, Residuate is its inverse:
        it asks "given what we know about A and the target C, how large can
        B be?"

        Only non-zero entries are visited for efficiency.

        Parameters
        ----------
        Tensor_A : ndarray, shape (n, m)
            The left relation matrix.
        Tensor_C : ndarray, shape (n, p)
            The target relation matrix. Must share the row dimension with A.
        temp : float
            Temperature passed to SmoothMin.
        threshold : float, optional
            Entries below this value are treated as zero. Default is 1e-6.

        Returns
        -------
        ndarray, shape (m, p)
            The greatest B satisfying Join(A, B) ≤ C.

        References
        ----------
        Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
        *Information and Control*, 30, 38–48. Theorem 5.
        """

        A, C = Tensor_A, Tensor_C
        n, m = A.shape
        o, p = C.shape
        assert n == o, "Shared (row) dimension mismatch"
        # min-reduction identity is Top (e.g., 1.0)
        B = np.full((m, p), Top, dtype=float)
        # Find non-zero columns in A[i,:] and C[i,:]
        absA, absC = Abs(A), Abs(C)
        js_list = [np.flatnonzero(absA[i, :] > threshold) for i in range(n)]
        ks_list = [np.flatnonzero(absC[i, :] > threshold) for i in range(n)]

        # this code is structured the same way as its adjoint operation for the same reasons: to avoid O(n3) complexity,
        # we filter for non zero entries and than iterate over subsets of 2d matrices
        for i in range(n):
            js = js_list[i] # Active columns in A
            ks = ks_list[i] # Active columns in C
            if not (len(js) and len(ks)): continue
            ix = np.ix_(js, ks)

            a_row = A[i, js]  # (len(js),)
            c_row = C[i, ks]  # (len(ks),)
            contrib = Implies(a_row[:, None], c_row[None, :])  # (len(js), len(ks))
            old = B[ix]
            B[ix] = self.SmoothMin((old, contrib), temp, axis=0)

        return B
    
    def Closure(self, E, R=None, temp=None, max_iters=100, eps=1e-3):
        """
        Compute the transitive closure of relation E.

        Iterates Join to fixpoint: at each step, extends the current relation
        R by one hop through E, merges the result back with E, then clips it
        down using Residuate to ensure no inferred relation exceeds what E can
        justify. Repeats until the relation stops changing.

        Diagonal entries are zeroed at each step to prevent self-loops from
        accumulating.

        Parameters
        ----------
        E : ndarray, shape (n, n)
            The base relation matrix.
        R : ndarray, shape (n, n), optional
            Starting point for iteration. Defaults to a copy of E.
        temp : float, optional
            Temperature passed to Join, SmoothMax, and Residuate.
        max_iters : int, optional
            Maximum number of iterations. Default is 100.
        eps : float, optional
            Convergence threshold. Default is 1e-3.

        Returns
        -------
        ndarray, shape (n, n)
            The converged closure of E.
        """
        if R is None: R = E.copy()

        # This code was very relevant during earlier testing.
        # However it has largely been subsumed by the more complex Attention mechanism several layers above
        # That mechanism works essentially the same way. Under review whether or not this function is worth keeping.
        def _f(R, temp):
            J            = self.Join(R, E, temp)
            Rn           = self.SmoothMax((J, E), temp, axis=0)
            np.fill_diagonal(Rn, 0)
            R_allowed    = self.Residuate(Rn, E, temp)
            Rn_corrected = self.SmoothMin((Rn, R_allowed), temp, axis=0)
            return Rn_corrected, Rn  # aux = Rn (before correction)

        fp = FixpointIterator(f=_f, state0=R, eps=eps, max_iters=max_iters)
        result = fp.run()
        if fp.energy == 0:
            print(f"✓ CONVERGED at iteration {fp._iter}")
        return result

