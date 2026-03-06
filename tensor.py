"""
Relational operations layer. Implements Domingos' core tensor logic
operations — Join, Closure, and Backward — grounded in UA rather than
arithmetic. Join(X, Y) is ∨⟨y· X[x,y] ∧ Y[y,z]⟩: for each (x,z) pair,
find the best intermediate y using SmoothMin (∧) then SmoothMax (∨). The
scores tensor produced by Join records how much each y contributed, and
is the primary carrier of reasoning provenance throughout the system.
Closure iterates Join to fixpoint to compute transitive reachability;
Backward transposes the relation first, giving reverse reachability needed
for witness validation.
"""

import numpy as np
from algebra import *
from activations import Activations

class Tensor(Activations):       

    def _track_witnesses(self, xs, y, zs, contrib, th):
        """Record witnesses for current y and merge into accumulated witnesses"""
        if not self.tracking:
            return
        i, j = np.where(contrib > th)
        for ii, jj in zip(i, j):
            key = (xs[ii], zs[jj])
            if key not in self._witnesses:
                self._witnesses[key] = {}
            self._witnesses[key][y] = contrib[ii, jj]

    def _clear_witnesses(self):
        self._witnesses = {}

    # v (y: A[x,y] ∧ B[y,z])
    def Join(self, Tensor_A, Tensor_B, temp, semiring='fuzzy', threshold=1e-6):
        if semiring == "arithmetic":
            return Tensor_A @ Tensor_B, None

        A, B = Tensor_A, Tensor_B
        th   = threshold
        n, m = A.shape
        _, p = B.shape
        result    = np.full((n, p), Bottom, dtype=float)

        for y in range(m):
            xs = np.where(Abs(A[:, y]) > th)[0]
            zs = np.where(Abs(B[y, :]) > th)[0]
            if not (len(xs) and len(zs)): continue

            a_col = A[:, y]
            b_row = B[y, :]
            contrib = self.SmoothMin((a_col[xs, None], b_row[None, zs]), temp, axis=0)
            result[np.ix_(xs, zs)] = self.SmoothMax((result[np.ix_(xs, zs)], contrib), temp, axis=0)

            self._track_witnesses(xs, y, zs, contrib, th)

        return result
    
    # ∧ (A[x,y]-¹ α C[x,z]) <= B[y,z]
    def Residuate(self, Tensor_A, Tensor_C, temp, threshold=1e-6):

        A, C = Tensor_A, Tensor_C
        n, m = A.shape
        o, p = C.shape
        assert n == o, "Shared (row) dimension mismatch"
        # min-reduction identity is Top (e.g., 1.0)
        B = np.full((m, p), Top, dtype=float)

        for i in range(n):
            # Find non-zero columns in A[i,:] and C[i,:]
            js = np.where(Abs(A[i, :]) > threshold)[0]  # Active columns in A
            ks = np.where(Abs(C[i, :]) > threshold)[0]  # Active columns in C
            if not (len(js) and len(ks)): continue

            a_row = A[i, js]  # (len(js),)
            c_row = C[i, ks]  # (len(ks),)
            contrib = Implies(a_row[:, None], c_row[None, :])  # (len(js), len(ks))
            B[np.ix_(js, ks)] = self.SmoothMin((B[np.ix_(js, ks)], contrib), temp, axis=0)

        return B
    
    def Closure(self, E, R=None, temp=None, max_iters=100, eps=1e-3):  
        if R is None: R = E.copy()

        for k in range(max_iters):
            J = self.Join(R, E, temp)
            Rn = self.SmoothMax((J, E), temp, axis=0)
            np.fill_diagonal(Rn, 0) 

            # Backward constraint propagation
            R_allowed = self.Residuate(Rn, E, temp)
            Rn_corrected = self.SmoothMin((Rn, R_allowed), temp, axis=0)

            dynamic_error = Sum(Abs(Rn_corrected - R) ** 2)  # ε_x
            sensory_error = Sum(Abs(Rn - Rn_corrected) ** 2) # ε_y 
            Energy = dynamic_error + sensory_error
            temp = -Energy / (R.size * np.mean(Log(np.clip(R, eps, 1))))                     

            n_new = Sum((Rn > eps) & (R <= eps))
            # print(f"iteration {k+1}: discovered {n_new} new relations, temperature is {temp:.10f}")
            if Energy <= eps:
                print(f"✓ CONVERGED at iteration {k + 1}")
                break

            R = Rn_corrected

        return R

    def ChainJoin(self, *EmbRs, temp=0.0, semiring='fuzzy'):
        result = EmbRs[0]
        for EmbR in EmbRs[1:]:
            result = self.Join(result, EmbR, temp=temp, semiring=semiring)
        return result    

    def Backward(self, E, temp=None, max_iters=100):
        return self.Closure(E.T, temp=temp, max_iters=max_iters)   