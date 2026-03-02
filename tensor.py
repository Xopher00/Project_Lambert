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

    def _record_witnesses(self, xs, y, zs, contrib, th):
        i, j = np.where(contrib > th)
        return {(xs[ii], y, zs[jj]): contrib[ii, jj] for ii, jj in zip(i, j)}
    
    def _cache_witnesses(self, witnesses):
        if witnesses:
            self._witness_cache.append(witnesses)
            self._witnesses.update(witnesses)

    def clear_witness_cache(self):
        self._witness_cache = []
        self._witnesses = {}
        self._block_counter = 0 


    def Join(self, Tensor_A, Tensor_B, temp, semiring='fuzzy', threshold=1e-6):
        if semiring == "arithmetic":
            return Tensor_A @ Tensor_B, None

        A, B = Tensor_A, Tensor_B
        th   = threshold
        n, m = A.shape
        _, p = B.shape
        result    = np.full((n, p), Bottom, dtype=float)
        witnesses = {}

        for y in range(m):
            xs = np.where(Abs(A[:, y]) > th)[0]
            zs = np.where(Abs(B[y, :]) > th)[0]
            if not (len(xs) and len(zs)): continue

            a_col = A[:, y]
            b_row = B[y, :]
            contrib = self.SmoothMin((a_col[xs, None], b_row[None, zs]), temp, axis=0)
            result[np.ix_(xs, zs)] = self.SmoothMax((result[np.ix_(xs, zs)], contrib), temp, axis=0)

            witnesses.update(self._record_witnesses(xs, y, zs, contrib, th))
            self._cache_witnesses(witnesses)

        return result, witnesses
    
    def Residuate(self, Tensor_A, Tensor_C, temp, threshold=1e-6):
        """
        Right residuation / relational division (Sánchez):
            B[j,k] = min_i Implies(A[i,j], C[i,k])

        A: (n, m)
        C: (n, p)
        returns B: (m, p)

        No (n,m,p) allocation.
        """
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
    
            
    def Closure(self, E, R=None, temp=None, max_iters=100, eps=1e-3, 
                return_witnesses=False, step_fn=None, use_residuate=False):  
        if R is None: R = E.copy() if hasattr(E, 'copy') else E
        last_witnesses = None

        for k in range(max_iters):
            if step_fn is not None:
                J, last_witnesses = step_fn(R, E, temp, k)
            else:
                J, last_witnesses = self.Join(R, E, temp)

            Rn = self.SmoothMax((J, E), temp, axis=0)
            np.fill_diagonal(Rn, 0) 

            # Backward constraint propagation
            if use_residuate:
                R_allowed = self.Residuate(Rn, E, temp)
                Rn_corrected = self.SmoothMin((Rn, R_allowed), temp, axis=0)
                # Prediction error (how much correction was needed)
                prediction_error = Sum(Abs(Rn - Rn_corrected) ** 2)
                # Total free energy
                Energy = Sum(Abs(Rn_corrected - R) ** 2) + prediction_error
                Rn = Rn_corrected
            else:
                Energy = Sum(Abs(Rn - R) ** 2)

            temp = -Energy / (R.size * np.mean(Log(np.clip(R, eps, 1))))

            n_new = Sum((Rn > eps) & (R <= eps))
            print(f"iteration {k+1}: discovered {n_new} new relations, temperature is {temp:.10f}")
            if Energy <= eps:
                print(f"✓ CONVERGED at iteration {k + 1}")
                break
            R = Rn
        return (R, last_witnesses) if return_witnesses else R
    

    def ChainJoin(self, *EmbRs, temp=0.0, semiring='fuzzy'):
        result = EmbRs[0]
        for EmbR in EmbRs[1:]:
            result = self.Join(result, EmbR, temp=temp, semiring=semiring)
        return result    

    def Backward(self, E, temp=None, max_iters=100):
        return self.Closure(E.T, temp=temp, max_iters=max_iters)   