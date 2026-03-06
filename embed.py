"""
Embedding layer. Compresses relations into a lower-dimensional space via
SVD-based Tucker decomposition (Project), allowing the tensor logic
operations from tensor.py to run on compact representations rather than
full N×N matrices — enabling the system to scale to larger graphs.
GramMatrix and EmbedSet support analogical reasoning, where similar entities borrow
inferences from one another proportionally to their embedding similarity.
"""

import numpy as np
from algebra import *
from tensor import Tensor

class Embed(Tensor):

    # Learn how to represent entities geometrically
    # derived from the structure of our tensor via svd
    def Embed(self, R, threshold=0.05):
        U, S, _ = np.linalg.svd(R, full_matrices=False)
        s0 = S[0] if S.size and S[0] != 0 else 1.0
        d = Max(int(Sum((S / s0) > threshold)), 1)
        return U[:, :d], d

    def Project(self, M, emb, temp=0.0):
        """Max-min Tucker projection: emb.T ∘ M ∘ emb"""
        return self.Join(self.Join(emb.T, M, temp), emb, temp)

    def Expand(self, EmbR, emb, temp):
        """Reconstruct: emb ∘ EmbR ∘ emb.T"""
        return self.Join(self.Join(emb, EmbR, temp), emb.T, temp)

    # represents a group of entities on a single vector by combining their embeddings
    def EmbedSet(self, indices, emb, temp):
        # acts as a thin wrapper for LogSumExp
        return self.SmoothMax(emb[indices, :], temp, axis=0)

    # ⟨x'· Sim[x,x'] ∧ R[x,y]⟩ <-> Performs a join on a matrix and its transpose
    # Finds embedding dimensions d shared between entities x and x'.
    # λx. λy. λz.  (x z)(y z)
    def GramMatrix(self, M, temp, semiring='fuzzy'):
        return self.Join(M, M.T, temp, semiring=semiring)

    def _score_concept(self, E, residual, j, temp):
        """Score the fuzzy concept rooted at column j against the current residual.
        Returns (a, b, score) — extent, intent, coverage score."""
        a = self.Join(residual[:, j:j+1].T, E, temp).squeeze()
        b = self.Join(E, a[:, None], temp).squeeze()
        covered = self.Join(b[:, None], a[None, :], temp)
        overlap = self.SmoothMin((covered, residual), temp, axis=0)
        score = Sum(overlap[overlap > Bottom])
        return a, b, score

    def _select_concept(self, E, residual, temp):
        """Scan all columns and return the (a, b) pair with the highest coverage score."""
        best_score = Bottom
        best_A, best_B = None, None
        cols = np.flatnonzero(np.any(residual > Bottom, axis=0))
        for j in cols: # range(E.shape[1]):
            a, b, score = self._score_concept(E, residual, j, temp)
            if score > best_score and score > 0:
                best_score = score
                best_A, best_B = a, b
        return best_A, best_B

    def _update_residual(self, best_A, best_B, residual, temp):
        """Subtract the selected concept's coverage from the residual."""
        covered = self.Join(best_B[:, None], best_A[None, :], temp)
        return Refutes(covered, residual)

    def Grecond(self, E, temp=None, threshold=0.5, max_iters=100):
        residual = E.copy()
        As, Bs = [], []
        for _ in range(max_iters):
            if np.all(residual <= Bottom):
                break
            best_A, best_B = self._select_concept(E, residual, temp)
            if best_A is None:
                break
            As.append(best_A)
            Bs.append(best_B)
            residual = self._update_residual(best_A, best_B, residual, temp)
        return As, Bs

    def GrecondSelect(self, R, temp=None, threshold=0.5):
        """Use Grecond intents to select representative columns from R.
        For each factor k, picks the column j* most aligned with intent As[k].
        Returns the reduced R using those columns, plus the selected indices."""
        n_rows, n_cols = R.shape
        cols = np.flatnonzero(np.any(R > Bottom, axis=0))   # skip dead columns
        tmp = np.empty(n_rows, dtype=R.dtype)              # reuse buffer (avoid np.ones alloc)
        As, _ = self.Grecond(R, temp=temp, threshold=threshold)
        if not As:
            return R, np.arange(R.shape[1])
        selected = []
        for a in As:
            scores = np.full(n_cols, Bottom, dtype=float)  # defaults so argmax is still valid
            # Score each column by min(R[:,j], a[j]) summed over rows — alignment with intent
            for j in cols:
                tmp.fill(a[j])  # replaces a[j] * np.ones(n_rows)
                scores[j] = Sum(self.SmoothMin((R[:, j], tmp), temp, axis=0))            
            selected.append(int(np.argmax(scores)))
        u_cols = np.unique(selected)
        return R[:, u_cols], u_cols
