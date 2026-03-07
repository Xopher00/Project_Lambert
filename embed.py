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

    def _concept_fixpoint(self, R, seed, temp, max_iters=20, eps=1e-3):
        active = np.flatnonzero(seed > 0)
        if len(active) == 0:
            return seed
        R_active = R[active, :]
        a = seed[active]
        for _ in range(max_iters):
            b     = np.atleast_1d(self.Residuate(R_active, a[:, None], temp).squeeze())
            a_new = np.atleast_1d(self.Residuate(R_active.T, b[:, None], temp).squeeze())
            Energy = Sum(Abs(a_new - a + eps))
            temp = -Energy / (R.size * np.mean(Log(np.clip(R, eps, 1))))  
            if  Energy < eps:
                break
            a = a_new
        return a

    def ConceptEmbed(self, R, temp, eps=1e-3):
        seen     = {}
        rep_cols = []
        for j in range(R.shape[1]):
            a   = self._concept_fixpoint(R, R[:, j], temp, eps=eps)
            n_active = int((R[:, j] > 0).sum())
            key = (tuple((a / eps).astype(int))) if n_active > 1 else (j,)
            if key not in seen:
                seen[key] = j
                rep_cols.append(j)
        emb  = R[:, rep_cols]
        EmbR = self.Project(R, emb, temp)
        return emb, EmbR, rep_cols 

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


