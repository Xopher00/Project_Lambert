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
from fixpoint import FixpointIterator

class Embed(Tensor):

    def _concept_fixpoint(self, R, seed, temp, max_iters=20, eps=1e-3):
        active = np.flatnonzero(seed > 0)
        R_active = R[active, :]
        def _f(a, t, R_active=R_active):
            b     = np.atleast_1d(self.Residuate(R_active, a[:, None], t).squeeze())
            a_new = np.atleast_1d(self.Residuate(R_active.T, b[:, None], t).squeeze())
            return a_new
        def _energy(new, old, aux):
            return float(Sum(Abs(new - old + eps)))
        fp = FixpointIterator(f=_f, energy_fn=_energy, 
            state0=seed[active].copy(), eps=eps, max_iters=max_iters)
        return fp.run()

    def ConceptEmbed(self, R, temp, eps=1e-3):
        seen     = {}
        rep_cols = []
        covered  = set()
        for j in range(R.shape[1]):
            a   = self._concept_fixpoint(R, R[:, j], temp, eps=eps)
            n_active = int((R[:, j] > 0).sum())
            key = (tuple((a / eps).astype(int))) if n_active > 1 else (j,)
            extent   = set(np.flatnonzero(R[:, j] > 0))
            if key not in seen or not extent.issubset(covered):
                seen[key] = j
                rep_cols.append(j)
                covered.update(extent)
        emb  = R[:, rep_cols]
        EmbR = self.Project(R, emb, temp)
        return emb, EmbR, rep_cols 

    def Project(self, M, emb, temp=0.0):
        """Max-min Tucker projection: emb.T ∘ M ∘ emb"""
        return self.Join(self.Join(emb.T, M, temp), emb, temp)

    def Expand(self, M, emb, temp):
        """Reconstruct: emb ∘ EmbR ∘ emb.T"""
        return self.Join(self.Join(emb, M, temp), emb.T, temp)

    # represents a group of entities on a single vector by combining their embeddings
    def EmbedSet(self, indices, emb, temp):
        # acts as a thin wrapper for LogSumExp
        return self.SmoothMax(emb[indices, :], temp, axis=0)

    # ⟨x'· Sim[x,x'] ∧ R[x,y]⟩ <-> Performs a join on a matrix and its transpose
    # Finds embedding dimensions d shared between entities x and x'.
    # λx. λy. λz.  (x z)(y z)
    def GramMatrix(self, M, temp, semiring='fuzzy'):
        return self.Join(M, M.T, temp, semiring=semiring)
    
    def Attend(self, q, emb, temp=0.0):
        """Hopfield retrieval: q ∘ emb.T ∘ emb"""
        q2d     = q.reshape(1, -1)           # (1, d)
        scores  = self.Join(q2d, emb.T, temp)  # (1, n_entities)
        out     = self.Join(scores, emb, temp) # (1, d)
        return out.squeeze()                 # back to (d,)
