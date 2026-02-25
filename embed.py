"""
Embedding layer. Compresses relations into a lower-dimensional space via
SVD-based Tucker decomposition (Project), allowing the tensor logic
operations from tensor.py to run on compact representations rather than
full N×N matrices — enabling the system to scale to larger graphs.
Extracy inverts this to recover entity-space relations, and
 ReEmbed snaps intermediate closure states back to clean
relations to prevent error accumulation across iterations. GramMatrix and
EmbedSet lay the groundwork for future analogical reasoning, where similar
entities will be able to borrow inferences from one another proportionally
to their embedding similarity.
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

    # Tucker projection: maps a relation between spaces using emb as the bridge.
    # Entity → Embedding: Project(R,    emb)    →  emb.T @ R    @ emb
    # Embedding → Entity: Project(EmbR, emb.T)  →  emb   @ EmbR @ emb.T
    def Project(self, M, emb):
        if hasattr(M, 'tocsr'):          # scipy sparse
            AE = M.dot(emb)
        else:
            AE = M @ emb                 # dense numpy
        return emb.T @ AE

    # Extract relationship, keep track of products to see how model reached conclusion
    def Expand(self, EmbR, emb, temp):
        logits = self.Project(EmbR, emb.T)
        return self.SoftMax(logits, temp, axis=-1)

    # represents a group of entities on a single vector by combining their embeddings
    def EmbedSet(self, indices, emb, temp):
        # acts as a thin wrapper for LogSumExp
        return self.SmoothMax(emb[indices, :], temp, axis=0)

    # ⟨x'· Sim[x,x'] ∧ R[x,y]⟩ <-> Performs a join on a matrix and its transpose
    # Finds embedding dimensions d shared between entities x and x'.
    # λx. λy. λz.  (x z)(y z)
    def GramMatrix(self, M, temp, semiring='fuzzy'):
        return self.Join(M, M.T, temp, semiring=semiring)    