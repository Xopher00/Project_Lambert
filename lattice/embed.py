"""
Embedding layer. Selects a small set of representative formal concepts from a
relation matrix and uses them as embedding dimensions.

Each dimension corresponds to a formal concept — a pair of a set of entities
(extent) and a set of attributes (intent) that are mutually closed under the
Residuate operation. Selecting concepts as embedding dimensions is the insight
from Boolean matrix factorisation (Belohlavek, Outrata & Trnecka 2010): a
relation matrix can be approximated as a composition of concept columns.

ConceptEmbed is the main entry point. Project and Expand are Tucker-style
projection and reconstruction under max-min composition; both are currently under
review — callers discard the projected matrix returned by ConceptEmbed, and
Expand is not called anywhere in the active codebase. GramMatrix, Attend, and
Recall are utilities for analogical reasoning and retrieval.
"""

import numpy as np
from core.algebra import *
from core.tensor import Tensor
from core.fixpoint import FixpointIterator

class Embed(Tensor):
    """
    Relational embedding built on top of Tensor.

    Inherits all relational operations (Join, Residuate, Closure) and adds
    methods for selecting a compact set of concept-based embedding dimensions
    from a relation matrix.

    The core operation is ConceptEmbed, which selects representative columns
    from a relation matrix R by finding formal concepts — stable (extent, intent)
    pairs derived by iterating Residuate to fixpoint — and deduplicating them
    by coverage. The resulting embedding matrix has one column per representative
    concept and can be used directly as entity embeddings.
    """

    def _concept_fixpoint(self, R, seed, temp, max_iters=20, eps=1e-3, full=False):
        """
        Find the formal concept anchored at a seed column.

        Iterates Recall to fixpoint, starting from the seed entity vector.
        Each step closes the current entity vector into a tighter (extent, intent)
        pair via alternating Residuate. The stable point is the unique formal
        concept whose extent contains the seed entities (Bělohlávek, 2000).

        Only rows active in the seed (seed > 0) are included, keeping the
        operation sparse.

        Parameters
        ----------
        R : ndarray, shape (n, m)
            The relation matrix.
        seed : ndarray, shape (n,)
            Starting entity vector. Typically a column of R.
        temp : float
            Temperature passed to Residuate.
        max_iters : int, optional
            Maximum iterations before stopping. Default is 20.
        eps : float, optional
            Convergence threshold. Default is 1e-3.

        Returns
        -------
        ndarray, shape (len(active),)
            The converged attribute vector over the active entity subset.

        References
        ----------
        Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory.
        *Information Sciences*, 128, 91–103.  cite{belohlavek2000}
        """
        active = np.flatnonzero(seed > 0)
        R_active = R.copy() if full else R[active, :]
        state0   = seed.copy() if full else seed[active].copy()
        fp = FixpointIterator(
            f      = lambda a, t, R=R_active: self.Recall(a, R, t),
            state0 = state0,
            eps    = eps,
            max_iters = max_iters,
        )
        return fp.run()

    def ConceptEmbed(self, R, temp, eps=1e-3, seen=None, covered=None, rep_cols=None):
        """
        Select a compact set of representative formal concepts from a relation matrix.

        Iterates over each column of R, closes it into a formal concept via
        _concept_fixpoint, and keeps it as a representative dimension if it covers
        at least one entity not yet covered by any previously accepted concept.

        Coverage is entity-level: an entity is covered once it appears in the
        extent of any accepted concept. A column is skipped if and only if every
        entity in its extent is already covered.

        Columns whose seed activates only one entity are keyed by column index
        rather than fixpoint vector, ensuring they are never merged with other
        concepts regardless of numerical similarity.

        Parameters
        ----------
        R : ndarray, shape (n, m)
            The relation matrix. Rows are entities, columns are attributes.
        temp : float
            Temperature passed to _concept_fixpoint and Project.
        eps : float, optional
            Convergence threshold for _concept_fixpoint, and quantisation unit
            for the deduplication key. Default is 1e-3.

        Returns
        -------
        emb : ndarray, shape (n, k)
            Embedding matrix. Each column is a representative concept vector.
            k <= m is the number of selected concepts.
        EmbR : ndarray, shape (k, k)
            Tucker projection of R onto the concept basis. Currently under
            review — callers typically discard this value.
        rep_cols : list of int
            Column indices of R selected as representative concepts.

        References
        ----------
        Belohlavek, R. & Vychodil, V. (2010). Discovery of optimal factors in
        binary data via a novel method of matrix decomposition. *Journal of
        Computer and System Sciences*, 76(1), 3–20.  cite{belohlavek2010}
        """
        seen    = seen    if seen    is not None else {}
        covered = covered if covered is not None else set()
        rep_cols = rep_cols if rep_cols is not None else []
        start = len(rep_cols)
        for j in range(start, R.shape[1]):
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
        """
        Compress a relation matrix into the concept embedding space.

        Computes the Tucker-style projection under max-min composition:

            emb.T ∘ M ∘ emb

        The result is a (k, k) matrix in concept space, where k is the number
        of embedding dimensions.

        # Under review — the projected matrix is currently discarded by callers.

        Parameters
        ----------
        M : ndarray, shape (n, n)
            The relation matrix to project.
        emb : ndarray, shape (n, k)
            The embedding matrix defining the concept basis.
        temp : float, optional
            Temperature passed to each Join. Default is 0.0.

        Returns
        -------
        ndarray, shape (k, k)
            The relation matrix expressed in concept space.
        """
        return self.Join(self.Join(emb.T, M, temp), emb, temp)

    def Expand(self, M, emb, temp):
        """
        Reconstruct a relation matrix from its concept-space representation.

        Computes the inverse of Project under max-min composition:

            emb ∘ M ∘ emb.T

        The result is an (n, n) matrix in entity space. This is the approximate
        reconstruction of the original relation from its compressed form.

        # Under review — not called anywhere in the active codebase.

        Parameters
        ----------
        M : ndarray, shape (k, k)
            The relation matrix in concept space.
        emb : ndarray, shape (n, k)
            The embedding matrix defining the concept basis.
        temp : float
            Temperature passed to each Join.

        Returns
        -------
        ndarray, shape (n, n)
            The reconstructed relation matrix in entity space.
        """
        return self.Join(self.Join(emb, M, temp), emb.T, temp)

    def GramMatrix(self, M, temp):
        """
        Compute entity-entity similarity via shared embedding dimensions.

        Composes M with its transpose under max-min:

            M ∘ M.T

        Entry (x, x') in the result measures how strongly entities x and x'
        are connected through shared intermediate dimensions — the more
        embedding dimensions they both participate in, the higher the score.

        Useful for analogical reasoning: entities with high Gram scores share
        relational structure and can borrow inferences from one another.

        Parameters
        ----------
        M : ndarray, shape (n, k)
            The embedding matrix.
        temp : float
            Temperature passed to Join.

        Returns
        -------
        ndarray, shape (n, n)
            The entity-entity similarity matrix.
        """
        return self.Join(M, M.T, temp)

    def Attend(self, q, emb, temp=0.0):
        """
        One step of Hopfield-style pattern retrieval in concept space.

        Scores the query against all entities, then pulls the result back into
        concept space:

            q ∘ emb.T ∘ emb

        This is structurally identical to one update step of a modern Hopfield
        network: the first Join scores similarity between the query and each
        stored pattern; the second Join reconstructs the concept-space output
        as a weighted combination of those patterns.

        Used internally as the core operation inside the attention fixpoint loop.
        Not intended to be called directly — use the retrieval method in the
        layer above.

        Parameters
        ----------
        q : ndarray, shape (k,)
            Query vector in concept space.
        emb : ndarray, shape (n, k)
            The embedding matrix storing entity patterns.
        temp : float, optional
            Temperature passed to each Join. Default is 0.0.

        Returns
        -------
        ndarray, shape (k,)
            Updated query vector after one retrieval step.

        References
        ----------
        Ramsauer, H. et al. (2020, revised 2021). Hopfield Networks is All You Need.
        *arXiv:2008.02217*.  cite{ramsauer2021}
        """
        q2d     = q.reshape(1, -1)           # (1, d)
        scores  = self.Join(q2d, emb.T, temp)  # (1, n_entities)
        out     = self.Join(scores, emb, temp) # (1, d)
        return out.squeeze()                 # back to (d,)
    
    def Recall(self, a, emb, temp=0.0):
        """
        One step of concept closure via alternating Residuate.

        Applies two adjoint Residuate operations in sequence:

            b = Residuate(emb,   a)   # entity vector → attribute vector (intent)
            a = Residuate(emb.T, b)   # attribute vector → entity vector (extent)

        Each step tightens the (extent, intent) pair toward a formal concept.
        Iterating to fixpoint recovers the unique concept whose extent contains
        the seed entities (Bělohlávek, 2000).

        Parameters
        ----------
        a : ndarray, shape (n,)
            Current entity vector.
        emb : ndarray, shape (n, k)
            The relation or embedding matrix.
        temp : float, optional
            Temperature passed to Residuate. Default is 0.0.

        Returns
        -------
        ndarray, shape (n,)
            Updated entity vector after one closure step.
        """
        a2d = a.reshape(-1, 1)                              # (n, 1)  mirrors q.reshape(1, -1)
        b   = self.Residuate(emb, a2d, temp).reshape(-1, 1) # (k, 1)
        return self.Residuate(emb.T, b, temp).reshape(-1)   # (n,)
