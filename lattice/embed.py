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
from lattice.coder import PathCoder
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

    def __init__(self):
        legs=[
            lambda x, rel, t: self.Join(x.reshape(1, -1), rel.T, t),
            lambda x, rel, t: self.Join(x.reshape(1, -1), rel, t).squeeze(),
            lambda x, rel, t: self.Residuate(rel, x.reshape(-1, 1), t).reshape(-1, 1),
            lambda x, rel, t: self.Residuate(rel.T, x.reshape(-1, 1), t).reshape(-1),
        ]
        self.coder = PathCoder(legs)

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
        f = self.coder.op("sesd")
        return f(q, emb, temp)
    
    def hop(self, q: np.ndarray, EmbR: np.ndarray, temp: float) -> np.ndarray:
        """
        One left-Kan step in concept space: Join(q[np.newaxis,:], EmbR)[0].

        Advances a concept-space query vector one relational hop forward
        using a concept-to-concept relation matrix EmbR. This is the
        existential (Σ_R) direction of the Kan adjunction:

            Σ_R(q) = Join(q, EmbR)

        The result is a new concept-space vector representing all concepts
        reachable from q via EmbR.

        Parameters
        ----------
        q : ndarray, shape (k,)
            Concept-space query vector. Must satisfy q.shape[0] == EmbR.shape[0].
        EmbR : ndarray, shape (k, k)
            Concept-to-concept relation matrix (Tucker core). Must be square:
            EmbR.shape[0] == EmbR.shape[1].
        temp : float
            Temperature passed to Join.

        Returns
        -------
        ndarray, shape (k,)
            Updated concept-space vector after one relational hop.

        Raises
        ------
        ValueError
            If q.shape[0] != EmbR.shape[0] (dimension mismatch) or
            EmbR.shape[0] != EmbR.shape[1] (EmbR is not square).

        References
        ----------
        Domingos, P. (2025). Tensor logic. — Multi-hop query chains as
        compositions of Tucker-core einsums.  cite{domingos2025}
        """
        if EmbR.ndim != 2 or EmbR.shape[0] != EmbR.shape[1]:
            raise ValueError(
                f"EmbR must be square (shape (k, k)). "
                f"Got EmbR.shape={EmbR.shape}."
            )
        k = EmbR.shape[0]
        if q.shape != (k,):
            raise ValueError(
                f"q.shape must equal (EmbR.shape[0],) = ({k},). "
                f"Got q.shape={q.shape}."
            )
        result = self.Join(q[np.newaxis, :], EmbR, temp=temp)  # (1, k)
        return result[0]                                        # (k,)

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
        f = self.coder("pepd")
        return f(a, emb, temp)


class Learner(Embed):
    """
    Algebraic learning rule for Lambert relation matrices.

    Wraps an existing relation matrix R and provides a learn() method
    that updates R using the FLBAM weight construction rule
    (Belohlavek 2000, Algorithm eq. 2; Sussner & Valle 2006):

        W = Y ⊗ₙ Xᵀ  =  Residuate(Y, X)

    Given a set of (entity_vector, attribute_vector) pattern pairs stored
    as rows of Y and X respectively, Residuate(Y, X) constructs the
    relation matrix contribution that stores all of them as stable
    attractors.  The contribution is merged into R via elementwise max:

        R_new = np.maximum(R_old, Residuate(Y, X))

    This is a join (∨) over stored pattern pairs — the algebraic
    construction from Belohlavek (2000) eq. 2.  Replacing R would lose
    existing attractors.  Averaging would violate the algebraic semantics
    (the construction rule is not defined for averages).

    Limitation: learn() updates R only. Call ConceptEmbed on the updated
    R to refresh emb and EmbR so that subsequent queries reflect the new
    knowledge.

    Parameters
    ----------
    R : ndarray, shape (n_entities, n_attributes)
        The initial relation matrix. Stored as self.R; updated in-place
        by each learn() call.

    References
    ----------
    Belohlavek, R. (2000). Fuzzy logical bidirectional associative memory.
    *Information Sciences*, 128, 91–103. — Algorithm eq. 2: I_ij = ∨_p
    A^p(g_i) ⊗ B^p(m_j); construction of the weight matrix from stored
    pattern pairs.  cite{belohlavek2000}

    Sussner, P., & Valle, M. E. (2006). Implicative fuzzy associative
    memories. *IEEE Transactions on Fuzzy Systems*, 14(6), 791–807. —
    W = Y ⊗ₙ Xᵀ = Residuate(Y, X); construction rule identical to
    Lambert's Residuate.  cite{sussner2006}
    """

    def __init__(self, R: np.ndarray):
        super().__init__()
        self.R = R.copy()

    def learn(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Construct a weight update from pattern pairs and merge into R.

        Implements the FLBAM construction rule (Belohlavek 2000 eq. 2):

            delta_R = Residuate(Y, X)
            R_new   = np.maximum(R_old, delta_R)

        The argument order follows W = Y ⊗ₙ Xᵀ = Residuate(Y, X) from
        Belohlavek (2000): Y carries the entity-space (row) patterns and
        X carries the attribute-space (row) patterns.

        Limitation: this method updates self.R only. Call ConceptEmbed
        on the updated R to refresh emb and EmbR so that subsequent
        queries reflect the new knowledge.

        Parameters
        ----------
        X : ndarray, shape (n_patterns, n_attributes)
            Output patterns in attribute space.  X.shape[1] must equal
            self.R.shape[1] (the number of attributes).
        Y : ndarray, shape (n_patterns, n_entities)
            Input patterns in entity space.  Y.shape[1] must equal
            self.R.shape[0] (the number of entities).

        Returns
        -------
        ndarray, shape (n_entities, n_attributes)
            The updated relation matrix (also stored as self.R).

        Raises
        ------
        ValueError
            If Y.shape[1] != self.R.shape[0] or X.shape[1] != self.R.shape[1],
            with a message that includes the expected and actual shapes.
        """
        n_entities, n_attributes = self.R.shape

        if Y.shape[1] != n_entities:
            raise ValueError(
                f"Y.shape[1] must equal R.shape[0] (n_entities). "
                f"Expected Y.shape[1]={n_entities}, got Y.shape[1]={Y.shape[1]}. "
                f"R.shape={self.R.shape}, Y.shape={Y.shape}."
            )
        if X.shape[1] != n_attributes:
            raise ValueError(
                f"X.shape[1] must equal R.shape[1] (n_attributes). "
                f"Expected X.shape[1]={n_attributes}, got X.shape[1]={X.shape[1]}. "
                f"R.shape={self.R.shape}, X.shape={X.shape}."
            )

        # Construct the weight update: Residuate(Y, X) at T=0 (exact construction).
        # Y: (n_patterns, n_entities), X: (n_patterns, n_attributes)
        # Residuate(A, C) returns (A.shape[1], C.shape[1]) = (n_entities, n_attributes)
        delta_R = self.Residuate(Y, X, temp=0)

        # Merge via elementwise max (join over stored pattern pairs, Belohlavek 2000 eq. 2).
        self.R = np.maximum(self.R, delta_R)
        return self.R
