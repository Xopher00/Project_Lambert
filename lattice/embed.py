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
        self.coder = PathCoder([
            lambda x, y, temp: self.Join(x.reshape(1, -1), y.T, temp),
            lambda x, y, temp: self.Join(x.reshape(1, -1), y, temp).squeeze(),
            lambda x, y, temp: self.Residuate(y, x.reshape(-1, 1), temp).reshape(-1, 1),
            lambda x, y, temp: self.Residuate(y.T, x.reshape(-1, 1), temp).reshape(-1),
        ])

        # Hopfield-style pattern retrieval: q ∘ emb.T ∘ emb
        # Scores the query against stored patterns then reconstructs in concept space.
        # One step of a modern Hopfield update (Ramsauer et al. 2021, cite{ramsauer2021}).
        self.Attend = self.coder.op("realize propagate")

        # Concept closure via alternating Residuate: abstract(support(a, emb), emb)
        # Tightens (extent, intent) toward a formal concept each step.
        # Iterating to fixpoint recovers the unique concept containing the seed
        # (Bělohlávek, 2000, cite{belohlavek2000}).
        self.Recall = self.coder.op("abstract support")

        # Left-Kan hop in concept space: Σ_R(q) = Join(q, EmbR)
        # Advances q one relational step forward through the Tucker-core EmbR.
        # (Domingos 2025, cite{domingos2025})
        self.Hop = self.coder.op("propagate")

        # Tucker-style projection: emb.T ∘ M ∘ emb  → (k, k) in concept space.
        # Under review — projected matrix currently discarded by callers.
        self.Project = self.coder.op("propagate:symmetry:converse realize")

        # Tucker-style reconstruction: emb ∘ M ∘ emb.T  → (n, n) in entity space.
        # Under review — not called anywhere in the active codebase.
        self.Expand = self.coder.op("propagate:symmetry realize")

        # Entity-entity similarity: M ∘ M.T
        # Measures shared embedding participation for analogical reasoning.
        self.GramMatrix = self.coder.op("realize:diagonal:converse")

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
