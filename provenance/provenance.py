"""
Proof extraction layer. Takes the scores tensor from Join and the backward
reachability matrix from Closure, and combines them — via Witnesses — to
identify which intermediate nodes y actually justify each (u,v) conclusion
in both directions simultaneously. Prove then walks this witness tensor
recursively to construct a human-readable proof tree, and Query is the
main entry point that runs the full pipeline and returns a formatted
explanation. This layer is what makes the system's reasoning transparent
by design: the proofs are not post-hoc explanations but a direct readout
of the tensor operations themselves.
"""

import numpy as np
from .tree import Tree
import networkx as nx
from core.tensor import Tensor as t
from .audit import format_proof
from core.algebra import Implies
from collections import deque

class Provenance(t):
    """
    Proof extraction over relational tensors.

    Extends Tensor with the ability to reconstruct human-readable proof
    trees explaining why a given (src, dst) pair is reachable in a
    relation. Witnesses records which intermediate nodes y justified each
    (u, v) connection during Join; Prove walks those witnesses recursively
    to build a proof tree; Query is the main entry point that runs the
    full pipeline.

    Proofs are a direct readout of the tensor operations — not post-hoc
    explanations but an audit trail of the relational reasoning itself.

    # Under review. As lattice traversal matures in CategoryExplorer,
    # this layer may become redundant — lattice paths through the concept
    # hierarchy could replace explicit witness-based proof trees.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._R_star = None
        self._R_source = None

    def witness_capacity(self, threshold=0.05):
        """
        Summarise the current witness store.

        Returns the number of (u, v) pairs with recorded witnesses, the
        average number of witnesses per pair above threshold, and the
        maximum branching factor. Useful for inspecting how much proof
        structure is available before calling Prove.
        """
        if not self._witnesses:
            return {"pairs": 0, "avg_branching": 0.0, "max_branching": 0}
        
        branch_counts = [
            sum(1 for score in witnesses.values() if score > threshold)
            for witnesses in self._witnesses.values()
        ]
        
        return {
            "pairs": len(branch_counts),
            "avg_branching": np.mean(branch_counts),
            "max_branching": max(branch_counts)
        }

    def _select_candidates(self, u, v, threshold, error=None):
        """
        Select the strongest witness nodes for the (u, v) step.

        Retrieves recorded witnesses for (u, v), grades each by how much
        its contribution exceeds the threshold via Refutes, and returns
        the top 3. Filters out self-loops and, if an error matrix is
        provided, nodes that carry forward known errors.
        """
        polynomial = self._witnesses.get((u, v), {})
        graded = sorted(
            [(y, score) for y, score in polynomial.items() if score > threshold],
            key=lambda x: x[1], reverse=True
        )
        candidates = np.array([y for y, _ in graded], dtype=int)[:3]
        # print(f"  [candidates] ({u},{v}): {len(polynomial)} witnesses, {len(candidates)} above threshold")
        if len(candidates) == 0:
            return candidates
        if error is not None:
            gaps = error[u, :] > threshold
            candidates = candidates[~gaps[candidates]]
            # print(f"  [candidates] ({u},{v}): {len(candidates)} after error filter")
        return candidates[(candidates != v) & (candidates != u)]

    def _recurse(self, E, u, v, candidates, threshold, seen):
        """
        Build proof branches for each candidate witness node.

        Orders candidates by directional coherence — Implies(E[u,y], E[y,v])
        is Top when the path strengthens from u to y to v, suspect otherwise.
        Recursively proves each sub-path and wraps the results in Tree.Pair
        nodes. Returns None if no valid branches are found.
        """
        # Sort by directional coherence: Implies(E[u,y], E[y,v]) is Top when
        # the path strengthens (u→y weaker than y→v), suspect otherwise
        ordered = sorted(
            [y for y in candidates if y not in seen],
            key=lambda y: Implies(E[u, y], E[y, v]),
            reverse=True
        )
        # print(f"  [recurse] ({u}→{v}): {len(ordered)} unseen candidates")
        branches = [
            Tree.Pair(
                self.Prove(E, u, y, threshold, seen | {y}) or Tree.Pair(u, y),
                self.Prove(E, y, v, threshold, seen | {y})
            )
            for y in ordered
        ]
        branches = [b for b in branches if b is not None]
        # print(f"  [recurse] ({u}→{v}): {len(branches)} valid branches")
        return {"node": (u, v), "branches": branches} if branches else None

    def Prove(self, E, u, v, threshold=0.05, seen=None):
        """
        Recursively construct a proof tree for the (u, v) connection.

        Selects witness nodes, recurses into sub-paths, and returns a
        nested Tree structure. Returns a direct Tree.Pair if (u, v) is
        a direct edge with no intermediate witnesses. Returns None if no
        proof can be constructed above the threshold.

        Parameters
        ----------
        E : ndarray
            The (closed) relation matrix.
        u : int
            Source entity index.
        v : int
            Destination entity index.
        threshold : float, optional
            Minimum edge strength to consider. Default is 0.05.
        seen : set, optional
            Entity indices already visited in this branch, to prevent cycles.
        """
        if seen is None:
            seen = set()
            # print(f"[Prove] ({u}→{v}) threshold={threshold:.4f}")
        if u == v:
            return None
        candidates = self._select_candidates(u, v, threshold)
        if len(candidates) == 0:
            if E[u, v] > threshold:
                # print(f"  [Prove] ({u}→{v}): direct edge, score={E[u,v]:.4f}")
                return Tree.Pair(u, v)
            # print(f"  [Prove] ({u}→{v}): no candidates, no direct edge")
            return None
        return self._recurse(E, u, v, candidates, threshold, seen)

    # Self-join on the converged closure — Join(R*, R*).
    # Finds intermediate nodes y such that u can reach y AND y can reach v.
    # λx. λy. λz.  (x z)(z y)
    def Witnesses(self, R, isClosed, temp):
        """
        Populate the witness store by self-joining the closure of R.

        Computes the transitive closure R* if not already closed, then
        runs Join(R*, R*) with witness tracking enabled. Every intermediate
        node y that connects u to v during the join is recorded in
        self._witnesses[(u, v)].

        Caches the result — if called again with the same R, returns
        immediately without recomputing.

        Parameters
        ----------
        R : ndarray
            The relation matrix.
        isClosed : bool
            If True, R is already the transitive closure and Closure is skipped.
        temp : float
            Temperature passed to Closure and Join.
        """
        if self._R_star is not None and self._R_source is R:
            return
        self._R_star = self.Closure(R, temp=temp) if not isClosed else R
        self._R_source = R
        self._clear_witnesses()
        self.tracking = True
        self.Join(self._R_star, self._R_star, temp)
        self.tracking = False

    def Query(self, W, src, dst, names, relation="related to", threshold=0.05, temp=0.05, return_proof=False, isClosed=False):
        """
        Main entry point. Proves and formats a reasoning chain from src to dst.

        Populates the witness store, constructs a proof tree via Prove, and
        formats it into a human-readable explanation via format_proof.

        Parameters
        ----------
        W : ndarray
            The relation matrix.
        src : int
            Source entity index.
        dst : int
            Destination entity index.
        names : list of str
            Entity names indexed by position.
        relation : str, optional
            Relation label used in the formatted output. Default is "related to".
        threshold : float, optional
            Minimum edge strength to include in the proof. Default is 0.05.
        temp : float, optional
            Temperature passed to Witnesses. Default is 0.05.
        return_proof : bool, optional
            If True, returns both the formatted string and the raw proof tree.
        isClosed : bool, optional
            If True, W is already the transitive closure. Default is False.

        Returns
        -------
        str or dict
            The formatted proof. If return_proof is True, returns a tuple of
            (formatted, raw_proof).
        """
        print(f"[Query] {names[src]} → {names[dst]}  relation={relation!r}  threshold={threshold}  temp={temp}")
        self.Witnesses(W, isClosed, temp=temp)
        proof = self.Prove(self._R_star, src, dst, threshold=threshold)
        print(f"[Query] proof {'found' if proof else 'not found'}")
        formatted = format_proof(proof, self._R_star, names, relation)
        if return_proof:
            return formatted, proof
        return formatted
