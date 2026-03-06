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
from tree import Tree
import networkx as nx
from tensor import Tensor as t
from audit import format_proof
from algebra import Implies, Refutes
from collections import deque

class Provenance(t):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._R_star = None
        self._R_source = None

    def witness_capacity(self, threshold=0.05):
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
        polynomial = self._witnesses.get((u, v), {})
        # Grade each witness by how much it exceeds the threshold
        graded = sorted(
            [(y, Refutes(threshold, score)) for y, score in polynomial.items()],
            key=lambda x: x[1], reverse=True
        )
        candidates = np.array([y for y, g in graded if g > 0], dtype=int)[:3]
        # print(f"  [candidates] ({u},{v}): {len(polynomial)} witnesses, {len(candidates)} above threshold")
        if len(candidates) == 0:
            return candidates
        if error is not None:
            gaps = error[u, :] > threshold
            candidates = candidates[~gaps[candidates]]
            # print(f"  [candidates] ({u},{v}): {len(candidates)} after error filter")
        return candidates[(candidates != v) & (candidates != u)]

    def _recurse(self, E, u, v, candidates, threshold, seen):
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
    def Witnesses(self, R, temp):
        if self._R_star is not None and self._R_source is R:
            return
        self._R_star = self.Closure(R, temp=temp)
        self._R_source = R
        self._clear_witnesses()
        self.tracking = True
        self.Join(self._R_star, self._R_star, temp)
        self.tracking = False

    def Query(self, W, src, dst, names, relation="related to", threshold=0.05, temp=0.05, return_proof=False):
        print(f"[Query] {names[src]} → {names[dst]}  relation={relation!r}  threshold={threshold}  temp={temp}")
        self.Witnesses(W, temp=temp)
        proof = self.Prove(self._R_star, src, dst, threshold=threshold)
        print(f"[Query] proof {'found' if proof else 'not found'}")
        formatted = format_proof(proof, self._R_star, names, relation)
        if return_proof:
            return formatted, proof
        return formatted
