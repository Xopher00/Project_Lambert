"""
Attention layer. Implements single-head and multi-head retrieval over a
concept embedding, using fixpoint iteration to converge on a stable set
of matching entities.

Each Attention head holds one embedding matrix and iterates a query vector
to fixpoint using Attend and Residuate — the same pattern as Closure, but
applied to a query rather than a relation. The converged state is a
concept-space vector whose projection back onto the embedding gives entity
scores.

MultiHeadAttention runs one Attention head per embedding, combines their
scores via a belief-propagation-like outer fixpoint, and returns the union
of matching entities across all heads.
"""

import numpy as np
from functools import reduce
from core.algebra import *
from lattice.embed import Embed
from core.fixpoint import FixpointIterator
from concurrent.futures import ThreadPoolExecutor

class Attention(Embed):
    """
    Single-head retrieval over a concept embedding.

    Holds one embedding matrix and a FixpointIterator that iterates a query
    vector to convergence. Each step calls Attend to score the query against
    stored patterns, normalises via SoftMax, then applies a Residuate
    correction to ensure the output stays within what the embedding can
    justify — the same constraint used in Closure.

    Parameters
    ----------
    emb : ndarray, shape (n, k)
        The embedding matrix. Rows are entities, columns are concept dimensions.
    temp : float, optional
        Initial temperature for the fixpoint iterator. Default is 1.0.
    eps : float, optional
        Convergence threshold. Default is 1e-3.
    max_iters : int, optional
        Maximum iterations before stopping. Default is 100.
    """

    def __init__(self, emb, temp=1.0, eps=1e-3, max_iters=100, operator=None):
        super().__init__()
        self.emb = emb
        self.Correct = self.coder.op("support propagate")
        """
        Project the converged query state back to entity scores.

        Computes Join(state, emb.T) to measure how well the current fixpoint
        state matches each entity in the embedding. The result is the head's
        confidence in each entity as a retrieval candidate.

        Returns
        -------
        ndarray, shape (n,)
            A score for each entity. Higher values indicate stronger match.
        """
        self.Scores = self.coder.op("realize")
        self.fp  = FixpointIterator(
            f         = self._step,
            state0    = emb[0].copy(),
            temp      = temp,
            eps       = eps,
            max_iters = max_iters,
        )

    def _step(self, q, temp):
        """
        One iteration of the attention fixpoint.

        Applies Attend to score the query, normalises the result with SoftMax,
        then applies a Residuate correction to constrain the output to what the
        embedding can support — preventing the query from drifting toward
        patterns the embedding cannot justify.

        This mirrors the correction step in Closure: raw output is clipped down
        to the greatest value consistent with the stored relation.

        # SoftMax is used here rather than Softplus — Softplus caused instability
        # during experimentation.

        Parameters
        ----------
        q : ndarray, shape (k,)
            Current query vector in concept space.
        temp : float
            Current temperature from the fixpoint iterator.

        Returns
        -------
        corrected : ndarray, shape (k,)
            Updated query vector after one step.
        raw : ndarray, shape (k,)
            The uncorrected SoftMax output, returned as aux for energy computation.
        """
        raw = self.SoftMax(self.Attend(q, self.emb, temp), temp, axis=0)
        corrected = self.SmoothMin((raw, self.Correct(raw, self.emb, temp)), temp, axis=0)
        return corrected, raw

    def _query(self, idx):
        """
        Build a query vector from one or more entity indices.

        For a single index, returns that entity's embedding row directly.
        For a list or array of indices, takes the elementwise minimum across
        the selected rows — the greatest lower bound in the concept lattice.
        This forms a conjunctive query: the result represents what all
        specified entities have in common, and retrieval will find entities
        that satisfy all of them simultaneously.

        Parameters
        ----------
        idx : int, list of int, or ndarray of int
            Index or indices of the query entities.

        Returns
        -------
        ndarray, shape (k,)
            The query vector in concept space.
        """
        if isinstance(idx, (list, np.ndarray)):
            q = self.emb[idx].min(axis=0)
        else:
            q = self.emb[idx].copy()
        return q

    def retrieve(self, idx, temp=None):
        """
        Run the attention fixpoint from a query and return matching entities.

        Builds a query vector from idx, perturbs the fixpoint iterator to that
        starting state, runs to convergence, then returns all entities whose
        score falls within eps of the maximum score.

        Returns empty results if the query vector is all zeros (unknown entity).

        Parameters
        ----------
        idx : int, list of int, or ndarray of int
            Index or indices of the query entities. Passed to _query.
        temp : float, optional
            Not currently used. Reserved for future use.

        Returns
        -------
        hits : ndarray of int
            Indices of entities whose score is within eps of the maximum.
        weights : ndarray, shape (k,)
            The converged fixpoint state, representing the head's belief
            distribution over concept dimensions.
        """
        if isinstance(idx, np.ndarray) and idx.dtype.kind == 'f':
            q = idx
        else:
            q = self._query(idx)
        if np.all(q == 0):
            return np.array([]), None
        self.fp.perturb(q)
        scores = self.Scores()
        mask   = scores >= scores.max() - self.fp.eps
        weights = self.fp.state
        return np.where(mask)[0], weights


class MultiHeadAttention(Embed):
    """
    Multi-head retrieval over a set of embedding matrices.

    Holds one Attention head per embedding. Each head may correspond to a
    relation, a set of attributes, or any other partition of the concept
    space — depending on how the data is structured when passed to the model.

    Given a query, runs all heads in parallel, collects their entity scores,
    and combines them via an outer fixpoint that propagates beliefs across
    heads using Residuate and Join.

    The outer fixpoint iterates until the combined score vector converges.
    At each step, the current combined scores are used to select active
    entities, each head retrieves against that active set, and the results
    are merged back into the combined scores.

    Parameters
    ----------
    heads : list of Attention
        One head per embedding.
    names : list of str
        Names corresponding to each head, used as keys in the intents dict.
    eps : float, optional
        Convergence threshold for both inner and outer fixpoints. Default is 1e-3.
    max_iters : int, optional
        Maximum outer fixpoint iterations. Default is 20.
    """

    def __init__(self, heads, names, eps=1e-3, max_iters=20):
        super().__init__()
        self.heads = heads
        self.names = names
        self.eps = eps
        self.intents = {}

        state0 = np.zeros(heads[0].emb.shape[0])
        self.fp = FixpointIterator(
            f         = self._outer_step,
            state0    = state0,
            eps       = eps,
            max_iters = max_iters,
        )

    def _run_head(self, head, name, idx):
        """
        Run a single head and store its result in the intents dict.

        Calls head.retrieve(idx) and, if any entities were found, stores the
        converged state and entity scores under the head's name. Called
        concurrently for all heads inside _outer_step.

        Writing to self.intents is safe across threads because each head writes
        to a distinct key.

        Parameters
        ----------
        head : Attention
            The head to run.
        name : str
            Key under which to store the result in self.intents.
        idx : ndarray of int
            Active entity indices passed as the query to head.retrieve.
        """
        hits, state = head.retrieve(idx)
        if len(hits) > 0:
            self.intents[name] = (np.clip(state, 0, None), head.scores())

    def _outer_step(self, combined_scores, temp):
        """
        One iteration of the outer fixpoint across all heads.

        Selects the currently active entities (those within eps of the maximum
        combined score), runs all heads in parallel against that active set,
        then merges the resulting scores into a new combined score vector.

        The merge uses elementwise minimum (hard intersection) across all head
        scores — the lattice infimum, keeping only entities satisfying all
        relations simultaneously.

        # A belief-propagation-like soft merge was tried previously — see
        # commented-out code below. It resolved instability on complex datasets
        # but caused category collapse.

        Parameters
        ----------
        combined_scores : ndarray, shape (n,)
            Current combined entity scores from the previous iteration.
        temp : float
            Current temperature from the outer fixpoint iterator.

        Returns
        -------
        raw : ndarray, shape (n,)
            Updated combined scores.
        combined_scores : ndarray, shape (n,)
            The previous combined scores, returned as aux for energy computation.
        """
        idx = np.where(combined_scores >= combined_scores.max() - self.eps)[0]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._run_head, head, name, idx)
                    for head, name in zip(self.heads, self.names)]
            for future in futures:
                future.result()
        if not self.intents:
            return combined_scores, None

        # Hard intersection across heads — elementwise minimum (lattice infimum).
        # Theoretically correct for multi-relational concept intersection (Brito et al., Theorem 8).
        new_combined = reduce(np.minimum, [scores for _, (_, scores) in self.intents.items()])
        return new_combined, combined_scores

        # Belief-propagation-like soft merge — under review.
        # Avoids instability of hard intersection on complex datasets but causes category collapse.
        # all_scores = np.stack([scores for _, (_, scores) in self.intents.items()])
        # head_weights = self.Residuate(combined_scores[:, None], all_scores.T, temp)
        # raw = self.Join(head_weights, all_scores, temp).squeeze()
        # raw = self.SmoothMax((raw, combined_scores), temp, axis=0)
        # return raw, combined_scores

    def retrieve(self, idx):
        """
        Run the outer fixpoint from a query and return matching entities.

        Initialises the combined score vector with 1.0 at the query indices,
        then runs the outer fixpoint to convergence. Returns all entities whose
        final score falls within eps of the maximum, along with the full intents
        dict collected across all heads.

        Parameters
        ----------
        idx : int or array-like of int
            Index or indices of the query entities.

        Returns
        -------
        final_idx : ndarray of int
            Indices of entities whose score is within eps of the maximum.
        intents : dict
            Mapping from head name to (state, scores) tuple for each head
            that returned results. State is the converged concept-space vector;
            scores are the entity-level confidence values.
        """
        scores0 = np.zeros(self.heads[0].emb.shape[0])
        if isinstance(idx, np.ndarray) and idx.dtype.kind == 'f':
            scores0 = idx                    # backward: pre-built scores vector
        else:
            scores0[idx] = 1.0               # forward: seed by entity indices
        self.fp.perturb(scores0)
        final_idx = np.where(self.fp.state >= self.fp.state.max() - self.eps)[0]
        return final_idx, self.intents
