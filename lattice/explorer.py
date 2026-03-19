"""
Category exploration layer. Sits above the attention layer and discovers
the full structure of the concept lattice implied by a set of relations.

Exploration proceeds in two phases. The first phase (explore) seeds the
MultiHeadAttention with each entity in turn, collects the converged extent
vector as a category, and deduplicates by extent key. Entities already
assigned to a category are skipped — querying them again would be redundant.

The second phase (explore_lattice) builds a new embedding matrix from the
discovered category extents and runs ConceptEmbed on it, closing the lattice
under transitive composition. The result is an embedding that captures all
relational connections implied by the data, including those not explicitly
present — an advanced form of transitive closure over the concept lattice.

Inspired by category theory: a category is defined by its relations, and
new categories can be defined by composing existing ones.
"""

from core.fixpoint import FixpointIterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from core.algebra import *
from lattice.embed import Embed


class CategoryExplorer(Embed):
    """
    Explores and closes the concept lattice of a MultiHeadAttention instance.

    Takes a trained MultiHeadAttention and systematically queries it to
    discover all categories implied by the data. Each category is a
    (key, extent) pair: the key identifies the category by its quantised
    extent vector, and the extent records the strongest defining features
    that caused each entity to be placed into that category.

    Parameters
    ----------
    mha : MultiHeadAttention
        The trained attention instance to explore.
    eps : float, optional
        Threshold for extent membership and category key quantisation.
        Default is 1e-3.
    """

    def __init__(self, mha, model, eps=1e-3):
        super().__init__()
        self.__dict__.update(model.__dict__)
        self.mha = mha
        self.eps = eps
        self.categories = {}
        self.seen    = {}
        self.covered = set()
        self.rep_cols = []

    # An earlier version of the intent key method using intents instead of extents
    # keep this version in mind as we research the nature of extents and intents in formal concept analysis
    # def _intent_key(self):
    #    parts = []
    #    for name, (intent, _) in self.mha.intents.items():
    #        parts.append(tuple((intent / self.eps).astype(int)))
    #    return tuple(parts)

    def _intent_key(self):
        """
        Compute a hashable key identifying the current category by its extent.

        Quantises the outer fixpoint's converged state vector by dividing by
        eps and casting to int, producing a tuple that can be used as a dict key.

        Extent-based keying is used deliberately: extent vectors are more stable
        identifiers than intent vectors, which can vary across heads even for
        structurally identical categories.

        # An earlier version keyed by per-head intent vectors instead:
        #   parts = []
        #   for name, (intent, _) in self.mha.intents.items():
        #       parts.append(tuple((intent / self.eps).astype(int)))
        #   return tuple(parts)
        # Retained here as a reference while the relationship between extents
        # and intents in this context is still under research.

        Returns
        -------
        tuple of int
            A hashable key derived from the quantised extent vector.
        """
        extent = self.mha.fp.state
        return tuple((extent / self.eps).astype(int))
    
    def explore(self, n_entities=None, seeds=None):
        """
        Perform the initial category discovery phase.

        Iterates over each entity, queries the MultiHeadAttention, and stores
        the converged extent vector as a new category if its key has not been
        seen before. Entities already covered by a discovered category are
        skipped — once an entity has been assigned to a category, querying it
        again is redundant.

        Progress is printed every 50 entities.

        Parameters
        ----------
        n_entities : int
            Number of entities to iterate over.

        Returns
        -------
        dict
            The categories discovered so far. Each entry maps a quantised
            extent key to a dict with keys:

            - ``'intents'``: the per-head intent dict from MultiHeadAttention
            - ``'extent'``: the converged outer fixpoint state vector
        """
        candidates = [seeds] if seeds is not None else [[i] for i in range(n_entities)]
        covered = set()
        for query in candidates:
            if all(i in covered for i in query):
                continue
            if n_entities and len(covered) % 50 == 0:
                print(f"  covered={len(covered)}  categories={len(self.categories)}")
            self.mha.intents = {}
            hits, _ = self.mha.retrieve(query)
            if not len(hits): continue
            key = self._intent_key()
            extent = self.mha.fp.state.copy()
            if key not in self.categories:
                self.categories[key] = {'intents': dict(self.mha.intents), 'extent': extent}
                covered.update(np.flatnonzero(extent > self.eps).tolist())
        return self.categories

    def _concept_fixpoint(self, R, seed, temp, max_iters=20, eps=1e-3):
        """
        Override of Embed._concept_fixpoint using MHA retrieval as the fixpoint step.

        Instead of alternating Residuate calls, each iteration queries the
        MultiHeadAttention with the currently active entities and takes the
        converged outer fixpoint state as the new state. This finds the stable
        set of relational connections — how entities are defined by combinations
        of features across all heads — rather than a single concept vector.

        Like Closure, this is a form of transitive closure: the fixpoint is
        reached when the set of entities and their relational structure stops
        changing.

        Returns the seed unchanged if no entities are active above eps.

        Parameters
        ----------
        R : ndarray
            The relation matrix. Passed through to satisfy the Embed interface
            but not used directly — retrieval is handled by the MHA.
        seed : ndarray, shape (n,)
            Starting state vector. Active entities are those with value > eps.
        temp : float
            Temperature passed to the inner FixpointIterator.
        max_iters : int, optional
            Maximum iterations before stopping. Default is 20.
        eps : float, optional
            Activity threshold and convergence threshold. Default is 1e-3.

        Returns
        -------
        ndarray, shape (n,)
            The converged state vector.
        """
        active = np.flatnonzero(seed > eps)
        if len(active) == 0:
            return seed
        def _f(state, temp):
            hits, _ = self.mha.retrieve(np.flatnonzero(state > eps).tolist())
            return self.mha.fp.state.copy(), None
        fp = FixpointIterator(
            f         = _f,
            state0    = seed.copy(),
            eps       = eps,
            max_iters = max_iters,
        )
        return fp.run()

    def explore_lattice(self, n_entities=None, seeds=None, verbose=False):
        """
        Perform full lattice closure over the discovered categories.

        Runs in two phases:

        1. Calls explore to discover all first-order categories implied by the
           data, one entity at a time.

        2. Builds a new embedding matrix where each column is one category's
           extent vector, then runs ConceptEmbed on it. This closes the lattice
           under transitive composition — finding all higher-order categories
           implied by combinations of the first-order ones.

        The result is an embedding that captures all relational connections in
        the data, including those not explicitly present. Each column of the
        returned embedding corresponds to a concept in the fully closed lattice.

        # The theoretical status of the second-phase embedding is still under
        # investigation — the structure of higher-order concept compositions
        # at this level of abstraction is not yet fully worked out.

        Parameters
        ----------
        n_entities : int
            Number of entities to explore in the first phase.

        Returns
        -------
        emb : ndarray, shape (n_entities, k)
            The final embedding matrix over the closed concept lattice.
        EmbR : ndarray, shape (k, k)
            Tucker projection of the category matrix. Currently under review —
            callers typically discard this value.
        rep_cols : list of int
            Column indices of the category matrix selected as representative
            concepts.
        """
        if verbose is True: print('Performing initial concept exploration . . .')
        self.explore(n_entities=n_entities, seeds=seeds)
        if verbose is True: print('Initial exploration phase complete.')
        n = n_entities or self.mha.heads[0].emb.shape[0]
        emb_new = np.zeros((n, len(self.categories)))
        if verbose is True: print('Beginning full concept exploration . . .')
        for col, (key, cat) in enumerate(self.categories.items()):
            emb_new[:, col] = np.where(cat['extent'] > self.eps, cat['extent'], 0)
        emb, EmbR, self.rep_cols = self.ConceptEmbed(emb_new, temp=self.mha.heads[0].fp.temp, 
                                                     seen=self.seen, covered=self.covered, rep_cols=self.rep_cols)
        unique, inverse = np.unique(emb, axis=0, return_inverse=True)
        self.concept_space.update({
            'emb':        emb,
            'EmbR':       EmbR,
            'rep_cols':   self.rep_cols,
            'unique':     unique,
            'inverse':    inverse,
            'categories': self.categories
        })
        if verbose is True: print('Transitive closure reached.')
