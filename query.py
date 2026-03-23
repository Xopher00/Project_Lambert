from dataclasses import dataclass
from functools import reduce
import numpy as np

@dataclass
class QueryResult:
    """
    The result of a concept retrieval query.

    Attributes
    ----------
    entities : list of str
        Names of the retrieved entities, ranked by score.
    scores : ndarray, shape (k,)
        Concept membership scores for each retrieved entity.
    intents : dict
        Raw intents dict from MultiHeadAttention — maps head name to
        (state, scores) tuple for each head that returned results.
    provenance : list of (str, str, float)
        Active features explaining the result, as (head_name, label, value)
        triples sorted by value descending.
    mode : str
        Query mode: 'forward' (single entity), 'intersection' (multiple
        entities), or 'backward' (feature-driven).
    """
    entities:   list
    scores:     np.ndarray
    intents:    dict
    provenance: list
    mode:       str

class Query:
    """
    High-level query interface over a trained Lambert model.

    Supports three query modes:

    - Forward (single entity): retrieve entities sharing the same concept.
    - Intersection (multiple entities): retrieve entities in the concept
      that is the greatest lower bound of all specified entities.
    - Backward (feature-driven): build a query vector from matching
      embedding columns and retrieve entities that satisfy those features.

    All modes run through MultiHeadAttention.retrieve and return a
    QueryResult with ranked entities and provenance.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : Lambert model
            Must expose entity_labels, heads, explorer.mha, and eps.
        """
        self.model = model
        self.n             = model.explorer.mha.heads[0].emb.shape[0]
        self._col_index = {
            f'[{hn}] {h["feature_labels"][h["rep_cols"][j]]}': (hn, j)
            for hn, h in model.heads.items()
            for j in range(len(h['rep_cols']))
        }

    def _resolve(self, entities):
        """Convert entity name(s) to integer indices."""
        if isinstance(entities, str):
            entities = [entities]
        return [self.model.entity_labels.index(e) for e in entities]

    def _rank(self, scores, top_k):
        """
        Return indices of top-scoring entities.

        Filters out zero and sentinel scores, then keeps all entities
        within eps of the maximum score, up to top_k.
        """
        ranked = np.argsort(scores)[::-1]
        ranked = ranked[(scores[ranked] > self.model.eps) & (scores[ranked] > -1e8)]
        if len(ranked) == 0:
            return ranked
        ranked = ranked[scores[ranked] >= scores[ranked[0]] - self.model.eps][:top_k]
        return ranked

    def _provenance(self, intents):
        """
        Extract active features from the intents dict.

        For each head that returned results, reads the converged intent
        vector and collects all concept dimensions above eps, labelled
        by their feature name.

        Returns
        -------
        list of (head_name, label, value)
            Sorted by value descending.
        """
        return sorted([
            (hn, f'[{hn}] {self.model.heads[hn]["feature_labels"][self.model.heads[hn]["rep_cols"][j]]}', float(v))
            for hn, (intent_vec, _) in intents.items()
            for j, v in enumerate(intent_vec) if v > self.model.eps
        ], key=lambda x: -x[2])

    def _build_seeds(self, entities=None, features=None):
        """
        Build the query seed and determine the query mode.

        For entity queries, resolves names to indices and sets mode to
        'forward' or 'intersection'. For feature queries, constructs a
        dense score vector by taking the elementwise maximum of all
        embedding columns whose label matches any of the requested
        features, and sets mode to 'backward'.

        Returns
        -------
        seeds : list of int or ndarray, shape (n,)
        mode : str
        """
        if entities is not None:
            idx  = self._resolve(entities)
            mode = 'intersection' if len(idx) > 1 else 'forward'
            return idx, mode
        if isinstance(features, str):
            features = [features]
        n = self.model.explorer.mha.heads[0].emb.shape[0]
        q = np.zeros(n)
        for feat in features:
            for label, (hn, j) in self._col_index.items():
                if feat.lower() in label.lower():
                    head_idx = self.model.explorer.mha.names.index(hn)
                    q = np.maximum(q, self.model.explorer.mha.heads[head_idx].emb[:, j])
        return q, 'backward'

    def __call__(self, entities=None, features=None, top_k=10):
        """
        Run a concept retrieval query and print the results.

        Parameters
        ----------
        entities : str or list of str, optional
            Entity name(s) to query from. Mutually exclusive with features.
        features : str or list of str, optional
            Feature name(s) to query from. Mutually exclusive with entities.
        top_k : int, optional
            Maximum number of results to return. Default is 10.

        Returns
        -------
        QueryResult
        """
        if entities is None and features is None:
            raise ValueError("Provide either entities or features.")
        seeds, mode = self._build_seeds(entities=entities, features=features)
        self.model.explorer.mha.intents = {}
        hits, intents = self.model.explorer.mha.retrieve(seeds)
        state  = self.model.explorer.mha.fp.state
        ranked = self._rank(state, top_k)
        prov   = self._provenance(intents)
        active = set(round(float(state[i]), 3) for i in ranked)
        prov   = [(hn, l, v) for hn, l, v in prov if round(v, 3) in active]
        return show(QueryResult(
            [self.model.entity_labels[i] for i in ranked],
            state[ranked], intents, prov, mode
        ))


def show(r, show_provenance=True):
    """
    Print a QueryResult and return it.

    Parameters
    ----------
    r : QueryResult
    show_provenance : bool, optional
        If True, prints the top 8 provenance features. Default is True.

    Returns
    -------
    QueryResult
    """
    print(f'\n[{r.mode}]  {len(r.entities)} results')
    if not r.entities:
        print('  (no shared concept)')
        return r
    for name, score in zip(r.entities, r.scores):
        print(f'  {name:<30s}  {score:.3f}')
    if show_provenance and r.provenance:
        print(f'\n  provenance ({len(r.provenance)} features):')
        for _, label, v in r.provenance[:8]:
            print(f'    {label:<40s}  {v:.3f}')
    return r
