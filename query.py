from dataclasses import dataclass
from functools import reduce
import numpy as np


@dataclass
class QueryResult:
    entities:   list
    scores:     np.ndarray
    intents:    dict
    provenance: list
    mode:       str


class Query:
    """
    Query interface for a trained Lambert pipeline.

    Supports forward retrieval, backward feature lookup, intersection,
    multi-hop composition, and relation queries between entity pairs.

    Parameters
    ----------
    pipe : Lambert
        A fully run() Lambert instance.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.mha  = pipe.explorer.mha
        self.eps  = pipe.eps
        self.n    = pipe.concept_space['emb'].shape[0]
        self._col_index = {
            f'[{hn}] {h["feature_labels"][h["rep_cols"][j]]}': (hn, j)
            for hn, h in pipe.heads.items()
            for j in range(len(h['rep_cols']))
        }

    def _resolve(self, entities):
        if isinstance(entities, str):
            entities = [entities]
        return [self.pipe.entity_labels.index(e) for e in entities]

    def _provenance(self, intents):
        p = self.pipe.heads
        return sorted([
            (hn, f'[{hn}] {p[hn]["feature_labels"][p[hn]["rep_cols"][j]]}', float(v))
            for hn, (intent_vec, _) in intents.items()
            for j, v in enumerate(intent_vec) if v > self.eps
        ], key=lambda x: -x[2])

    def _hop(self, seed):
        self.mha.intents = {}
        self.mha.fp.perturb(seed)
        state = self.mha.fp.state.copy()
        idx   = np.where(state >= state.max() - self.eps)[0]
        return idx, dict(self.mha.intents), state

    def _rank(self, scores, top_k):
        ranked = np.argsort(scores)[::-1]
        ranked = ranked[(scores[ranked] > self.eps) & (scores[ranked] > -1e8)]
        ranked = ranked[scores[ranked] >= scores[ranked[0]] - self.eps][:top_k]
        return ranked

    def _feature_scores(self, features):
        head_scores = {}
        for feat, strength in features.items():
            matches = [(hn, j) for k, (hn, j) in self._col_index.items()
                       if feat.lower() in k.lower()]
            if not matches:
                print(f'  [warn] feature not found: {feat}')
                continue
            for hn, j in matches:
                head   = self.mha.heads[self.mha.names.index(hn)]
                active = np.where(head.emb[:, j] > self.eps)[0]
                if not len(active):
                    continue
                head.retrieve(active)
                s = np.clip(head.scores(), 0, None)
                head_scores[hn] = np.maximum(head_scores.get(hn, np.zeros(self.n)), s)
        return reduce(np.minimum, head_scores.values()) if head_scores else None

    def search(self, entities=None, features=None, hops=1, top_k=10):
        """
        Retrieve entities matching a query.

        Parameters
        ----------
        entities : str or list of str, optional
        features : dict, optional
            Maps feature substring to strength in [0,1].
        hops : int
            Forward composition steps.
        top_k : int
        """
        idx = self._resolve(entities) if entities is not None else []

        if features is not None:
            scores0 = self._feature_scores(features)
            if scores0 is None:
                return QueryResult([], np.array([]), {}, [], 'backward')
            if idx:
                anchor = np.zeros(self.n); anchor[idx] = 1.0
                scores0 = np.minimum(scores0, anchor)
            ranked = self._rank(scores0, top_k)
            mode   = 'novel' if idx else 'backward'
            return QueryResult([self.pipe.entity_labels[i] for i in ranked],
                               scores0[ranked], {}, [], mode)

        scores0 = np.zeros(self.n); scores0[idx] = 1.0
        final_idx, intents, state = self._hop(scores0)
        for _ in range(hops - 1):
            if not len(final_idx): break
            seed = np.zeros(self.n); seed[final_idx] = 1.0
            final_idx, intents, state = self._hop(seed)

        ranked = self._rank(state, top_k)
        prov   = self._provenance(intents)
        active = set(round(float(state[i]), 3) for i in ranked)
        prov   = [(hn, l, v) for hn, l, v in prov if round(v, 3) in active]
        mode   = 'multihop' if hops > 1 else 'intersection' if len(idx) > 1 else 'forward'
        return QueryResult([self.pipe.entity_labels[i] for i in ranked],
                           state[ranked], intents, prov, mode)

    def relate(self, entities):
        """
        Describe what a set of entities have in common.
        Returns provenance only — no entity list.
        """
        r = self.search(entities)
        if not r.provenance:
            print('  (no shared concept)')
            return
        for _, label, v in r.provenance:
            print(f'  {label:<40s}  {v:.3f}')


def show(r, show_provenance=True):
    print(f'\n[{r.mode}]  {len(r.entities)} results')
    if not r.entities:
        print('  (no shared concept)')
        return
    for name, score in zip(r.entities, r.scores):
        print(f'  {name:<30s}  {score:.3f}')
    if show_provenance and r.provenance:
        print(f'\n  provenance ({len(r.provenance)} features):')
        for _, label, v in r.provenance[:8]:
            print(f'    {label:<40s}  {v:.3f}')