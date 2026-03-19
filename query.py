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

    def __init__(self, model):
        self.__dict__ = model.__dict__
        self.n             = self.explorer.mha.heads[0].emb.shape[0]
        self._col_index = {
            f'[{hn}] {h["feature_labels"][h["rep_cols"][j]]}': (hn, j)
            for hn, h in self.heads.items()
            for j in range(len(h['rep_cols']))
        }

    def _resolve(self, entities):
        if isinstance(entities, str):
            entities = [entities]
        return [self.entity_labels.index(e) for e in entities]

    def _rank(self, scores, top_k):
        ranked = np.argsort(scores)[::-1]
        ranked = ranked[(scores[ranked] > self.eps) & (scores[ranked] > -1e8)]
        if len(ranked) == 0:
            return ranked
        ranked = ranked[scores[ranked] >= scores[ranked[0]] - self.eps][:top_k]
        return ranked

    def _provenance(self, intents):
        return sorted([
            (hn, f'[{hn}] {self.heads[hn]["feature_labels"][self.heads[hn]["rep_cols"][j]]}', float(v))
            for hn, (intent_vec, _) in intents.items()
            for j, v in enumerate(intent_vec) if v > self.eps
        ], key=lambda x: -x[2])

    def _feature_scores(self, features):
        matches = [(hn, j, k) for feat in features
                for k, (hn, j) in self._col_index.items()
                if feat.lower() in k.lower()]
        if not matches:
            return None, []
        head_scores, prov = {}, []
        for hn, j, k in matches:
            col = self.explorer.mha.heads[self.explorer.mha.names.index(hn)].emb[:, j]
            head_scores[hn] = np.maximum(head_scores.get(hn, np.zeros(self.n)), col)
            prov.append((hn, k, float(col.max())))
        return reduce(np.maximum, head_scores.values()), prov

    def __call__(self, entities=None, features=None, top_k=10):
        if features is not None:
            scores0, prov = self._feature_scores(features)
            if scores0 is None:
                return show(QueryResult([], np.array([]), {}, [], 'backward'))
            ranked = self._rank(scores0, top_k)
            return show(QueryResult([self.entity_labels[i] for i in ranked],
                                    scores0[ranked], {}, prov, 'backward'))

        idx     = self._resolve(entities) if entities is not None else []
        self.explorer.explore(seeds=idx)
        state   = self.explorer.mha.fp.state
        intents = dict(self.explorer.mha.intents)
        ranked  = self._rank(state, top_k)
        prov    = self._provenance(intents)
        active  = set(round(float(state[i]), 3) for i in ranked)
        prov    = [(hn, l, v) for hn, l, v in prov if round(v, 3) in active]
        mode    = 'intersection' if len(idx) > 1 else 'forward'
        return show(QueryResult([self.entity_labels[i] for i in ranked],
                                state[ranked], intents, prov, mode))


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