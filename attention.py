from embed import Embed
import numpy as np
from functools import reduce
from algebra import *
from fixpoint import FixpointIterator
from concurrent.futures import ThreadPoolExecutor

class Attention(Embed):
    def __init__(self, emb, temp=1.0, eps=1e-3, max_iters=100):
        super().__init__()
        self.emb = emb
        self.fp  = FixpointIterator(
            f         = self._step,
            energy_fn = self._energy,
            state0    = emb[0].copy(),
            temp      = temp,
            eps       = eps,
            max_iters = max_iters,
        )

    def _step(self, q, temp):
        J   = self.Attend(q, self.emb, temp)
        raw = self.SoftMax(J, temp, axis=0)
        allowed   = self.Residuate(raw[:, None], self.emb.T, temp).squeeze()
        corrected = self.SmoothMin((raw, self.Join(allowed[None,:], self.emb, temp).squeeze()), temp, axis=0)
        return corrected, raw

    def _energy(self, new, old, aux):
        raw           = aux
        dynamic_error = float(Sum(Abs(new - old) ** 2))
        sensory_error = float(Sum(Abs(raw - new) ** 2))
        return dynamic_error + sensory_error
    
    def scores(self):
        return self.Join(self.fp.state[None,:], self.emb.T, self.fp.temp).squeeze()
    
    def _query(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            q = self.emb[idx].min(axis=0)
        else:
            q = self.emb[idx].copy()
        return q

    def retrieve(self, idx, temp=None):
        q = self._query(idx)
        if np.all(q == 0):
            return np.array([]), None
        self.fp.perturb(q)
        scores = self.scores()
        mask   = scores >= scores.max() - self.fp.eps
        weights = self.fp.state
        return np.where(mask)[0], weights
    

class MultiHeadAttention(Embed):
    def __init__(self, heads, names, eps=1e-3, max_iters=20):
        super().__init__()
        self.heads = heads
        self.names = names
        self.eps = eps
        self.intents = {}

        state0 = np.zeros(heads[0].emb.shape[0])
        self.fp = FixpointIterator(
            f         = self._outer_step,
            energy_fn = self._outer_energy,
            state0    = state0,
            eps       = eps,
            max_iters = max_iters,
        )
    
    def _run_head(self, head, name, idx):
        hits, state = head.retrieve(idx)
        if len(hits) > 0:
            self.intents[name] = (np.clip(state, 0, None), head.scores())

    def _outer_step(self, combined_scores, temp):
        idx = np.where(combined_scores >= combined_scores.max() - self.eps)[0]
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._run_head, head, name, idx)
                    for head, name in zip(self.heads, self.names)]
            for future in futures:
                future.result()

        new_combined = reduce(np.minimum, [scores for _, (_, scores) in self.intents.items()])

        return new_combined

    def _outer_energy(self, new, old, aux):
        return float(Sum(Abs(new - old) ** 2))

    def retrieve(self, idx):
        scores0 = np.zeros(self.heads[0].emb.shape[0])
        scores0[idx] = 1.0
        self.fp.perturb(scores0)
        final_idx = np.where(self.fp.state >= self.fp.state.max() - self.eps)[0]
        return final_idx, self.intents