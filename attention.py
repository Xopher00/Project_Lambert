from embed import Embed
import numpy as np
from algebra import *
from fixpoint import FixpointIterator

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
    
    def scores(self, state):
        return self.Join(state[None,:], self.emb.T, self.fp.temp).squeeze()

    def retrieve(self, q):
        state = self.fp.perturb(q)
        scores = self.scores(state)
        mask = scores >= scores.max() - self.fp.eps
        return np.where(mask)[0]
    

class MultiHead(Attention):
    def __init__(self, emb, head_cols, temp=1.0, eps=1e-3, max_iters=100):
        self.head_cols = head_cols
        self.heads = [Attention(emb[:, cols], temp=temp, eps=eps, max_iters=max_iters)
                      for cols in head_cols]
        self.active_heads = None  # in __init__
        super().__init__(emb, temp=temp, eps=eps, max_iters=max_iters)

    def _step(self, q, temp):
        corrected = np.zeros_like(q)
        raw = np.zeros_like(q)
        for head, cols in zip(self.heads, self.head_cols):
            if not np.any(q[cols]):
                continue
            c, r = head._step(q[cols], temp)
            corrected[cols] = c
            raw[cols] = r
        return corrected, raw
    
    def scores(self, state):
        active = self.active_heads
        head_scores = [self.Join(state[None, cols], head.emb.T, self.fp.temp).squeeze()
                    for i, (head, cols) in enumerate(zip(self.heads, self.head_cols))
                    if (active is None or i in active) and len(cols) > 0]
        return self.SmoothMin(tuple(head_scores), self.fp.temp, axis=0)
