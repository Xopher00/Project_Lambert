from embed import Embed
import numpy as np
from algebra import *
from fixpoint import FixpointIterator

class Memory(Embed):
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
        raw       = self.Attend(q, self.emb, temp)
        raw = self.Softplus(raw, temp)
        allowed   = self.Residuate(raw[:, None], self.emb.T, temp).squeeze()  # (250,)
        corrected = self.SmoothMin((raw, self.Join(allowed[None,:], self.emb, temp).squeeze()), temp, axis=0)
        return corrected, raw

    def _energy(self, new, old, aux):
        raw           = aux
        dynamic_error = float(Sum(Abs(new - old) ** 2))
        sensory_error = float(Sum(Abs(raw - new) ** 2))
        return dynamic_error + sensory_error

    def retrieve(self, q):
        return self.fp.perturb(q)
