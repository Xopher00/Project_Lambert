import numpy as np
from core.algebra import *
from core.tensor import Tensor
from core.fixpoint import FixpointIterator

class Concept(Tensor):

    def __init__(self, R, temp=1.0, eps=1e-3, max_iters=20):
        super().__init__()
        self.R = R
        self.eps = eps
        self.fp = FixpointIterator(
            f         = self._step,
            state0    = R[:, 0].copy(),
            temp      = temp,
            eps       = eps,
            max_iters = max_iters,
        )

    def _step(self, a, temp):
        active = np.flatnonzero(a > 0)
        R_active = self.R[active, :]
        a_active = a[active]
        result = np.zeros_like(a)
        result[active] = self.Recall(a_active, R_active, temp)
        return result, None

    def scores(self):
        return np.where(self.fp.state > self.eps, self.fp.state, 0)

    def retrieve(self, seed):
        if np.all(seed == 0):
            return np.zeros_like(seed)
        self.fp.perturb(seed)
        return self.fp.state


class Lattice: