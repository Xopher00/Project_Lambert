"""
Fixpoint iterator abstraction. A FixpointIterator wraps any operator f and
iterates it until energy reaches zero (fixpoint). Energy and temperature are
first-class attributes — energy measures distance from fixpoint, temperature
is derived from energy and current state and controls smoothness of operators.

Both Closure (tensor.py) and _concept_fixpoint (embed.py) are special cases:
they differ only in f and energy_fn, not in how the iteration or temperature
update works.
"""

import numpy as np
from algebra import Abs, Sum, Log, Bottom


class FixpointIterator:

    def __init__(self, f, state0, eps=1e-3, max_iters=100, temp=1.0):
        self.f         = f          # (state, temp) -> new_state  OR  (new_state, aux)
        self.energy_fn = self.default_energy  # (new_state, old_state, aux) -> float
        self.state     = state0.copy()
        self.energy    = Bottom
        self.temp      = temp
        self._init_temp = temp
        self.eps       = eps
        self.max_iters = max_iters
        self._iter     = 0
        self._history = []

    @staticmethod
    def default_energy(new, old, aux):
        dynamic_error = Sum(Abs(new - old) ** 2)
        if aux is None:
            return dynamic_error
        sensory_error = Sum(Abs(aux - new) ** 2)
        return dynamic_error + sensory_error

    def _update_temp(self, old_state):
        log_mean = np.mean(Log(np.clip(old_state, self.eps, 1.0)))
        if log_mean != 0:
            self.temp = Abs(-self.energy / (old_state.size * log_mean))

    def step(self):
        old    = self.state
        result = self.f(self.state, self.temp)
        if isinstance(result, tuple):
            new, aux = result
        else:
            new, aux = result, None
        self.energy = self.energy_fn(new, old, aux)
        self._update_temp(old)
        self.state = new
        self._iter += 1
        self._record()
        return self.energy <= self.eps

    def run(self, verbose=False):
        for _ in range(self.max_iters):
            converged = self.step()
            if verbose:
                print(f"  iter {self._iter:3d}  energy={self.energy:.6f}  temp={self.temp:.6f}")
            if converged:
                self.energy = 0
                if verbose:
                    print(f"  converged at iter {self._iter}")
                break
        return self.state

    def perturb(self, new_state, verbose=False):
        self.state = new_state.copy()
        self.energy = Bottom
        self._iter  = 0
        self.temp   = self._init_temp
        return self.run(verbose=verbose)

    def __repr__(self):
        return (f"FixpointIterator(iter={self._iter}, "
                f"energy={self.energy:.4f}, temp={self.temp:.4f})")
