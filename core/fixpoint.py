"""
Fixpoint iteration infrastructure. Wraps any operator f and iterates it
until convergence.

Temperature is derived from energy at each step using a formula inspired
by the Boltzmann distribution in statistical mechanics: high energy means
high temperature, which makes the fuzzy relational operations in the layer
above explore more freely. As energy falls, temperature falls too, and the
system converges toward hard logical outcomes.

This dynamic is also inspired by predictive coding: energy measures how
much the current state still needs to change, and each iteration reduces
it by updating the state in the direction that minimises the error.
"""

import torch
from core.algebra import Abs, Sum, Log, Bottom


class FixpointIterator:
    """
    Iterates an operator f until the state stops changing.

    At each step, calls f(state, temp) to get a new state, measures the
    energy (how much the state changed), then derives a new temperature
    from that energy. High energy keeps temperature high, allowing fuzzy
    exploration. As the state settles, energy and temperature both fall,
    and the system converges toward hard logical outcomes.

    Parameters
    ----------
    f : callable(state, temp) -> new_state or (new_state, aux)
        The operator to iterate. May return an optional auxiliary value
        which is passed to energy_fn but otherwise ignored.
    state0 : ndarray
        The initial state.
    eps : float, optional
        Convergence threshold. Iteration stops when energy <= eps.
        Default is 1e-3.
    max_iters : int, optional
        Maximum number of iterations before stopping. Default is 100.
    temp : float, optional
        Initial temperature. Default is 1.0.
    """

    def __init__(self, f, state0, eps=1e-3, max_iters=100, temp=1.0):
        self.f         = f          # (state, temp) -> new_state  OR  (new_state, aux)
        self.energy_fn = self.default_energy  # (new_state, old_state, aux) -> float
        self.state     = state0.clone()
        self.energy    = Bottom
        self.temp      = temp
        self._init_temp = temp
        self.eps       = eps
        self.max_iters = max_iters
        self._iter     = 0
        self._history = []

    @staticmethod
    def default_energy(new, old, aux):
        """
        Measure how much the state changed in this iteration. 
        The system's loss function.

        Returns the sum of two error terms, inspired by predictive coding:

        - Dynamic error: how much the state changed from the previous step.
          When this is zero, the system has reached a fixpoint.
        - Sensory error: how far the raw prediction (aux) was from the
          corrected belief (new). Measures how much correction was needed
          this step.

        # Parametric error (KL divergence between current and prior parameters)
        # is not included. May become relevant when parameter learning is added.

        When aux is None, only the dynamic error is returned.

        Parameters
        ----------
        new : ndarray
            The state after this iteration.
        old : ndarray
            The state before this iteration.
        aux : ndarray or None
            The raw prediction before any correction, returned by f.

        Returns
        -------
        float
            Total energy. Zero means the state has not changed.
        """
        dynamic_error = Sum(Abs(new - old) ** 2)
        if aux is None:
            return dynamic_error
        sensory_error = Sum(Abs(aux - new) ** 2)
        return dynamic_error + sensory_error

    def _update_temp(self, old_state):
        """
        Derive the next temperature from the current energy and state.

        Uses a formula inspired by the Boltzmann distribution: temperature
        is proportional to energy divided by the mean log-magnitude of the
        state. When energy is high, temperature stays high, keeping the
        fuzzy operations exploratory. As energy falls toward zero,
        temperature falls too and the system converges toward hard outputs.

        Parameters
        ----------
        old_state : ndarray
            The state before this iteration. Used to compute the log-mean
            magnitude, which acts as a normalising factor.
        """
        log_mean = torch.mean(Log(torch.clamp(old_state, self.eps, 1.0)))
        if log_mean != 0:
            self.temp = Abs(-self.energy / (old_state.numel() * log_mean))

    def step(self):
        """
        Advance the iterator by one step.

        Calls f(state, temp), measures energy, updates temperature, and
        stores the new state. Returns True if energy has fallen below the
        convergence threshold.

        Returns
        -------
        bool
            True if the iterator has converged, False otherwise.
        """
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
        return self.energy <= self.eps

    def run(self, verbose=False):
        """
        Run the iterator until convergence or max_iters is reached.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints energy and temperature at each iteration.
            Default is False.

        Returns
        -------
        ndarray
            The final state.
        """
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
        """
        Reset the iterator to a new starting state and run to convergence.

        Useful for incremental updates: rather than constructing a new
        iterator from scratch, the existing one can be reseeded with a
        new state when new data arrives.

        Parameters
        ----------
        new_state : ndarray
            The new starting state.
        verbose : bool, optional
            If True, prints energy and temperature at each iteration.
            Default is False.

        Returns
        -------
        ndarray
            The final state after convergence.
        """
        self.state = new_state.clone()
        self.energy = Bottom
        self._iter  = 0
        self.temp   = self._init_temp
        return self.run(verbose=verbose)

    def __repr__(self):
        return (f"FixpointIterator(iter={self._iter}, "
                f"energy={self.energy:.4f}, temp={self.temp:.4f})")
