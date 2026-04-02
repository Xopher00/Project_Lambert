"""
Relational operations layer. Implements three core operations on matrices:

- Join: for each output cell (x, z), finds the best intermediate node y
  by taking the min of A[x,y] and B[y,z], then the max over all y.
- Residuate: given A and C, finds the greatest B such that A composed
  with B stays within C.
- Closure: iterates Join to fixpoint, computing transitive reachability
  across the relation.

Join and Residuate are implemented via torch-semiring-einsum using the
(SmoothMax, SmoothMin) and (SmoothMin, Implies) semirings respectively.
Computation runs on GPU when available. Inputs and outputs are numpy arrays;
conversion happens once at the boundary of each public method.
"""

import numpy as np
import torch
import torch_semiring_einsum as tse

from core.algebra import *
from core.activations import Activations
from core.fixpoint import FixpointIterator

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _block_size(n, p, k):
    """
    Compute a safe tse block size given output dimensions n and p.

    tse's intermediate term tensor has shape (n, p, block_size). Its automatic
    sizing ignores the output dimensions, so we derive block_size from the
    caller-supplied memory budget: block_size = budget / (n * p * sizeof(float32)).
    """
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0]
        budget = int(free * 0.5)  # use 50% of free memory
    else:
        budget = 2**28
    block = max(1, budget // (n * p * 4))
    return min(block, k)


class Tensor(Activations):
    """
    Relational operations built on top of Activations.

    Inherits all temperature-controlled activation functions and adds
    matrix-level operations: Join, Residuate, and Closure.

    All methods accept and return numpy arrays. Internally, computation
    is performed on torch tensors on the available device (GPU if present).
    """

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(_DEVICE)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32).to(_DEVICE)

    # v (y: A[x,y] ∧ B[y,z])
    def Join(self, Tensor_A, Tensor_B, temp, equation='ik,kj->ij'):
        """
        Relational composition of two matrices.

        For each output cell (x, z), finds the best intermediate node y by
        taking the min of A[x, y] and B[y, z], then taking the max over all y:

            result[x, z] = SmoothMax_y( SmoothMin(A[x, y], B[y, z]) )

        Implemented via the (SmoothMax, SmoothMin) semiring over the blocked
        einsum 'ik,kj->ij'. Degenerates to hard (max, min) as temp → 0.

        Parameters
        ----------
        Tensor_A : ndarray or Tensor, shape (n, m)
            Left relation matrix.
        Tensor_B : ndarray or Tensor, shape (m, p)
            Right relation matrix.
        temp : float
            Temperature passed to SmoothMin and SmoothMax.
        equation : str, optional
            Einsum equation. Default is 'ik,kj->ij'.

        Returns
        -------
        ndarray, shape (n, p)
            The composed relation matrix.
        """
        T = float(temp)
        A = self._to_device(Tensor_A)
        B = self._to_device(Tensor_B)
        eq = tse.compile_equation(equation)

        def func(compute_sum):
            def mul_in_place(a, b):
                a.copy_(self.SmoothMin((a, b), T))
            def add_in_place(a, b):
                a.copy_(self.SmoothMax((a, b), T))
            def sum_block(a, dims):
                return self.SmoothMax(a, T, axis=dims) if dims else a
            return compute_sum(add_in_place, sum_block, mul_in_place)

        return tse.semiring_einsum_forward(eq, [A, B], _block_size(A.shape[0], B.shape[1], A.shape[1]), func)

    # ∧ (A[x,y]-¹ α C[x,z]) <= B[y,z]
    def Residuate(self, Tensor_A, Tensor_C, temp):
        """
        The adjoint of Join. Given A and C, finds the greatest B such that
        Join(A, B) stays within C.

        For each cell (y, z), computes the tightest upper bound that every
        row i of A places on B[y, z], using the Implies operation:

            B[y, z] = SmoothMin_i( Implies(A[i, y], C[i, z]) )

        where Implies(a, b) = Top if a ≤ b, else b.

        Implemented via the (SmoothMin, Implies) semiring over the blocked
        einsum 'iy,iz->yz'.

        Parameters
        ----------
        Tensor_A : ndarray or Tensor, shape (n, m)
            The left relation matrix.
        Tensor_C : ndarray or Tensor, shape (n, p)
            The target relation matrix. Must share the row dimension with A.
        temp : float
            Temperature passed to SmoothMin.

        Returns
        -------
        ndarray, shape (m, p)
            The greatest B satisfying Join(A, B) ≤ C.

        References
        ----------
        Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
        *Information and Control*, 30, 38–48. Theorem 5.  cite{sanchez1976}
        """
        T = float(temp)
        A = self._to_device(Tensor_A)
        C = self._to_device(Tensor_C)
        eq = tse.compile_equation('iy,iz->yz')

        def func(compute_sum):
            def mul_in_place(a, b):
                a.copy_(Implies(a, b))
            def add_in_place(a, b):
                a.copy_(self.SmoothMin((a, b), T))
            def sum_block(a, dims):
                return self.SmoothMin(a, T, axis=dims) if dims else a
            return compute_sum(add_in_place, sum_block, mul_in_place)

        return tse.semiring_einsum_forward(eq, [A, C], _block_size(A.shape[1], C.shape[1], A.shape[0]), func)

    def Closure(self, E, R=None, temp=None, max_iters=100, eps=1e-3):
        """
        Compute the transitive closure of relation E.

        Iterates Join to fixpoint: at each step, extends the current relation
        R by one hop through E, merges the result back with E, then clips it
        down using Residuate to ensure no inferred relation exceeds what E can
        justify. Repeats until the relation stops changing.

        Diagonal entries are zeroed at each step to prevent self-loops from
        accumulating.

        Parameters
        ----------
        E : ndarray, shape (n, n)
            The base relation matrix.
        R : ndarray, shape (n, n), optional
            Starting point for iteration. Defaults to a copy of E.
        temp : float, optional
            Temperature passed to Join, SmoothMax, and Residuate.
        max_iters : int, optional
            Maximum number of iterations. Default is 100.
        eps : float, optional
            Convergence threshold. Default is 1e-3.

        Returns
        -------
        ndarray, shape (n, n)
            The converged closure of E.
        """
        if R is None: R = E.clone() if isinstance(E, torch.Tensor) else E.copy()
        E_t = self._to_device(E)

        def _f(R, temp):
            J            = self.Join(R, E_t, temp)
            Rn           = self.SmoothMax((J, E_t), temp, axis=0)
            Rn.fill_diagonal_(0)
            R_allowed    = self.Residuate(Rn, E_t, temp)
            Rn_corrected = self.SmoothMin((Rn, R_allowed), temp, axis=0)
            return Rn_corrected, Rn  # aux = Rn (before correction)

        fp = FixpointIterator(f=_f, state0=self._to_device(R), eps=eps, max_iters=max_iters)
        result = fp.run()
        if fp.energy == 0:
            print(f"✓ CONVERGED at iteration {fp._iter}")
        return result
