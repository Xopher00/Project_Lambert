"""
relational_einsum.py — Semiring einsum for the max-min relational algebra.

Defines two semirings matching the core operations in tensor.py:

  Join semiring      (max-min):       addition = SmoothMax, multiplication = SmoothMin
  Residuate semiring (min-implies):   addition = SmoothMin, multiplication = Implies

Both collapse to exact hard operations at temp=0.

  Join(A, B)[x,z]      = SmoothMax_y SmoothMin(A[x,y], B[y,z])
  Residuate(A,C)[y,z]  = SmoothMin_i Implies(A[i,y], C[i,z])

Usage
-----
from torch_semiring_einsum import compile_equation
from core.relational_einsum import join_einsum_forward, residuate_einsum_forward

eq = compile_equation("ij,jk->ik")
C  = join_einsum_forward(eq, A, B, temp=0.0)        # exact Join
C  = join_einsum_forward(eq, A, B, temp=1.0)        # smooth Join

eq = compile_equation("ij,ik->jk")
B  = residuate_einsum_forward(eq, A, C, temp=0.0)   # exact Residuate

Notes
-----
- Inputs must be torch.Tensor.
- SmoothMax  = temp * log(sum(exp(x / temp)))       — LogSumExp
- SmoothMin  = -SmoothMax(-x)                       — De Morgan dual
- Implies(a, b) = Top if a <= b, else b             — Sanchez (1976) cite{sanchez1976}
- Bottom (-1e9) is the identity for SmoothMax.
- Top    ( 1e9) is the identity for SmoothMin.
"""

import typing

import torch

from torch_semiring_einsum import semiring_einsum_forward
from torch_semiring_einsum.equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE

_TOP    =  1e9
_BOTTOM = -1e9

# ---------------------------------------------------------------------------
# In-place primitives — hard (temp=0)
# ---------------------------------------------------------------------------

def _max_in_place(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.max(a, b, out=a)

def _min_in_place(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.min(a, b, out=a)

def _implies_in_place(a: torch.Tensor, b: torch.Tensor) -> None:
    """Implies(a, b) = Top if a <= b, else b.  Sanchez (1976) cite{sanchez1976}"""
    a.copy_(torch.where(a <= b, b.new_full((), _TOP), b))

# ---------------------------------------------------------------------------
# Block-reduction primitives — hard (temp=0)
# ---------------------------------------------------------------------------

if hasattr(torch, 'amax'):
    def _max_block(a: torch.Tensor, dims) -> torch.Tensor:
        return torch.amax(a, dim=list(dims)) if dims else a
else:
    def _max_block(a: torch.Tensor, dims) -> torch.Tensor:
        result = a
        for dim in reversed(dims):
            result = torch.max(result, dim=dim)[0]
        return result

if hasattr(torch, 'amin'):
    def _min_block(a: torch.Tensor, dims) -> torch.Tensor:
        return torch.amin(a, dim=list(dims)) if dims else a
else:
    def _min_block(a: torch.Tensor, dims) -> torch.Tensor:
        result = a
        for dim in reversed(dims):
            result = torch.min(result, dim=dim)[0]
        return result

# ---------------------------------------------------------------------------
# In-place primitives — smooth (temp > 0)
#
# SmoothMax(a, b) = temp * log(exp(a/temp) + exp(b/temp))
#                = temp * logaddexp(a/temp, b/temp)
#
# SmoothMin(a, b) = -SmoothMax(-a, -b)   (De Morgan dual)
# ---------------------------------------------------------------------------

def _make_smooth_max_in_place(temp: float):
    """SmoothMax addition for Join: a = temp * logaddexp(a/temp, b/temp)."""
    def _fn(a: torch.Tensor, b: torch.Tensor) -> None:
        a.copy_(temp * torch.logaddexp(a / temp, b / temp))
    return _fn

def _make_smooth_min_in_place(temp: float):
    """SmoothMin for both Join multiplication and Residuate addition."""
    def _fn(a: torch.Tensor, b: torch.Tensor) -> None:
        # SmoothMin(a, b) = -SmoothMax(-a, -b)
        a.copy_(-(temp * torch.logaddexp(-a / temp, -b / temp)))
    return _fn

# ---------------------------------------------------------------------------
# Block-reduction primitives — smooth (temp > 0)
#
# SmoothMax_block(a, dims) = temp * logsumexp(a / temp, dim=dims)
# SmoothMin_block(a, dims) = -temp * logsumexp(-a / temp, dim=dims)
# ---------------------------------------------------------------------------

def _make_smooth_max_block(temp: float):
    """SmoothMax reduction for Join: temp * logsumexp(a / temp, dim=dims)."""
    def _fn(a: torch.Tensor, dims) -> torch.Tensor:
        if not dims:
            return a
        return temp * torch.logsumexp(a / temp, dim=list(dims))
    return _fn

def _make_smooth_min_block(temp: float):
    """SmoothMin reduction for Residuate: -temp * logsumexp(-a / temp, dim=dims)."""
    def _fn(a: torch.Tensor, dims) -> torch.Tensor:
        if not dims:
            return a
        return -(temp * torch.logsumexp(-a / temp, dim=list(dims)))
    return _fn

# ---------------------------------------------------------------------------
# Join semiring  (max-min / SmoothMax-SmoothMin)
# ---------------------------------------------------------------------------

def join_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        temp: float = 0.0,
        block_size: typing.Union[int, AutomaticBlockSize] = AUTOMATIC_BLOCK_SIZE,
) -> torch.Tensor:
    r"""Join einsum under the max-min (SmoothMax-SmoothMin) semiring.

    At temp=0 computes the exact relational composition::

        result[i, k] = max_j  min(A[i, j], B[j, k])

    At temp>0, max and min are replaced by their smooth LogSumExp-based
    approximations (SmoothMax and SmoothMin from activations.py), matching
    the temperature-controlled behaviour of tensor.py's Join.

    :param equation: A pre-compiled equation (from ``compile_equation``).
    :param args: Input tensors matching the equation's input count.
    :param temp: Temperature. 0.0 = exact max/min; >0 = smooth.
    :param block_size: Block size for memory control.
    :return: Output tensor.
    """
    if temp == 0.0:
        def _callback(compute_sum):
            return compute_sum(_max_in_place, _max_block, _min_in_place)
    else:
        _add   = _make_smooth_max_in_place(temp)
        _sum   = _make_smooth_max_block(temp)
        _mul   = _make_smooth_min_in_place(temp)
        def _callback(compute_sum):
            return compute_sum(_add, _sum, _mul)
    return semiring_einsum_forward(equation, args, block_size, _callback)

# ---------------------------------------------------------------------------
# Residuate semiring  (min-implies / SmoothMin-implies)
# ---------------------------------------------------------------------------

def residuate_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        temp: float = 0.0,
        block_size: typing.Union[int, AutomaticBlockSize] = AUTOMATIC_BLOCK_SIZE,
) -> torch.Tensor:
    r"""Residuate einsum under the min-implies (SmoothMin-implies) semiring.

    At temp=0 computes the exact adjoint of Join::

        result[j, k] = min_i  implies(A[i, j], C[i, k])

    where ``implies(a, b) = Top`` if ``a <= b``, else ``b``.

    At temp>0, min is replaced by SmoothMin (De Morgan dual of LogSumExp).
    The implies multiplication remains exact — no smooth variant is defined.

    :param equation: A pre-compiled equation (from ``compile_equation``).
    :param args: Input tensors. First arg is the left relation A, second is
        the target C. Must match the equation's input count.
    :param temp: Temperature for the SmoothMin reduction. 0.0 = exact min.
    :param block_size: Block size for memory control.
    :return: Output tensor, with Top as the identity for uncontributed cells.
    """
    if temp == 0.0:
        def _callback(compute_sum):
            return compute_sum(_min_in_place, _min_block, _implies_in_place)
    else:
        _add = _make_smooth_min_in_place(temp)
        _sum = _make_smooth_min_block(temp)
        def _callback(compute_sum):
            return compute_sum(_add, _sum, _implies_in_place)
    return semiring_einsum_forward(equation, args, block_size, _callback)
