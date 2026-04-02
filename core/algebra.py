"""
The bottom layer. Every other module imports from this one.

Defines the core operation — max (∨) — plus basic arithmetic helpers
(Sum, Log, Exp, Abs, Negate) and the logical operator Implies borrowed
from fuzzy set theory.

Top and Bottom mark the endpoints of whatever ordered domain the system
is working in. In boolean logic they are True and False; on the real
number line they are +∞ and −∞. Here they are set to 1e9 and -1e9 as
practical stand-ins for infinity.
"""

import numpy as np
import torch

Top = 1e9
Bottom = -1e9

def Max(*args, axis=None, keepdims=False):
    """
    Return the maximum of the inputs.

    Can be called two ways:

    - ``Max(a, b)`` — elementwise maximum of two arrays.
    - ``Max(array, axis=...)`` — reduce an array along an axis.

    Parameters
    ----------
    *args : array-like
        Either one array (for reduction) or two arrays (for elementwise max).
    axis : int, optional
        Axis to reduce along. Only used when a single array is passed.
    keepdims : bool, optional
        If True, keep reduced axes as size-1 dimensions.

    Returns
    -------
    ndarray or Tensor
        The maximum values.
    """
    if len(args) == 1:
        x = args[0]
        if isinstance(x, torch.Tensor):
            return torch.amax(x, dim=axis, keepdim=keepdims)
        return np.max(x, axis=axis, keepdims=keepdims)
    elif len(args) == 2:
        a, b = args
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.maximum(a, b)
        return np.maximum(a, b)
    else:
        if isinstance(args[0], torch.Tensor):
            return torch.amax(torch.stack(list(args)), dim=0)
        return np.maximum.reduce(np.array(args))

def Sum(args, axis=None, keepdims=False):
    """
    Add up the elements of an array.

    Parameters
    ----------
    args : array-like
        The array to sum.
    axis : int, optional
        Axis to sum along. If None, sums all elements.
    keepdims : bool, optional
        If True, keep reduced axes as size-1 dimensions.

    Returns
    -------
    ndarray or Tensor or scalar
        The sum.
    """
    if isinstance(args, torch.Tensor):
        return torch.sum(args, dim=axis, keepdim=keepdims if axis is not None else False)
    return np.sum(args, axis=axis, keepdims=keepdims)

def Implies(a, b):
    """
    Fuzzy implication: given two values a and b, ask "does a imply b?"

    Returns Top (∞) if a <= b, meaning a is no stronger than b so the
    implication holds without restriction. Returns b otherwise, capping
    the result at the weaker value.

    Parameters
    ----------
    a : array-like
        The antecedent (the "if" side).
    b : array-like
        The consequent (the "then" side).

    Returns
    -------
    ndarray or Tensor
        Top where a <= b, otherwise b.

    References
    ----------
    Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
    *Information and Control*, 30, 38–48. Section 6, the α operation.  cite{sanchez1976}
    """
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.where(a <= b, torch.full_like(b, Top), b)
    return np.where(a <= b, Top, b)

def Log(args):
    """
    Natural logarithm (base e), applied elementwise.

    Parameters
    ----------
    args : array-like
        Input values. Must be positive.

    Returns
    -------
    ndarray or Tensor
        The natural logarithm of each element.
    """
    if isinstance(args, torch.Tensor):
        return torch.log(args)
    return np.log(args)

def Exp(args):
    """
    Exponential function (e raised to the power of each element), applied elementwise.

    Parameters
    ----------
    args : array-like
        Input values.

    Returns
    -------
    ndarray or Tensor
        e ** args, elementwise.
    """
    if isinstance(args, torch.Tensor):
        return torch.exp(args)
    return np.exp(args)

def Abs(x):
    """
    Absolute value, applied elementwise.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    ndarray or Tensor
        The absolute value of each element.
    """
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    return np.abs(x)

def Negate(x):
    """
    Negate the input.

    Handles scalars, numpy arrays, torch tensors, and tuples or lists
    of arrays — negating each element in the collection individually.

    Parameters
    ----------
    x : scalar, array-like, tuple, or list
        The value or values to negate.

    Returns
    -------
    scalar, ndarray, Tensor, or tuple
        The negated value(s).
    """
    if isinstance(x, (tuple, list)):
        return tuple(-elem for elem in x)
    return -x
