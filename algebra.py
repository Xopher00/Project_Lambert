"""
Foundation layer for Unified Algebra (Hehner). Defines the two primitive
operators — ∧ (min) and ∨ (max) — along with the arithmetic scaffolding
(Sum, Product, Log, Exp) that everything above this layer is built from.
Also encodes Hehner's divide-by-zero rule: x/0 = (x > 0), mapping the
singularity to a binary truth value rather than an error. All other modules
import from this one; nothing here depends on anything above it.
"""

import time
import numpy as np

Top = 1e9
Bottom = -1e9

def Max(*args, axis=None, keepdims=False):
    """Max: binary (x,y) or quantifier (f, axis=...)"""
    if len(args) == 1:
        return np.max(args[0], axis=axis, keepdims=keepdims)
    elif len(args) == 2:
        return np.maximum(args[0], args[1])
    else:
        np.maximum.reduce(np.array(args))

def Sum(args, axis=None, keepdims=False):
    """Sum over domain"""
    return np.sum(args, axis=axis, keepdims=keepdims)

def Implies(a, b):
    """
    Pseudocomplement for the min t-norm (Gödel implication):
    Least upper bound
    a α b = 1 if a <= b else b
    Works elementwise on numpy arrays.
    """
    return np.where(a <= b, Top, b)

def Refutes(a, b):
    """
    Greatest Lower bound, also called dual pseudocomplement
    a eps b = 0 if b <= a else b
    """
    return np.where(a >= b, Bottom, b)

def Log(args):
    """Log over args"""
    return np.log(args)

def Exp(args):
    """Euler's Number ^ args"""
    return np.exp(args)

def Abs(x):
    """Absolute value |x|"""
    return np.abs(x)

def Negate(x):
    """Negate: works on scalars, arrays, or tuples"""
    if isinstance(x, (tuple, list)):
        return tuple(-elem for elem in x)
    return -x

def Error(actual, predicted, base_threshold=0.05):
    error = np.abs(actual - predicted)
    threshold = np.clip(base_threshold - np.mean(error), 0.1, base_threshold)
    return error, threshold

def timed(fn):
    def wrapper(*args, **kwargs):
        t = time.time()
        result = fn(*args, **kwargs)
        print(f"  total_time={time.time()-t:.2f}s")
        return result
    return wrapper