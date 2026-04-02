"""
Smooth activation layer. Every function here is built on LogSumExp.

Temperature T controls the sharpness: at T=0 each function collapses to
its exact hard counterpart (max, min, step). At T>0 the output is a
smooth, differentiable curve with a known bounded gap to the hard value.
The gap shrinks back to zero as T approaches zero.
"""

import numpy as np
from scipy.special import logsumexp
from core.algebra import *

class Activations:
    """
    Collection of temperature-controlled activation functions.

    All methods are stateless except for the witness tracking attributes,
    which are used by the layer above this one.

    Parameters
    ----------
    temp : float, optional
        Default temperature. Default is 1.0.
    """
    def __init__(self, temp=1.0):
        self.temp = temp
        self._witness_cache = []
        self._witnesses = {}
        self.tracking = False

    # ∨f  ≤  T × ln(+(exp(f/T)))  ≤  ∨f + T × ln(#f)
    def LogSumExp(self, x, temp, axis=None, keepdims=False):
        """
        Tempered log-sum-exp: T × ln(∑ exp(x / T)).

        The fundamental building block for all other functions in this class.
        Approximates the maximum of x from above, with a bounded gap:

            max(x)  ≤  T × ln(∑ exp(x/T))  ≤  max(x) + T × ln(len(x))

        At T=0, returns the exact maximum. At T>0, returns a smooth
        approximation that is larger than the maximum by at most T × ln(len(x)).

        Parameters
        ----------
        x : array-like or 2-tuple of arrays
            Input values. If a 2-tuple, a fast binary path is used.
        temp : float
            Temperature. Must be non-negative.
        axis : int, optional
            Axis to reduce along.
        keepdims : bool, optional
            If True, keep reduced axes as size-1 dimensions.

        Returns
        -------
        ndarray or scalar
            T × ln(∑ exp(x / T)), or max(x) when T=0.

        References
        ----------
        Nesterov, Y. (2005). Smooth minimization of non-smooth functions.
        *Mathematical Programming, Series A*, 103, 127–152.
        """
        temp = Abs(temp)
        # Binary fast path
        if isinstance(x, (tuple, list)) and len(x) == 2:
            a, b = x
            if temp < 1e-12:
               return Max(a, b)           
            return temp * np.logaddexp(a/temp, b/temp)
        # Array path - use scipy
        if temp == 0 or temp is None:
            return Max(x, axis=axis, keepdims=keepdims)
        return temp * logsumexp(x/temp, axis=axis, keepdims=keepdims)
    
    # Use as an alias for LSE
    # T × ln(+(exp(f/T))) -> smooth ∨f
    def SmoothMax(self, x, temp, axis=0):
        """Smooth maximum via LogSumExp. An alias — see LogSumExp for full details."""
        return self.LogSumExp(x, temp, axis)
    
    # -T × ln(+(exp(-f/T))) -> smooth ∧f
    def SmoothMin(self, x, temp, axis=0):
        """
        Smooth approximation of the minimum, via LogSumExp.

        Derived from SmoothMax by De Morgan's duality law: the minimum of x
        is the negation of the maximum of the negation of x.

            SmoothMin(x) = -SmoothMax(-x)

        At T=0 returns the exact minimum. At T>0 returns a smooth
        approximation slightly below the minimum, with a bounded gap.

        Parameters
        ----------
        x : array-like or 2-tuple of arrays
            Input values.
        temp : float
            Temperature. Must be non-negative.
        axis : int, optional
            Axis to reduce along. Default is 0.

        Returns
        -------
        ndarray or scalar
            The smooth minimum of x.
        """
        # De Morgan's Duality Law:  -(x ∨ y) = -x ∧ -y
        return -self.LogSumExp(Negate(x), temp, axis=axis)

    # x ∨ 0  ≤  T × ln(1 + exp(x/T))  ≤  (x ∨ 0) + T × ln 2
    def Softplus(self, x, temp, axis=0):
        """
        Smooth approximation of max(x, 0), via LogSumExp over (0, x).

        Satisfies the bound:

            max(x, 0)  ≤  T × ln(1 + exp(x/T))  ≤  max(x, 0) + T × ln 2

        The gap is largest at x=0 and shrinks to zero as T approaches zero.

        Parameters
        ----------
        x : array-like
            Input values.
        temp : float
            Temperature. Must be non-negative.
        axis : int, optional
            Axis to reduce along. Default is 0.

        Returns
        -------
        ndarray or scalar
            The smooth approximation of max(x, 0).
        """
        return self.LogSumExp((0.0, x), temp, axis=axis)

    # x ∨ 0
    def Relu(self, x, axis=0):
        """Exact max(x, 0). The T=0 special case of Softplus."""
        return self.Softplus(x, temp=0.0, axis=axis)

    # exp(f n / T) / +(exp(f/T))
    def SoftMax(self, x, temp, axis):
        """
        Distribute the input as shares that sum to 1.

        Each element's share is exp(x/T) / ∑ exp(x/T). At T=0 returns the
        hard maximum. At high T every element gets an equal share. At low T
        the largest element dominates.

        When axis is None, operates in binary/sigmoid mode: computes each
        element's share relative to 0, equivalent to sigmoid(x).

        Parameters
        ----------
        x : array-like
            Input values.
        temp : float
            Temperature. At T=0 returns the hard maximum.
        axis : int or None
            Axis to normalise along. If None, uses binary/sigmoid mode.

        Returns
        -------
        ndarray
            Values in (0, 1) summing to 1 along the given axis.
        """
        if temp == 0:
            return Max(x, axis=axis, keepdims=False)
        # Binary/sigmoid mode without stacking: softmax([x, 0])[:, 0]
        if axis is None:
            lse = self.Softplus(x, temp, axis=axis)
        else:
            # General case (also covers temp == 0 via your division override)
            lse = self.LogSumExp(x, temp, axis=axis, keepdims=True)
        return Exp((x - lse) / temp)
    
    # -exp(-f n / T) / +(exp(-f/T))
    def SoftMin(self, x, temp, axis):
        """De Morgan dual of SoftMax. An alias — see SoftMax for full details."""
        return -self.SoftMax(Negate(x), temp, axis=axis)
