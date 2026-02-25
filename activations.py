"""
Smooth approximation layer. Replaces the sharp corners of ∧ and ∨ with
differentiable analogues — SmoothMin and SmoothMax — via the LogSumExp
identity. Temperature T is the single dial controlling this: at T=0 the
functions collapse back to exact UA min/max, and at T>0 they become
continuous and gradient-friendly. SoftMax sits on top of LogSumExp as a
normalised distribution over candidates, used later by provenance to
select the most likely witness for each reasoning step.
"""

import numpy as np
from scipy.special import logsumexp
from algebra import *

class Activations:
    def __init__(self, temp=1.0):
        self.temp = temp
        self._witness_cache = []
        self._witnesses = {}

    # ∨f  ≤  T × ln(+(exp(f/T)))  ≤  ∨f + T × ln(#f)
    # Instead of your complex LogSumExp, just:
    def LogSumExp(self, x, temp, axis=None, keepdims=False):
        temp = Abs(temp)
        # Binary fast path
        if isinstance(x, (tuple, list)) and len(x) == 2:
            a, b = x
            if temp == 0:
               return Max(a, b)           
            return temp * np.logaddexp(a/temp, b/temp)
        # Array path - use scipy
        if temp == 0:
            return Max(x, axis=axis, keepdims=keepdims)
        return temp * logsumexp(x/temp, axis=axis, keepdims=keepdims)
    
    # Use as an alias for LSE
    # T × ln(+(exp(f/T))) -> smooth ∨f
    def SmoothMax(self, x, temp, axis=0):
        return self.LogSumExp(x, temp, axis)
    
    # -T × ln(+(exp(-f/T))) -> smooth ∧f
    def SmoothMin(self, x, temp, axis=0):                                                                                                                         
        # De Morgan's Duality Law:  -(x ∨ y) = -x ∧ -y
        return -self.LogSumExp(Negate(x), temp, axis=axis)

    # x ∨ 0  ≤  T × ln(1 + exp(x/T))  ≤  (x ∨ 0) + T × ln 2
    def Softplus(self, x, temp, axis=0):
        return self.LogSumExp((0.0, x), temp, axis=axis)

    # x ∨ 0
    def Relu(self, x, axis=0):
        return self.Softplus(x, temp=0.0, axis=axis)

    # exp(f n / T) / +(exp(f/T))
    def SoftMax(self, x, temp, axis):
        if temp == 0:
            return Max(x) 
        # Binary/sigmoid mode without stacking: softmax([x, 0])[:, 0]
        if axis is None:
            lse = self.Softplus(x, temp, axis=axis)
        else:
            # General case (also covers temp == 0 via your division override)
            lse = self.LogSumExp(x, temp, axis=axis, keepdims=True)
        return Exp((x - lse) / temp)
    
    def SoftMin(self, x, temp, axis):
        return -self.SoftMax(Negate(x), temp, axis=axis)

    # Normalize each row to zero mean, unit variance
    # Stabilizes activations between layers
    def LayerNorm(self, M, eps=1e-8):
        mean = np.mean(M, axis=-1, keepdims=True)
        std  = np.std(M,  axis=-1, keepdims=True)
        return (M - mean) / (std + eps)