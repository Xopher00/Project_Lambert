"""
Training layer. Consumes the outputs of Join — results and witnesses —
to perform sparse gradient updates. Sits above Tensor in the stack,
keeping learning logic separate from relational operations. As embedding
integration develops, this module will also handle embedding-aware
training procedures.
"""

import numpy as np
from algebra import *
from tree import Tree

from transformer import Transformer

class Train(Transformer):
    def Loss(self, actual, predicted):
        error, _ = self.Error(actual, predicted)
        loss = np.mean(error ** 2)
        gradient = 2 * (predicted - actual)
        return gradient, loss

    def Update(self, gradient, witnesses, temp, axis):
        # weight each witness by its normalised contribution — §⟨y· witnesses[x,y,z]⟩
        witness_weights = self.SoftMax(witnesses, temp=0.1, axis=1) 
        update = Sum(gradient[:, None, :] * witness_weights, axis=axis)
        return update
    
    def SGD(self, params, grads, lr):
        """Tree-aware gradient descent: params - lr * grads"""
        paired = Tree.zip(params, grads)
        return Tree.map(lambda p: p.fst - lr * p.snd, paired)

    def LearnEmbedPC(self, R_dict):
        R_all  = Max(list(R_dict.values()))
        # R_all  = np.maximum.reduce(list(R_dict.values()))
        emb, _ = self.Embed(R_all)
        norms  = np.linalg.norm(emb, axis=1, keepdims=True)
        emb   /= np.where(norms > 0, norms, 1.0)
        R_star = self.Closure(R_all, temp=1.0)
        EmbR_all = self.Project(R_star, emb)
        EmbR = {r: self.Project(R, emb) for r, R in R_dict.items()}
        EmbR['_all'] = EmbR_all
        return emb, EmbR

