"""
Attention layer. Implements transformer-style attention using the fuzzy
semiring operators from tensor.py and the embedding operations from embed.py.
GramMatrix provides the dot-product similarity, SoftMax normalises it, and
witnesses record which paths drove each attention score — giving provenance
over attention that standard transformers lack. MultiHeadAttention runs
parallel heads over different subspaces of the embedding. LayerNorm and
FeedForward complete the standard transformer block.
"""

import numpy as np
from embed import Embed
from algebra import *
from functools import reduce

class Transformer(Embed):
    def __init__(self, temp=1.0):
        super().__init__(temp)
        self._W = {} 
        self._PE = None
        self._block_counter = 0 

    def _init_weights(self, d, heads, N):
        self._W = {
            'W_o': np.random.randn(d, d) * 0.01,
            'W1':  np.random.randn(d, d * 4)     * 0.01
        }

    def _get_pe(self, d, N):
        if self._PE is None or self._PE.shape != (N, d):
            self._PE = self.PositionEncode(d, N)
        return self._PE

    def _witness_weight_update(self, witnesses, temp):
        y_scores = {}
        for (_, y, _), s in witnesses.items():
            y_scores.setdefault(y, []).append(s)
        scale = np.zeros(self._W['W1'].shape[0])
        for y, scores in y_scores.items():
            if y < scale.shape[0]:
                scale[y] = self.SmoothMax(np.array(scores), temp, axis=0)
        scale = scale / (scale.max() + 1e-9)
        self._W['W1'] *= (1 + scale[:, None])
        
    def Attention(self, M, temp):
        scores, witnesses = self.GramMatrix(M, temp, semiring='fuzzy')
        weights = self.SoftMax(scores, temp, axis=-1)
        result = self.Join(weights, M, temp, semiring='fuzzy')[0]  # N×d
        return result, witnesses

    # Run h parallel attention heads over different embedding subspaces
    # then concatenate results — each head attends to different structure
    def MultiHeadAttention(self, M, emb, temp, heads=8):
        d = emb.shape[1]
        head_dim = d // heads
        outputs = []
        all_witnesses = {}
        for i in range(heads):
            M_proj = M[:, i*head_dim:(i+1)*head_dim]
            weights, witnesses = self.Attention(M_proj, temp)
            outputs.append(weights)
            all_witnesses.update(witnesses)
        return np.concatenate(outputs, axis=-1), all_witnesses

    # Two projections with a nonlinearity between them
    # Project up → activate → Project down
    def FeedForward(self, M, W1, temp):
        hidden, _ = self.Join(M, W1, temp)
        hidden = self.Softplus(hidden, temp)
        result, _ = self.Join(hidden, W1.T, temp)
        return result
    
    # Positional encoding
    def PositionEncode(self, d, N):
        pos = np.arange(N)
        d_even = d + 1 if d % 2 != 0 else d
        i = np.arange(d_even // 2)
        PE = np.zeros((N, d_even))
        PE[:, 0::2] = np.sin(pos[:, None] / 10000 ** (2 * i / d_even))
        PE[:, 1::2] = np.cos(pos[:, None] / 10000 ** (2 * i / d_even))
        return PE[:, :d]
    
    # Full transformer encoder block:
    # MultiHeadAttention → residual → LayerNorm → FeedForward → residual → LayerNorm
    def TransformerBlock(self, M, emb, temp, heads=8):
        d, N = M.shape[1], M.shape[0]
        if not self._W:
            self._init_weights(d, heads, N)
        # Self-attention sublayer with residual
        attn, witnesses = self.MultiHeadAttention(M, emb, temp, heads)
        self._cache_witnesses(witnesses)
        self._block_counter += 1
        attn = attn @ self._W['W_o']
        M = self.LayerNorm(M + attn)
        # Feed-forward sublayer with residual
        ff = self.FeedForward(M, self._W['W1'], temp)
        M = self.LayerNorm(M + ff)
        self._witness_weight_update(witnesses, temp)
        self._W['W1'] /= (Max(Abs(self._W['W1'])) + 1e-9)
        return M, witnesses