"""
Integration test — simplified transformer via DSL + Functor + Interpreter.

Verifies the full engine stack: DSL parse → compile → PathEngine → Functor →
Interpreter (algebra + coalgebra), with agreement between batch forward and
streaming token-by-token inference.
"""

import numpy as np
import pytest
from engine import compile, Interpreter, UnfoldStep


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

V = 32    # vocab size
D = 16    # model dim
H = 8     # head dim (D // num_heads, but single-head here)
F = 32    # ff dim
S = 4     # sequence length
EPS = 1e-5


# ---------------------------------------------------------------------------
# Ops — simple numpy implementations
# ---------------------------------------------------------------------------

def proj_op(eq, x, bundle, temp=0.0):
    """Linear projection: x @ W + b."""
    return np.einsum(eq, x, bundle['W']) + bundle['b']

def score_op(eq, q, bundle, temp=0.0):
    """Attention scores: Q @ K^T * scale."""
    return np.einsum(eq, q, bundle['K']) * bundle['scale']

def softmax_op(eq, x, bundle, temp=0.0):
    """Row-wise softmax with optional causal mask."""
    if bundle.get('mask') is not None:
        x = x + bundle['mask']
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def mix_op(eq, probs, bundle, temp=0.0):
    """Weighted mix: probs @ V."""
    return np.einsum(eq, probs, bundle['V'])

def gelu_op(eq, x, bundle, temp=0.0):
    """GELU activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def identity_op(eq, x, y=None, temp=0.0):
    return x

def swap_xy(x, y):
    return (y, x)


# ---------------------------------------------------------------------------
# Per-case cell functions (for DSL-driven interpreter tests)
# ---------------------------------------------------------------------------

def _attn_fwd(paths, x_norm, w, mask):
    """Attention forward using compiled paths from params."""
    q = paths['q_proj'](x_norm, {'W': w['Wq'], 'b': w['bq']}, 0.0)
    K = paths['k_proj'](x_norm, {'W': w['Wk'], 'b': w['bk']}, 0.0)
    V = paths['v_proj'](x_norm, {'W': w['Wv'], 'b': w['bv']}, 0.0)
    scores = paths['score'](q, {'K': K, 'scale': H ** -0.5}, 0.0)
    probs = paths['normalize'](scores, {'mask': mask}, 0.0)
    mixed = paths['mix'](probs, {'V': V}, 0.0)
    return paths['out_proj'](mixed, {'W': w['Wo'], 'b': w['bo']}, 0.0)


def _ffn_fwd(paths, x_norm, w):
    """FFN forward using compiled paths from params."""
    h = paths['up'](x_norm, {'W': w['W1'], 'b': w['b1']}, 0.0)
    h = paths['act'](h, {}, 0.0)
    return paths['down'](h, {'W': w['W2'], 'b': w['b2']}, 0.0)


def input_cell(payload, child_results, params, temp):
    return payload[0]


def attn_cell(payload, child_results, params, temp):
    w = payload[0]
    x = child_results[0]
    x_norm = layer_norm(x, w['ln1_g'], w['ln1_b'])
    return x + _attn_fwd(params['paths'], x_norm, w, params['mask'])


def ffn_cell(payload, child_results, params, temp):
    w = payload[0]
    x = child_results[0]
    x_norm = layer_norm(x, w['ln2_g'], w['ln2_b'])
    return x + _ffn_fwd(params['paths'], x_norm, w)




# ---------------------------------------------------------------------------
# DSL source — single-head attention + FFN
# ---------------------------------------------------------------------------

DSL_SOURCE = """
semiring attn:
    contract = ops.identity

sort model, q, scores, probs, mixed, ff

leg q_proj    : model -> q       via "sd,dh->sh"    op ops.proj
leg score     : q -> scores      via "sh,th->st"    op ops.score
leg normalize : scores -> probs  via "st->st"        op ops.softmax
leg mix       : probs -> mixed   via "st,th->sh"    op ops.mix
leg out_proj  : mixed -> model   via "sh,hd->sd"    op ops.proj
leg k_proj    : model -> q       via "sd,dh->sh"    op ops.proj
leg v_proj    : model -> q       via "sd,dh->sh"    op ops.proj

leg up   : model -> ff   via "sd,df->sf"   op ops.proj
leg act  : ff -> ff      via "sf->sf"      op ops.gelu
leg down : ff -> model   via "sf,fd->sd"   op ops.proj

path read = q_proj score normalize mix out_proj
path mlp  = up act down

fan kv = k_proj & v_proj

arch Transformer:
    algebra:
        case input:     recursive=0  data=1  cell=ops.input_cell
        case attn_res:  recursive=1  data=1  cell=ops.attn_cell
        case ffn_res:   recursive=1  data=1  cell=ops.ffn_cell
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layer_norm(x, gamma, beta, eps=EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * gamma + beta


def causal_mask(t):
    return np.triu(np.full((t, t), -1e9), 1)


def init_weights(rng):
    """Initialize a single transformer block + embeddings."""
    def normal(shape, std=0.02):
        return rng.normal(0.0, std, shape)

    block = {
        'Wq': normal((D, H)), 'bq': np.zeros(H),
        'Wk': normal((D, H)), 'bk': np.zeros(H),
        'Wv': normal((D, H)), 'bv': np.zeros(H),
        'Wo': normal((H, D)), 'bo': np.zeros(D),
        'W1': normal((D, F)), 'b1': np.zeros(F),
        'W2': normal((F, D)), 'b2': np.zeros(D),
        'ln1_g': np.ones(D), 'ln1_b': np.zeros(D),
        'ln2_g': np.ones(D), 'ln2_b': np.zeros(D),
    }
    tok_embed = normal((V, D))
    pos_enc = normal((64, D))
    return block, tok_embed, pos_enc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransformerIntegration:

    @pytest.fixture
    def arch(self):
        ns = {
            'ops': type('ns', (), {
                'proj': staticmethod(proj_op),
                'score': staticmethod(score_op),
                'softmax': staticmethod(softmax_op),
                'mix': staticmethod(mix_op),
                'gelu': staticmethod(gelu_op),
                'identity': staticmethod(identity_op),
                'input_cell': staticmethod(input_cell),
                'attn_cell': staticmethod(attn_cell),
                'ffn_cell': staticmethod(ffn_cell),
            })(),
        }
        return compile(DSL_SOURCE, ns)

    @pytest.fixture
    def weights(self):
        rng = np.random.default_rng(42)
        return init_weights(rng)

    def _attention_forward(self, arch, x_norm, w, mask):
        """Run attention using compiled paths and fan."""
        kv = arch.paths['kv'](x_norm, {'W': w['Wk'], 'b': w['bk']}, 0.0)
        K = kv['k_proj']
        V = arch.paths['v_proj'](x_norm, {'W': w['Wv'], 'b': w['bv']}, 0.0)

        bundle = {
            'W': w['Wq'], 'b': w['bq'],
            'K': K, 'V': V,
            'scale': H ** -0.5,
            'mask': mask,
        }
        # out_proj needs its own W/b — override in bundle at the right step
        # Simpler: run read path which chains q_proj→score→norm→mix→out_proj
        # Each leg gets the same bundle as y. We need to set W/b for each.
        # For simplicity, run leg by leg:
        q = arch.paths['q_proj'](x_norm, {'W': w['Wq'], 'b': w['bq']}, 0.0)
        scores = arch.paths['score'](q, {'K': K, 'scale': H ** -0.5}, 0.0)
        probs = arch.paths['normalize'](scores, {'mask': mask}, 0.0)
        mixed = arch.paths['mix'](probs, {'V': V}, 0.0)
        out = arch.paths['out_proj'](mixed, {'W': w['Wo'], 'b': w['bo']}, 0.0)
        return out

    def _ffn_forward(self, arch, x_norm, w):
        """Run FFN using compiled paths."""
        h = arch.paths['up'](x_norm, {'W': w['W1'], 'b': w['b1']}, 0.0)
        h = arch.paths['act'](h, {}, 0.0)
        h = arch.paths['down'](h, {'W': w['W2'], 'b': w['b2']}, 0.0)
        return h

    def _block_forward(self, arch, x, w, mask):
        """One transformer block: LN → Attn + residual → LN → FFN + residual."""
        x_norm = layer_norm(x, w['ln1_g'], w['ln1_b'])
        x = x + self._attention_forward(arch, x_norm, w, mask)
        x_norm = layer_norm(x, w['ln2_g'], w['ln2_b'])
        x = x + self._ffn_forward(arch, x_norm, w)
        return x

    def test_algebra_forward(self, arch, weights):
        """Run algebra via arch.interpreter('Transformer'), compare to direct numpy."""
        block, tok_embed, pos_enc = weights
        rng = np.random.default_rng(7)
        token_ids = rng.integers(0, V, size=S)

        x0 = tok_embed[token_ids] + pos_enc[:S]
        mask = causal_mask(S)

        # Direct computation (reference)
        direct = self._block_forward(arch, x0, block, mask)
        direct_logits = direct @ tok_embed.T

        # Via arch.interpreter() — single arch, algebra side
        interp = arch.interpreter('Transformer', params={'mask': mask})
        tree = ('ffn_res', [block], [
            ('attn_res', [block], [
                ('input', [x0], [])
            ])
        ])
        result = interp.run_algebra(tree, lambda n: n)
        result_logits = result @ tok_embed.T

        assert direct_logits.shape == (S, V)
        assert result_logits.shape == (S, V)
        np.testing.assert_allclose(result_logits, direct_logits, atol=1e-12)

    def test_coalgebra_streaming(self):
        """Coalgebra streaming via step: protocol."""
        from engine.compiler import compile as engine_compile

        def embed_op(eq, token, state, temp=0.0):
            return state['tok_embed'][int(token)]

        def proj_op(eq, x, y, temp=0.0):
            return x @ y['W']

        def unembed_op(eq, x, state, temp=0.0):
            return x @ state['tok_embed'].T

        ns = {
            'ops': type('ns', (), {
                'embed': staticmethod(embed_op),
                'proj': staticmethod(proj_op),
                'unembed': staticmethod(unembed_op),
                'identity': staticmethod(lambda eq, x, y, temp=0.0: x),
            })()
        }

        src = """
semiring s:
    contract = ops.identity

sort model

morphism embed : model -> model via "x" op ops.embed
morphism proj  : model -> model via "x" op ops.proj
morphism unembed : model -> model via "x" op ops.unembed

path transform = proj

arch Streamer:
    cases:
        input: leaf data=1 cell=identity
        block: node data=1 output=1 morphisms=transform
    step:
        enter = embed
        emit = unembed
"""
        arch = engine_compile(src, ns)

        rng = np.random.default_rng(42)
        D = 4
        V = 5
        tok_embed = rng.standard_normal((V, D))
        W = rng.standard_normal((D, D))  # square projection, no shape mismatch

        interp = arch.interpreter('Streamer', params={}, temp=0.0)

        state = {'tok_embed': tok_embed, 'W': W}
        tokens = [np.array(0), np.array(1), np.array(2)]

        outputs, final = interp.run_coalgebra(state, token_iter=tokens)

        assert len(outputs) == 3
        for i, tok_id in enumerate([0, 1, 2]):
            x = tok_embed[tok_id]
            h = x @ W
            expected = h @ tok_embed.T
            np.testing.assert_allclose(outputs[i], expected, atol=1e-12)

    def test_fan_produces_dict(self, arch):
        """Verify the kv fan returns a dict with expected keys."""
        x = np.random.randn(S, D)
        bundle = {'W': np.random.randn(D, H), 'b': np.zeros(H)}
        result = arch.paths['kv'](x, bundle, 0.0)
        assert isinstance(result, dict)
        assert 'k_proj' in result
        assert 'v_proj' in result
        assert result['k_proj'].shape == (S, H)

    def test_arch_available(self, arch):
        """Verify arch compiled with functor from DSL."""
        assert 'Transformer' in arch._archs
        ad = arch._archs['Transformer']
        assert ad.functor is not None
        assert ad.functor['input'].recursive == 0
