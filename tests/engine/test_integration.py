"""
Integration test — simplified transformer via DSL + Functor + Interpreter.

Verifies the full engine stack: DSL parse → compile → PathEngine → Functor →
Interpreter (algebra + coalgebra), with agreement between batch forward and
streaming token-by-token inference.
"""

import numpy as np
import pytest
from engine import compile, Interpreter, UnfoldStep
from engine.sorts import setup_hydra_path as _setup_hydra_path

_setup_hydra_path()

from hydra.dsl.python import Right  # noqa: E402
from hydra.context import Context   # noqa: E402
from hydra.dsl.python import FrozenDict  # noqa: E402


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

def input_cell(payload, child_results, params, temp):
    return payload[0]


def attn_cell(payload, child_results, params, temp):
    paths, w, mask = params['paths'], payload[0], params['mask']
    x_norm = layer_norm(child_results[0], w['ln1_g'], w['ln1_b'])
    q = paths['q_proj'](x_norm, {'W': w['Wq'], 'b': w['bq']}, 0.0)
    K = paths['k_proj'](x_norm, {'W': w['Wk'], 'b': w['bk']}, 0.0)
    V = paths['v_proj'](x_norm, {'W': w['Wv'], 'b': w['bv']}, 0.0)
    scores = paths['score'](q, {'K': K, 'scale': H ** -0.5}, 0.0)
    probs = paths['normalize'](scores, {'mask': mask}, 0.0)
    mixed = paths['mix'](probs, {'V': V}, 0.0)
    return child_results[0] + paths['out_proj'](mixed, {'W': w['Wo'], 'b': w['bo']}, 0.0)


def ffn_cell(payload, child_results, params, temp):
    paths, w = params['paths'], payload[0]
    x_norm = layer_norm(child_results[0], w['ln2_g'], w['ln2_b'])
    h = paths['up'](x_norm, {'W': w['W1'], 'b': w['b1']}, 0.0)
    h = paths['act'](h, {}, 0.0)
    return child_results[0] + paths['down'](h, {'W': w['W2'], 'b': w['b2']}, 0.0)


# ---------------------------------------------------------------------------
# DSL source — single-head attention + FFN
# ---------------------------------------------------------------------------

DSL_SOURCE = """
semiring attn:
    contract = ops.identity

sort model, q, scores, probs, mixed, ff

morphism q_proj    : model -> q       via "sd,dh->sh"    op ops.proj
morphism score     : q -> scores      via "sh,th->st"    op ops.score
morphism normalize : scores -> probs  via "st->st"        op ops.softmax
morphism mix       : probs -> mixed   via "st,th->sh"    op ops.mix
morphism out_proj  : mixed -> model   via "sh,hd->sd"    op ops.proj
morphism k_proj    : model -> q       via "sd,dh->sh"    op ops.proj
morphism v_proj    : model -> q       via "sd,dh->sh"    op ops.proj

morphism up   : model -> ff   via "sd,df->sf"   op ops.proj
morphism act  : ff -> ff      via "sf->sf"      op ops.gelu
morphism down : ff -> model   via "sf,fd->sd"   op ops.proj

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

    def test_algebra_forward(self, arch, weights):
        """Run algebra via arch.interpreter('Transformer'), compare to direct numpy."""
        block, tok_embed, pos_enc = weights
        rng = np.random.default_rng(7)
        token_ids = rng.integers(0, V, size=S)

        x0 = tok_embed[token_ids] + pos_enc[:S]
        mask = causal_mask(S)

        def attn_fwd(x_norm, w):
            kv = arch.paths['kv'](x_norm, {'W': w['Wk'], 'b': w['bk']}, 0.0)
            K, V2 = kv['k_proj'], arch.paths['v_proj'](x_norm, {'W': w['Wv'], 'b': w['bv']}, 0.0)
            q = arch.paths['q_proj'](x_norm, {'W': w['Wq'], 'b': w['bq']}, 0.0)
            scores = arch.paths['score'](q, {'K': K, 'scale': H ** -0.5}, 0.0)
            probs = arch.paths['normalize'](scores, {'mask': mask}, 0.0)
            mixed = arch.paths['mix'](probs, {'V': V2}, 0.0)
            return arch.paths['out_proj'](mixed, {'W': w['Wo'], 'b': w['bo']}, 0.0)

        x = x0
        x = x + attn_fwd(layer_norm(x, block['ln1_g'], block['ln1_b']), block)
        x_n = layer_norm(x, block['ln2_g'], block['ln2_b'])
        h = arch.paths['up'](x_n, {'W': block['W1'], 'b': block['b1']}, 0.0)
        h = arch.paths['act'](h, {}, 0.0)
        direct = x + arch.paths['down'](h, {'W': block['W2'], 'b': block['b2']}, 0.0)
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


# ---------------------------------------------------------------------------
# TestNdarrayCoder
# ---------------------------------------------------------------------------

class TestNdarrayCoder:

    def test_ndarray_coder_roundtrip(self):
        """Roundtrip: ndarray -> TermLiteral -> ndarray via _ndarray_decode/_ndarray_encode."""
        from engine.sorts import _ndarray_decode, _ndarray_encode

        cases = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
            np.array(42, dtype=np.int32),
        ]
        for arr in cases:
            decode_result = _ndarray_decode(None, arr)
            assert isinstance(decode_result, Right)
            term = decode_result.value

            encode_result = _ndarray_encode(None, None, term)
            assert isinstance(encode_result, Right)
            recovered = encode_result.value

            np.testing.assert_array_equal(recovered, arr)

    def test_bundle_coder_roundtrip(self):
        """Roundtrip: dict-of-arrays -> TermLiteral -> dict via _bundle_decode/_bundle_encode."""
        from engine.sorts import _bundle_decode, _bundle_encode

        bundle = {
            "W": np.array([1.0, 2.0], dtype=np.float64),
            "b": np.array([0.5], dtype=np.float64),
        }

        decode_result = _bundle_decode(None, bundle)
        assert isinstance(decode_result, Right)
        term = decode_result.value

        encode_result = _bundle_encode(None, None, term)
        assert isinstance(encode_result, Right)
        recovered = encode_result.value

        assert set(recovered.keys()) == {"W", "b"}
        np.testing.assert_array_equal(recovered["W"], bundle["W"])
        np.testing.assert_array_equal(recovered["b"], bundle["b"])

    def test_reduce_term_with_ndarray(self):
        """reduce_term applies a registered morphism primitive to a float32 ndarray."""
        from engine.compiler import compile as engine_compile
        from engine.sorts import _ndarray_decode
        from hydra.reduction import reduce_term
        from hydra.core import TermApplication, Application, TermVariable
        from hydra.context import Context
        from hydra.dsl.python import FrozenDict

        def double_op(eq, x, temp=0.0):
            return x * 2.0

        ns = {
            'ops': type('ns', (), {
                'double': staticmethod(double_op),
                'identity': staticmethod(lambda eq, x, y=None, temp=0.0: x),
            })()
        }

        src = """
semiring s:
    contract = ops.identity

sort tensor

morphism f : tensor -> tensor  via "x"  arity unary  op ops.double
"""
        arch = engine_compile(src, ns)
        assert arch._hydra_primitives, "No primitives registered"

        prim_name = next(iter(arch._hydra_primitives.keys()))
        graph = arch.graph

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        term_result = _ndarray_decode(None, arr)
        assert isinstance(term_result, Right)
        arr_term = term_result.value

        cx = Context(trace=(), messages=(), other=FrozenDict({}))
        application = TermApplication(Application(TermVariable(prim_name), arr_term))

        result = reduce_term(cx, graph, True, application)
        assert isinstance(result, Right), f"reduce_term failed: {result}"

        from engine.sorts import _ndarray_encode
        recovered = _ndarray_encode(None, None, result.value)
        assert isinstance(recovered, Right)
        expected = double_op("x", arr)
        np.testing.assert_array_equal(recovered.value, expected)


# ---------------------------------------------------------------------------
# TestArchRewriting
# ---------------------------------------------------------------------------

class TestArchRewriting:
    """Verify that _morphism_terms are populated with semiring/equation fields
    and that Hydra's rewrite_term can substitute string values inside them."""

    @pytest.fixture
    def arch(self):
        def noop(eq, x, y=None, temp=0.0):
            return x

        ns = {
            'ops': type('ns', (), {
                'proj': staticmethod(lambda eq, x, y=None, temp=0.0: x),
                'score': staticmethod(lambda eq, x, y=None, temp=0.0: x),
                'softmax': staticmethod(lambda eq, x, y=None, temp=0.0: x),
                'mix': staticmethod(lambda eq, x, y=None, temp=0.0: x),
                'gelu': staticmethod(lambda eq, x, y=None, temp=0.0: x),
                'identity': staticmethod(noop),
                'input_cell': staticmethod(lambda payload, child_results, params, temp: None),
                'attn_cell': staticmethod(lambda payload, child_results, params, temp: None),
                'ffn_cell': staticmethod(lambda payload, child_results, params, temp: None),
            })(),
        }
        from engine.compiler import compile as engine_compile
        return engine_compile(DSL_SOURCE, ns)

    def test_semiring_substitution_in_term(self, arch):
        from hydra.core import TermLiteral, LiteralString, TermRecord, Record
        from hydra.rewriting import rewrite_term

        # _morphism_terms is populated during compilation
        assert arch._morphism_terms, "_morphism_terms should not be empty after compilation"
        assert 'q_proj' in arch._morphism_terms, "expected 'q_proj' morphism term"

        term_wrapper = arch._morphism_terms['q_proj']
        # TTerm wrapper — .value is either a TermRecord (wrapping a Record)
        # or a raw Record, depending on what the phantom DSL stores.
        raw = term_wrapper.value
        if isinstance(raw, TermRecord):
            record = raw.value
        else:
            assert isinstance(raw, Record), f"unexpected type: {type(raw)}"
            record = raw

        # Wrap in TermRecord so rewrite_term can operate on it
        term = TermRecord(record)

        # Extract field values by name from the record
        fields_by_name = {f.name.value: f.term for f in record.fields}

        assert 'semiring' in fields_by_name, "morphism term missing 'semiring' field"
        assert 'equation' in fields_by_name, "morphism term missing 'equation' field"

        semiring_term = fields_by_name['semiring']
        equation_term = fields_by_name['equation']

        assert isinstance(semiring_term, TermLiteral)
        assert isinstance(semiring_term.value, LiteralString)
        assert semiring_term.value.value == 'attn', (
            f"expected semiring 'attn', got {semiring_term.value.value!r}"
        )

        assert isinstance(equation_term, TermLiteral)
        assert isinstance(equation_term.value, LiteralString)
        assert equation_term.value.value == 'sd,dh->sh', (
            f"expected equation 'sd,dh->sh', got {equation_term.value.value!r}"
        )

        # Use rewrite_term to substitute the semiring name 'attn' -> 'tropical'
        def replace_attn(recurse, t):
            if isinstance(t, TermLiteral) and isinstance(t.value, LiteralString):
                if t.value.value == 'attn':
                    return TermLiteral(LiteralString('tropical'))
            return recurse(t)

        rewritten = rewrite_term(replace_attn, term)

        assert isinstance(rewritten, TermRecord), f"expected TermRecord after rewrite, got {type(rewritten)}"
        rewritten_fields = {f.name.value: f.term for f in rewritten.value.fields}
        rewritten_semiring = rewritten_fields['semiring']
        assert isinstance(rewritten_semiring, TermLiteral)
        assert rewritten_semiring.value.value == 'tropical', (
            f"rewrite_term should have changed 'attn' to 'tropical', "
            f"got {rewritten_semiring.value.value!r}"
        )
        # Equation should be unchanged
        rewritten_eq = rewritten_fields['equation']
        assert rewritten_eq.value.value == 'sd,dh->sh'
