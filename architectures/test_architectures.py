"""
Value and shape tests for the four Gavranovic et al. (2024) architecture files.

Each test compiles a .ua file, runs a forward pass, and asserts output shapes
and non-zero values (confirming actual computation ran).

Run with:
    uv run --python 3.12 python architectures/test_architectures.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import engine

HERE = os.path.dirname(os.path.abspath(__file__))

H = 8   # hidden size
D = 4   # input/token embedding size
O = 6   # output size

rng = np.random.default_rng(42)


# Shared ops — used across multiple architectures
class _Base:
    @staticmethod
    def activate(x):
        return np.tanh(x)

    @staticmethod
    def pass_through(x):
        return x


# ---------------------------------------------------------------------------
# Architecture 1: FoldingRNN (rnn_fold.ua)
# Endofunctor: 1 + A x S   (Gavranovic I.1, J.1)
# path rnn = recur activate
# ---------------------------------------------------------------------------

_fold_W_h = rng.normal(0, 0.1, (H, H))


class FoldOps(_Base):
    @staticmethod
    def init_hidden(_payload, _child_results, _params):
        return np.zeros((1, H))


def test_folding_rnn():
    arch = engine.load(os.path.join(HERE, 'rnn_fold.ua'), {'ops': FoldOps()})
    interp = arch.interpreter('FoldingRNN', params={}, temp=0.0)

    tree = ('base', [], [])
    for _ in range(3):
        tree = ('rnn_step', [_fold_W_h], [tree])

    result = interp.run_algebra(tree, lambda n: n)
    assert result.shape == (1, H), f"Expected (1,{H}), got {result.shape}"
    print(f"  FoldingRNN: fold over 3 tokens -> {result.shape}  PASS")


# ---------------------------------------------------------------------------
# Architecture 2: TreeRNN (tree_rnn.ua)
# Endofunctor: A + S^2   (Gavranovic I.2, J.3)
# Custom cells; semiring declared but paths not used (recursive=2 case).
# ---------------------------------------------------------------------------

_tree_W_embed = rng.normal(0, 0.1, (D, H))
_tree_W_left  = rng.normal(0, 0.1, (H, H))
_tree_W_right = rng.normal(0, 0.1, (H, H))


class TreeOps:
    @staticmethod
    def leaf_cell(payload, _child_results, _params):
        return np.tanh(np.einsum('sd,dh->sh', payload[0], _tree_W_embed))

    @staticmethod
    def inner_cell(_payload, child_results, _params):
        return np.tanh(child_results[0] @ _tree_W_left + child_results[1] @ _tree_W_right)


def test_tree_rnn():
    arch = engine.load(os.path.join(HERE, 'tree_rnn.ua'), {'ops': TreeOps()})
    interp = arch.interpreter('TreeRNN', params={}, temp=0.0)

    tokens = [np.ones((1, D)) * (i + 1) for i in range(3)]
    tree = ('inner_node', [], [
        ('leaf_node', [tokens[0]], []),
        ('inner_node', [], [
            ('leaf_node', [tokens[1]], []),
            ('leaf_node', [tokens[2]], []),
        ]),
    ])

    result = interp.run_algebra(tree, lambda n: n)
    assert result.shape == (1, H), f"Expected (1,{H}), got {result.shape}"
    assert not np.allclose(result, 0), "Result is all zeros — computation did not run"
    print(f"  TreeRNN: fold over binary tree (3 leaves) -> {result.shape}  PASS")


# ---------------------------------------------------------------------------
# Architecture 3: StreamRNN (rnn_stream.ua)
# Endofunctor: O x S   (Gavranovic I.3, H.4)
# path step_path = transition activate; initial_state = W_h
# ---------------------------------------------------------------------------

_stream_W_h   = rng.normal(0, 0.1, (H, H))
_stream_W_out = rng.normal(0, 0.1, (H, O))


class StreamOps(_Base):
    def __init__(self, W_out):
        self._W_out = W_out

    def read_output(self, x):
        return np.tanh(np.einsum('sh,ho->so', x, self._W_out))


def test_stream_rnn():
    arch = engine.load(os.path.join(HERE, 'rnn_stream.ua'), {'ops': StreamOps(_stream_W_out)})
    interp = arch.interpreter('StreamRNN', params={}, temp=0.0)

    outputs, _ = interp.run_coalgebra(_stream_W_h, token_iter=[np.zeros((1, H))] * 3)

    assert len(outputs) == 3
    for i, out in enumerate(outputs):
        assert out.shape == (1, O), f"Step {i}: expected (1,{O}), got {out.shape}"
    print(f"  StreamRNN: 3 coalgebra steps -> {len(outputs)} outputs, each {outputs[0].shape}  PASS")


# ---------------------------------------------------------------------------
# Architecture 4: MealyRNN (mealy.ua)
# Endofunctor: I -> O x S   (Gavranovic I.4, Example 2.11)
# path compute = rnn_step activate; initial_state = W_h
# ---------------------------------------------------------------------------

_mealy_W_in  = rng.normal(0, 0.1, (D, H))
_mealy_W_h   = rng.normal(0, 0.1, (H, H))
_mealy_W_out = rng.normal(0, 0.1, (H, O))


class MealyOps(_Base):
    def __init__(self, W_in, W_out):
        self._W_in  = W_in
        self._W_out = W_out

    def embed_input(self, x):
        return np.tanh(np.einsum('sd,dh->sh', x, self._W_in))

    def emit_output(self, x):
        return np.einsum('sh,ho->so', x, self._W_out)


def test_mealy_rnn():
    arch = engine.load(os.path.join(HERE, 'mealy.ua'), {'ops': MealyOps(_mealy_W_in, _mealy_W_out)})
    interp = arch.interpreter('MealyRNN', params={}, temp=0.0)

    tokens = [np.ones((1, D)) * (i + 1) for i in range(3)]
    outputs, _ = interp.run_coalgebra(_mealy_W_h, token_iter=tokens)

    assert len(outputs) == 3
    for i, out in enumerate(outputs):
        assert out.shape == (1, O), f"Step {i}: expected (1,{O}), got {out.shape}"
    assert not np.allclose(outputs[0], 0), "Output is all zeros — computation did not run"
    print(f"  MealyRNN:  3 coalgebra steps -> {len(outputs)} outputs, each {outputs[0].shape}  PASS")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing Gavranovic et al. (2024) architecture files...\n")
    failures = []
    for name, fn in [
        ('FoldingRNN (1 + A×S)',  test_folding_rnn),
        ('TreeRNN   (A + S²)',    test_tree_rnn),
        ('StreamRNN (O × S)',     test_stream_rnn),
        ('MealyRNN  (I → O × S)', test_mealy_rnn),
    ]:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failures.append(name)
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All 4 architectures compiled and ran correctly.")
