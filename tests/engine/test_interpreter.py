"""Tests for engine/interpreter.py — Interpreter algebra/coalgebra."""

import pytest
from engine.functor import Functor, Case, UnfoldStep, Interpreter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Simple tree functor: leaf (no children, 1 data) + node (2 children, 0 data)
TreeF = Functor([
    Case('leaf', recursive=0, data=1),
    Case('node', recursive=2, data=0),
])

# Linear functor: step (1 child, 1 data) + base (0 children, 1 data)
LinearF = Functor([
    Case('base', recursive=0, data=1),
    Case('step', recursive=1, data=1),
])

# Coalgebra functor: silent (no output) + emit (has output)
StepF = Functor([
    Case('silent', recursive=1, data=0, output=0),
    Case('emit',   recursive=1, data=0, output=1),
])


# ---------------------------------------------------------------------------
# TestAlgebra
# ---------------------------------------------------------------------------

class TestAlgebra:

    def test_leaf_only(self):
        """Algebra on a single leaf node."""
        def cell(case_name, payload, child_results, params):
            if case_name == 'leaf':
                return payload[0]
            raise ValueError(case_name)

        interp = Interpreter(TreeF, cell, params={})
        result = interp.run_algebra(('leaf', [42], []), lambda n: n)
        assert result == 42

    def test_tree_sum(self):
        """Sum all leaf values in a binary tree."""
        def cell(case_name, payload, child_results, params):
            if case_name == 'leaf':
                return payload[0]
            if case_name == 'node':
                return sum(child_results)
            raise ValueError(case_name)

        tree = ('node', [], [
            ('leaf', [10], []),
            ('node', [], [
                ('leaf', [20], []),
                ('leaf', [30], []),
            ]),
        ])

        interp = Interpreter(TreeF, cell, params={})
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 60

    def test_linear_fold(self):
        """Fold a linear chain: base(1) -> step(2) -> step(3) = 1+2+3 = 6."""
        def cell(case_name, payload, child_results, params):
            if case_name == 'base':
                return payload[0]
            if case_name == 'step':
                return payload[0] + child_results[0]
            raise ValueError(case_name)

        chain = ('step', [3], [
            ('step', [2], [
                ('base', [1], []),
            ]),
        ])

        interp = Interpreter(LinearF, cell, params={})
        result = interp.run_algebra(chain, lambda n: n)
        assert result == 6

    def test_params_passed(self):
        """Verify params dict reaches the cell."""
        def cell(case_name, payload, child_results, params):
            return payload[0] * params['scale']

        interp = Interpreter(
            Functor([Case('leaf', 0, 1)]),
            cell, params={'scale': 10}
        )
        result = interp.run_algebra(('leaf', [5], []), lambda n: n)
        assert result == 50

    def test_temp_passed(self):
        """Verify temp reaches the cell via params['_temp']."""
        def cell(case_name, payload, child_results, params):
            return params.get('_temp', 0.0)

        interp = Interpreter(
            Functor([Case('leaf', 0, 0)]),
            cell, params={}, temp=0.5
        )
        result = interp.run_algebra(('leaf', [], []), lambda n: n)
        assert result == 0.5

    def test_wrong_payload_count_raises(self):
        def cell(case_name, payload, child_results, params):
            return 0

        interp = Interpreter(
            Functor([Case('leaf', recursive=0, data=2)]),
            cell, params={}
        )
        with pytest.raises(ValueError, match="declared data=2, got 1"):
            interp.run_algebra(('leaf', [1], []), lambda n: n)

    def test_wrong_children_count_raises(self):
        def cell(case_name, payload, child_results, params):
            return 0

        interp = Interpreter(
            Functor([Case('node', recursive=2, data=0)]),
            cell, params={}
        )
        with pytest.raises(ValueError, match="declared recursive=2, got 1"):
            interp.run_algebra(('node', [], [('leaf', [1], [])]), lambda n: n)


# ---------------------------------------------------------------------------
# TestCoalgebra
# ---------------------------------------------------------------------------

class TestCoalgebra:

    def test_token_exhaustion(self):
        """Coalgebra stops when tokens run out."""
        counter = [0]

        def cell(state, token, params):
            counter[0] += 1
            return UnfoldStep(
                case_name='silent',
                payload=[],
                next_states=[state + 1],
            )

        interp = Interpreter(StepF, cell, params={})
        outputs, final = interp.run_coalgebra(state=0, token_iter=[10, 20, 30])
        assert outputs == []
        assert final == 3
        assert counter[0] == 3

    def test_emit_case(self):
        """Coalgebra emits output when case has output=1."""
        def cell(state, token, params):
            return UnfoldStep(
                case_name='emit',
                payload=[],
                next_states=[state + 1],
                output=token * 2,
            )

        interp = Interpreter(StepF, cell, params={})
        outputs, final = interp.run_coalgebra(state=0, token_iter=[1, 2, 3])
        assert outputs == [2, 4, 6]
        assert final == 3

    def test_case_variation(self):
        """Mix of silent and emit cases."""
        def cell(state, token, params):
            if token == 'emit':
                return UnfoldStep(
                    case_name='emit',
                    payload=[],
                    next_states=[state + 1],
                    output=state,
                )
            return UnfoldStep(
                case_name='silent',
                payload=[],
                next_states=[state + 1],
            )

        interp = Interpreter(StepF, cell, params={})
        outputs, final = interp.run_coalgebra(
            state=0, token_iter=['skip', 'skip', 'emit', 'skip', 'emit']
        )
        assert outputs == [2, 4]
        assert final == 5

    def test_stop_callback(self):
        """Coalgebra stops when stop() returns True."""
        def cell(state, token, params):
            return UnfoldStep(
                case_name='silent',
                payload=[],
                next_states=[state + 1],
            )

        interp = Interpreter(StepF, cell, params={})
        outputs, final = interp.run_coalgebra(
            state=0,
            token_iter=range(100),
            stop=lambda step, state, outputs: step >= 5,
        )
        assert final == 5

    def test_non_linear_case_raises(self):
        """Coalgebra rejects cases with recursive != 1."""
        BranchF = Functor([
            Case('branch', recursive=2, data=0),
        ])

        def cell(state, token, params):
            return UnfoldStep(
                case_name='branch',
                payload=[],
                next_states=[state, state],
            )

        interp = Interpreter(BranchF, cell, params={})
        with pytest.raises(ValueError, match="single-successor"):
            interp.run_coalgebra(state=0, token_iter=[1])

    def test_payload_mismatch_raises(self):
        """Coalgebra validates payload count."""
        PayloadF = Functor([
            Case('step', recursive=1, data=2),
        ])

        def cell(state, token, params):
            return UnfoldStep(
                case_name='step',
                payload=[1],  # declared data=2, only 1 given
                next_states=[state],
            )

        interp = Interpreter(PayloadF, cell, params={})
        with pytest.raises(ValueError, match="declared data=2"):
            interp.run_coalgebra(state=0, token_iter=[1])

    def test_output_mismatch_raises(self):
        """Coalgebra validates output presence matches case declaration."""
        def cell(state, token, params):
            # silent case (output=0) but we provide an output
            return UnfoldStep(
                case_name='silent',
                payload=[],
                next_states=[state + 1],
                output='unexpected',
            )

        interp = Interpreter(StepF, cell, params={})
        with pytest.raises(ValueError, match="output presence does not match"):
            interp.run_coalgebra(state=0, token_iter=[1])

    def test_empty_tokens(self):
        """Coalgebra with empty token_iter returns immediately."""
        def cell(state, token, params):
            raise AssertionError("Should not be called")

        interp = Interpreter(StepF, cell, params={})
        outputs, final = interp.run_coalgebra(state=42, token_iter=[])
        assert outputs == []
        assert final == 42
