"""Tests for engine/functor.py — Case, Functor, UnfoldStep."""

import pytest
from engine.functor import Case, Functor, UnfoldStep, NO_OUTPUT


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

class TestCase:

    def test_basic_construction(self):
        c = Case('step', recursive=1, data=2)
        assert c.name == 'step'
        assert c.recursive == 1
        assert c.data == 2
        assert c.output == 0

    def test_output_flag(self):
        c = Case('emit', recursive=1, data=0, output=1)
        assert c.output == 1

    def test_negative_recursive_raises(self):
        with pytest.raises(ValueError, match="recursive must be >= 0"):
            Case('bad', recursive=-1, data=0)

    def test_negative_data_raises(self):
        with pytest.raises(ValueError, match="data must be >= 0"):
            Case('bad', recursive=0, data=-1)

    def test_invalid_output_raises(self):
        with pytest.raises(ValueError, match="output must be 0 or 1"):
            Case('bad', recursive=0, data=0, output=2)

    def test_zero_recursive_zero_data(self):
        c = Case('leaf', recursive=0, data=0)
        assert c.recursive == 0
        assert c.data == 0


# ---------------------------------------------------------------------------
# TestFunctor
# ---------------------------------------------------------------------------

class TestFunctor:

    def test_basic_construction(self):
        f = Functor([Case('a', 0, 1), Case('b', 1, 0)])
        assert len(f.cases) == 2

    def test_getitem(self):
        f = Functor([Case('step', 1, 2), Case('done', 0, 1)])
        assert f['step'].name == 'step'
        assert f['done'].recursive == 0

    def test_unknown_case_raises(self):
        f = Functor([Case('only', 0, 0)])
        with pytest.raises(KeyError, match="Unknown case"):
            f['nonexistent']

    def test_duplicate_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate case names"):
            Functor([Case('x', 0, 0), Case('x', 1, 1)])

    def test_single_case(self):
        f = Functor([Case('sole', recursive=0, data=0)])
        assert f['sole'].name == 'sole'


# ---------------------------------------------------------------------------
# TestUnfoldStep
# ---------------------------------------------------------------------------

class TestUnfoldStep:

    def test_construction(self):
        r = UnfoldStep(case_name='step', payload=[1, 2], next_states=['s'])
        assert r.case_name == 'step'
        assert r.payload == [1, 2]
        assert r.next_states == ['s']
        assert r.output is NO_OUTPUT

    def test_with_output(self):
        r = UnfoldStep(case_name='emit', payload=[], next_states=['s'], output=42)
        assert r.output == 42

    def test_no_output_sentinel(self):
        r = UnfoldStep(case_name='x', payload=[], next_states=[])
        assert r.output is NO_OUTPUT
        assert r.output is not None
