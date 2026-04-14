"""
Unit tests for query.py: Query class and QueryResult dataclass.

The multihop-specific tests live in tests/test_multihop.py.
This file covers the foundational Query helpers used by both the standard
call path and the multihop path.

Import path: from query import Query, QueryResult, show
"""

import numpy as np
import pytest

from legacy.query import QueryResult, show


# ---------------------------------------------------------------------------
# QueryResult dataclass
# ---------------------------------------------------------------------------

class TestQueryResult:
    """QueryResult is a plain dataclass — verify construction and fields."""

    def test_queryresult_fields(self):
        """QueryResult stores all five fields correctly."""
        entities = ['Alice', 'Bob']
        scores = np.array([0.9, 0.7])
        intents = {'head_0': (np.zeros(3), np.zeros(2))}
        prov = [('head_0', '[head_0] feat', 0.9)]
        mode = 'forward'

        r = QueryResult(entities=entities, scores=scores, intents=intents,
                        provenance=prov, mode=mode)

        assert r.entities == entities
        np.testing.assert_array_equal(r.scores, scores)
        assert r.intents is intents
        assert r.provenance is prov
        assert r.mode == mode

    def test_queryresult_mode_multihop(self):
        """QueryResult accepts mode='multihop' (exact string contract)."""
        r = QueryResult(entities=[], scores=np.array([]), intents={},
                        provenance=[], mode='multihop')
        assert r.mode == 'multihop'


# ---------------------------------------------------------------------------
# show() function
# ---------------------------------------------------------------------------

class TestShow:
    """show() prints and returns the QueryResult unchanged."""

    def test_show_returns_result(self):
        """show() returns the same QueryResult it received."""
        r = QueryResult(
            entities=['X'],
            scores=np.array([0.8]),
            intents={},
            provenance=[],
            mode='forward',
        )
        returned = show(r, show_provenance=False)
        assert returned is r

    def test_show_empty(self):
        """show() handles an empty result without raising."""
        r = QueryResult(entities=[], scores=np.array([]), intents={},
                        provenance=[], mode='intersection')
        returned = show(r, show_provenance=False)
        assert returned is r
