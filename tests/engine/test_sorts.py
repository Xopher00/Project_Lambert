"""Tests for engine.sorts and engine.primitives."""

import pytest

from engine.sorts import (
    bundle_coder,
    equation_coder,
    ndarray_coder,
    sort_to_type,
    sort_types_from_defs,
    temp_coder,
)
from engine.decl import SortDecl
from engine.primitives import build_engine_graph, qname

from hydra.core import (
    Name,
    TypeLiteral,
    TypeVariable,
)
from hydra.dsl.python import Right


# ---------------------------------------------------------------------------
# sort_types_from_defs (sort_to_type is tested in test_hydra_integration.py)
# ---------------------------------------------------------------------------

class TestSortTypesFromDefs:
    def test_combines_declared_and_morphism_sorts(self):
        defs = {"i": SortDecl("i"), "j": SortDecl("j")}
        morph_sorts = {"i", "j", "k"}
        result = sort_types_from_defs(defs, morph_sorts)
        assert set(result.keys()) == {"i", "j", "k"}
        for v in result.values():
            assert isinstance(v, TypeVariable)

    def test_empty(self):
        result = sort_types_from_defs({}, set())
        assert result == {}


# ---------------------------------------------------------------------------
# TermCoders
# ---------------------------------------------------------------------------

class TestTermCoders:
    def test_ndarray_coder_type(self):
        c = ndarray_coder()
        assert isinstance(c.type, TypeVariable)
        assert c.type.value == Name("ua.tensor.NDArray")

    def test_ndarray_coder_roundtrip(self):
        c = ndarray_coder()
        sentinel = object()
        result = c.decode(None, sentinel)
        assert isinstance(result, Right)
        assert result.value is sentinel

    def test_bundle_coder_type(self):
        c = bundle_coder()
        assert isinstance(c.type, TypeVariable)
        assert c.type.value == Name("ua.tensor.Bundle")

    def test_temp_coder_type(self):
        c = temp_coder()
        assert isinstance(c.type, TypeLiteral)

    def test_equation_coder_type(self):
        c = equation_coder()
        assert isinstance(c.type, TypeLiteral)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestHydraPrimitives:
    def test_qname(self):
        n = qname("ua.lib.tensor", "join")
        assert n == Name("ua.lib.tensor.join")

    def test_build_empty_graph(self):
        g = build_engine_graph({})
        assert len(g.primitives) == 0

    def test_build_graph_with_primitives(self):
        from hydra.dsl import prims
        nd = ndarray_coder()
        p = prims.prim1(Name("test.id"), lambda x: x, [], nd, nd)
        g = build_engine_graph({Name("test.id"): p})
        assert len(g.primitives) == 1
        assert Name("test.id") in g.primitives
