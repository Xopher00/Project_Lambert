"""End-to-end Hydra integration tests for the engine DSL compiler.

Verifies the full Hydra integration path: compiling a DSL spec and checking
that the resulting ArchDef exposes correct Module, Graph, term representations,
union types, and sort type checking.
"""

import types as pytypes

import numpy as np
import pytest
import engine

from engine.sorts import sort_to_type
from engine.decl import SortDecl


# ---------------------------------------------------------------------------
# Shared DSL spec and fixture
# ---------------------------------------------------------------------------

_DSL = """
semiring attn:
    contract = ops.join_fn
    compiler = ops.compile_einsum

sort model, q, scores, mixed

morphism q_proj : model -> q via "sd,dh->sh"
morphism score  : q -> scores via "sh,sh->s"
morphism mix    : scores -> mixed via "s,sh->sh"
morphism out    : mixed -> model via "sh,hd->sd"

path read = q_proj score mix out
path attn = q_proj score mix out  residual

fan kv = q_proj & score  merge dict

arch Transformer:
    algebra:
        case input: recursive=0  data=1  cell=identity
        case block: recursive=1  data=1  morphisms = read
"""


def _make_namespace():
    ops = pytypes.SimpleNamespace(
        join_fn=lambda eq, x, y, temp=0.0: x,
        compile_einsum=lambda eq: eq,
    )
    return {"ops": ops}


@pytest.fixture(scope="module")
def arch():
    return engine.compile(_DSL, _make_namespace())


# ---------------------------------------------------------------------------
# TestModuleAssembly
# ---------------------------------------------------------------------------

class TestModuleAssembly:

    def test_module_has_namespace(self, arch):
        mod = arch.module
        assert mod.namespace.value == "ua.engine.compiled"

    def test_module_has_all_definitions(self, arch):
        mod = arch.module
        names = {d.value.name.value for d in mod.definitions}
        assert "ua.engine.compiled.q_proj" in names
        assert "ua.engine.compiled.score" in names
        assert "ua.engine.compiled.read" in names
        assert "ua.engine.compiled.attn" in names
        assert "ua.engine.compiled.kv" in names
        assert "ua.engine.compiled.Transformer" in names

    def test_module_definition_count(self, arch):
        # 4 morphisms + 2 paths + 1 fan + 1 arch = 8
        mod = arch.module
        assert len(mod.definitions) == 8

    def test_module_type_dependencies(self, arch):
        mod = arch.module
        dep_ns = [d.value for d in mod.type_dependencies]
        assert "ua.engine" in dep_ns

    def test_module_has_description(self, arch):
        from hydra.dsl.python import Just
        mod = arch.module
        assert isinstance(mod.description, Just)


# ---------------------------------------------------------------------------
# TestGraphConstruction
# ---------------------------------------------------------------------------

class TestGraphConstruction:

    def test_graph_has_bound_terms(self, arch):
        g = arch.graph
        assert len(g.bound_terms) > 0

    def test_graph_bound_terms_match_module_defs(self, arch):
        mod = arch.module
        g = arch.graph
        mod_names = {d.value.name for d in mod.definitions}
        graph_names = set(g.bound_terms.keys())
        assert mod_names == graph_names

    def test_graph_has_primitives(self, arch):
        g = arch.graph
        assert len(g.primitives) > 0

    def test_graph_primitive_names(self, arch):
        from hydra.core import Name
        g = arch.graph
        assert Name("ua.lib.tensor.attn") in g.primitives

    def test_graph_bound_term_for_morphism(self, arch):
        from hydra.core import Name
        g = arch.graph
        assert Name("ua.engine.compiled.q_proj") in g.bound_terms

    def test_graph_bound_term_for_fan(self, arch):
        from hydra.core import Name
        g = arch.graph
        assert Name("ua.engine.compiled.kv") in g.bound_terms

    def test_graph_bound_term_for_arch(self, arch):
        from hydra.core import Name
        g = arch.graph
        assert Name("ua.engine.compiled.Transformer") in g.bound_terms


# ---------------------------------------------------------------------------
# TestMorphismTerms
# ---------------------------------------------------------------------------

class TestMorphismTerms:

    def test_all_morphisms_have_terms(self, arch):
        for name in arch._morphism_specs:
            assert name in arch._morphism_terms, f"Missing term for morphism {name}"

    def test_morphism_term_is_record(self, arch):
        from hydra.core import TermRecord
        for term in arch._morphism_terms.values():
            assert isinstance(term, TermRecord)

    def test_morphism_term_has_name_field(self, arch):
        term = arch._morphism_terms["q_proj"]
        field_names = [f.name.value for f in term.value.fields]
        assert "name" in field_names

    def test_morphism_term_has_equation_field(self, arch):
        term = arch._morphism_terms["q_proj"]
        field_names = [f.name.value for f in term.value.fields]
        assert "equation" in field_names

    def test_morphism_term_has_sort_fields(self, arch):
        term = arch._morphism_terms["q_proj"]
        field_names = [f.name.value for f in term.value.fields]
        assert "srcSort" in field_names
        assert "tgtSort" in field_names

    def test_morphism_term_has_arity_field(self, arch):
        term = arch._morphism_terms["q_proj"]
        field_names = [f.name.value for f in term.value.fields]
        assert "arity" in field_names

    def test_morphism_term_name_value(self, arch):
        from hydra.core import TermLiteral
        term = arch._morphism_terms["score"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert isinstance(field_map["name"], TermLiteral)
        assert field_map["name"].value.value == "score"


# ---------------------------------------------------------------------------
# TestPathTerms
# ---------------------------------------------------------------------------

class TestPathTerms:

    def test_paths_have_terms(self, arch):
        assert "read" in arch._path_terms
        assert "attn" in arch._path_terms

    def test_path_term_has_morphisms_list(self, arch):
        from hydra.core import TermList
        term = arch._path_terms["read"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "morphisms" in field_map
        assert isinstance(field_map["morphisms"], TermList)

    def test_read_path_morphisms_count(self, arch):
        term = arch._path_terms["read"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        # read = q_proj score mix out (4 morphisms)
        assert len(field_map["morphisms"].value) == 4

    def test_path_term_has_residual_field(self, arch):
        term = arch._path_terms["read"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "residual" in field_map

    def test_read_path_residual_false(self, arch):
        from hydra.core import TermLiteral
        term = arch._path_terms["read"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert isinstance(field_map["residual"], TermLiteral)
        assert field_map["residual"].value.value is False

    def test_residual_path_has_residual_true(self, arch):
        from hydra.core import TermLiteral
        term = arch._path_terms["attn"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "residual" in field_map
        assert isinstance(field_map["residual"], TermLiteral)
        assert field_map["residual"].value.value is True

    def test_path_term_has_name_field(self, arch):
        term = arch._path_terms["read"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "name" in field_map


# ---------------------------------------------------------------------------
# TestFanTerms
# ---------------------------------------------------------------------------

class TestFanTerms:

    def test_fan_has_term(self, arch):
        assert "kv" in arch._fan_terms

    def test_fan_term_has_branches(self, arch):
        from hydra.core import TermList
        term = arch._fan_terms["kv"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "branches" in field_map
        assert isinstance(field_map["branches"], TermList)

    def test_fan_branch_count(self, arch):
        term = arch._fan_terms["kv"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        # kv = q_proj & score (2 branches)
        assert len(field_map["branches"].value) == 2

    def test_fan_term_has_merge_field(self, arch):
        term = arch._fan_terms["kv"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "merge" in field_map

    def test_fan_merge_is_dict(self, arch):
        from hydra.core import TermLiteral
        term = arch._fan_terms["kv"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert isinstance(field_map["merge"], TermLiteral)
        assert field_map["merge"].value.value == "dict"

    def test_fan_term_has_name_field(self, arch):
        term = arch._fan_terms["kv"]
        field_map = {f.name.value: f.term for f in term.value.fields}
        assert "name" in field_map


# ---------------------------------------------------------------------------
# TestArchTypes
# ---------------------------------------------------------------------------

class TestArchTypes:

    def test_arch_has_union_type(self, arch):
        from hydra.core import TypeUnion
        assert "Transformer.cases" in arch._arch_types
        assert isinstance(arch._arch_types["Transformer.cases"], TypeUnion)

    def test_union_type_has_cases(self, arch):
        union = arch._arch_types["Transformer.cases"]
        case_names = [f.name.value for f in union.value]
        assert "input" in case_names
        assert "block" in case_names

    def test_union_type_case_count(self, arch):
        union = arch._arch_types["Transformer.cases"]
        assert len(union.value) == 2

    def test_arch_has_term(self, arch):
        from hydra.core import TermRecord
        assert "Transformer" in arch._arch_terms
        assert isinstance(arch._arch_terms["Transformer"], TermRecord)

    def test_arch_term_has_algebra_cases_field(self, arch):
        term = arch._arch_terms["Transformer"]
        field_names = [f.name.value for f in term.value.fields]
        assert "algebraCases" in field_names

    def test_arch_term_has_name_field(self, arch):
        term = arch._arch_terms["Transformer"]
        field_names = [f.name.value for f in term.value.fields]
        assert "name" in field_names

    def test_union_input_case_has_data_field(self, arch):
        from hydra.core import TypeRecord
        union = arch._arch_types["Transformer.cases"]
        input_field = next(f for f in union.value if f.name.value == "input")
        assert isinstance(input_field.type, TypeRecord)
        inner_field_names = [ff.name.value for ff in input_field.type.value]
        assert "data" in inner_field_names

    def test_union_block_case_has_children_field(self, arch):
        # recursive=1 case should have a 'children' field
        from hydra.core import TypeRecord
        union = arch._arch_types["Transformer.cases"]
        block_field = next(f for f in union.value if f.name.value == "block")
        assert isinstance(block_field.type, TypeRecord)
        inner_field_names = [ff.name.value for ff in block_field.type.value]
        assert "children" in inner_field_names


# ---------------------------------------------------------------------------
# TestSortTypeChecking
# ---------------------------------------------------------------------------

class TestSortTypeChecking:

    def test_valid_path_compiles(self, arch):
        # 'read' path compiled successfully — sorts matched end-to-end
        assert "read" in arch.paths

    def test_sort_mismatch_raises(self):
        bad_dsl = """
semiring s:
    contract = ops.fn

sort a, b, c

morphism f : a -> b via "x"
morphism g : c -> a via "x"
path bad = f g
"""
        ops = pytypes.SimpleNamespace(fn=lambda eq, x, y, temp=0.0: x)
        with pytest.raises(TypeError, match="Type mismatch"):
            engine.compile(bad_dsl, {"ops": ops})

    def test_sort_mismatch_error_mentions_morphisms(self):
        bad_dsl = """
semiring s:
    contract = ops.fn

sort a, b, c

morphism alpha : a -> b via "x"
morphism beta  : c -> a via "x"
path bad = alpha beta
"""
        ops = pytypes.SimpleNamespace(fn=lambda eq, x, y, temp=0.0: x)
        with pytest.raises(TypeError, match="alpha"):
            engine.compile(bad_dsl, {"ops": ops})

    def test_opaque_sort_type_is_variable(self):
        from hydra.core import TypeVariable
        t = sort_to_type("model")
        assert isinstance(t, TypeVariable)
        assert t.value.value == "ua.sort.model"

    def test_opaque_sorts_different_names_not_equal(self):
        t1 = sort_to_type("q")
        t2 = sort_to_type("model")
        assert t1 != t2

    def test_structured_sort_self_equality(self):
        defs = {"cache": SortDecl("cache", fields={"K": "tensor", "V": "tensor"})}
        t1 = sort_to_type("cache", defs)
        t2 = sort_to_type("cache", defs)
        assert t1 == t2

    def test_structured_sort_is_record_type(self):
        from hydra.core import TypeRecord
        defs = {"cache": SortDecl("cache", fields={"K": "tensor", "V": "tensor"})}
        t = sort_to_type("cache", defs)
        assert isinstance(t, TypeRecord)

    def test_structured_sort_field_names(self):
        from hydra.core import TypeRecord
        defs = {"cache": SortDecl("cache", fields={"K": "tensor", "V": "tensor"})}
        t = sort_to_type("cache", defs)
        assert isinstance(t, TypeRecord)
        field_names = [f.name.value for f in t.value]
        assert "K" in field_names
        assert "V" in field_names

    def test_opaque_sort_ignores_defs_for_fields_none(self):
        from hydra.core import TypeVariable
        defs = {"model": SortDecl("model", fields=None)}
        t = sort_to_type("model", defs)
        assert isinstance(t, TypeVariable)

    def test_chained_sort_mismatch_at_second_step(self):
        """Sort mismatch at step 2 in a 3-morphism chain raises TypeError."""
        dsl = """
semiring s:
    contract = ops.fn

sort a, b, c, d

morphism f : a -> b via "x"
morphism g : b -> c via "x"
morphism h : d -> b via "x"

path bad = f g h
"""
        ops = pytypes.SimpleNamespace(fn=lambda eq, x, y, temp=0.0: x)
        with pytest.raises(TypeError):
            engine.compile(dsl, {"ops": ops})


# ---------------------------------------------------------------------------
# TestRuntimeUnchanged
# ---------------------------------------------------------------------------

class TestRuntimeUnchanged:

    def test_paths_still_callable(self, arch):
        x = np.ones((2, 3))
        y = {"W": np.ones((3, 4))}
        result = arch.paths["q_proj"](x, y, 0.0)
        # join_fn returns x unchanged — shape stays (2, 3)
        assert result.shape == (2, 3)

    def test_interpreter_still_works(self, arch):
        x0 = np.ones((2, 3))
        interp = arch.interpreter("Transformer", params={"W": np.eye(3)}, temp=0.0)
        tree = ("input", [x0], [])
        result = interp.run_algebra(tree, lambda n: n)
        assert result is not None
        assert result.shape == (2, 3)

    def test_fan_still_returns_dict(self, arch):
        x = np.ones((2, 3))
        result = arch.paths["kv"](x, None, 0.0)
        assert isinstance(result, dict)
        assert "q_proj" in result
        assert "score" in result

    def test_all_named_paths_callable(self, arch):
        for name, fn in arch.paths.items():
            assert callable(fn), f"Path {name!r} is not callable"

    def test_explain_still_works(self, arch):
        text = arch.explain("read")
        assert "q_proj" in text

    def test_morphism_semiring_populated(self, arch):
        assert "q_proj" in arch.morphism_semiring
        assert arch.morphism_semiring["q_proj"] == "attn"
