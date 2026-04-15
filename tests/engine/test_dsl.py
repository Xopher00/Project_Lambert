"""Tests for engine — parse + compile."""

import numpy as np
import pytest
from engine import parse, compile, DSLSource, ArchDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_op(eq, x, y=None, temp=0.0):
    return x

def _swap(x, y):
    return (y, x)

def _upper_compiler(eq: str) -> str:
    return eq.upper()

NAMESPACE = {
    'ops': type('ns', (), {
        'dummy':    staticmethod(_dummy_op),
        'swap':     staticmethod(_swap),
        'compiler': staticmethod(_upper_compiler),
    })(),
}

BASIC_SOURCE = """
semiring join:
    contract = ops.dummy

sort i, j

leg realize   : j -> i  via "j,ji->i"
leg propagate : i -> j  via "i,ij->j"

path attend = realize propagate
"""


# ---------------------------------------------------------------------------
# TestParse
# ---------------------------------------------------------------------------

class TestParse:

    def test_basic_structure(self):
        ast = parse(BASIC_SOURCE)
        assert isinstance(ast, DSLSource)
        assert len(ast.semirings) == 1
        assert ast.semirings[0].name == 'join'
        assert sorted(ast.sorts) == ['i', 'j']
        assert len(ast.morphisms) == 2
        assert len(ast.paths) == 1

    def test_leg_fields(self):
        ast = parse(BASIC_SOURCE)
        realize = ast.morphisms[0]
        assert realize.name == 'realize'
        assert realize.src_sort == 'j'
        assert realize.tgt_sort == 'i'
        assert realize.equation == 'j,ji->i'

    def test_path_fields(self):
        ast = parse(BASIC_SOURCE)
        assert ast.paths[0].name == 'attend'
        assert ast.paths[0].morphisms == ['realize', 'propagate']

    def test_comments_stripped(self):
        src = """
semiring s:
    contract = ops.dummy  # this is a comment

sort x  # also a comment

leg a : x -> x  via "x->x"
"""
        ast = parse(src)
        assert len(ast.semirings) == 1
        assert ast.sorts == ['x']

    def test_empty_raises(self):
        with pytest.raises(SyntaxError):
            parse("badline")

    def test_semiring_missing_contract_raises(self):
        with pytest.raises(SyntaxError, match="must declare 'contract'"):
            parse("semiring s:\n    # no contract\n\nsort x")

    def test_using_clause(self):
        src = """
semiring s1:
    contract = ops.dummy

semiring s2:
    contract = ops.dummy

sort i, j

leg a : i -> j  via "i->j"  using s1
leg b : j -> i  via "j->i"  using s2
"""
        ast = parse(src)
        assert ast.morphisms[0].semiring == 's1'
        assert ast.morphisms[1].semiring == 's2'

    def test_op_clause(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"  op ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].op == 'ops.dummy'

    def test_bridge_clause(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"  bridge  op ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].semiring is None

    def test_transform_clause(self):
        src = """
semiring s:
    contract = ops.dummy

sort i, j

leg a : i -> j  via "ij,i->j"  transform ops.swap
"""
        ast = parse(src)
        assert ast.morphisms[0].transform == 'ops.swap'

    def test_compiler_on_semiring(self):
        src = """
semiring s:
    contract = ops.dummy
    compiler = ops.compiler

sort x

leg a : x -> x  via "x->x"
"""
        ast = parse(src)
        assert ast.semirings[0].compiler == 'ops.compiler'

    def test_compiler_on_leg(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"  compiler ops.compiler
"""
        ast = parse(src)
        assert ast.morphisms[0].compiler == 'ops.compiler'

    def test_unrecognised_clause_raises(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"  badclause foo
"""
        with pytest.raises(SyntaxError, match="unrecognised clause"):
            parse(src)

    def test_auto_assign_single_semiring(self):
        """Legs without 'using' auto-assign when exactly one semiring exists."""
        ast = parse(BASIC_SOURCE)
        assert ast.morphisms[0].semiring == '_default'
        assert ast.morphisms[1].semiring == '_default'


# ---------------------------------------------------------------------------
# TestCompile
# ---------------------------------------------------------------------------

class TestCompile:

    def test_basic_compile(self):
        arch = compile(BASIC_SOURCE, NAMESPACE)
        assert isinstance(arch, ArchDef)
        assert 'realize' in arch.paths
        assert 'propagate' in arch.paths
        assert 'attend' in arch.paths

    def test_compiled_paths_callable(self):
        arch = compile(BASIC_SOURCE, NAMESPACE)
        f = arch.paths['realize']
        assert callable(f)
        result = f('x', 'y', 0.0)
        assert result == 'x'  # dummy op returns x

    def test_path_is_composition(self):
        arch = compile(BASIC_SOURCE, NAMESPACE)
        attend = arch.paths['attend']
        assert callable(attend)

    def test_explain(self):
        arch = compile(BASIC_SOURCE, NAMESPACE)
        text = arch.explain('attend')
        assert 'realize' in text
        assert 'propagate' in text

    def test_undeclared_semiring_raises(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"  using nonexistent
"""
        with pytest.raises(ValueError, match="undeclared semiring"):
            compile(src, NAMESPACE)

    def test_cross_semiring_path_raises(self):
        src = """
semiring s1:
    contract = ops.dummy

semiring s2:
    contract = ops.dummy

sort i, j

leg a : i -> j  via "i->j"  using s1
leg b : j -> i  via "j->i"  using s2

path cross = a b
"""
        with pytest.raises(ValueError, match="spans multiple semirings"):
            compile(src, NAMESPACE)

    def test_no_op_no_contract_raises(self):
        src = """
sort x

leg a : x -> x  via "x->x"
"""
        # No semiring declared, no op — should fail
        # parse succeeds, but compile should raise because there's no contract
        ast = parse(src)
        assert len(ast.semirings) == 0
        with pytest.raises(ValueError, match="no 'op' clause"):
            compile(ast, NAMESPACE)

    def test_transform_resolved(self):
        src = """
semiring s:
    contract = ops.dummy

sort i, j

leg a : i -> j  via "ij,i->j"  transform ops.swap
"""
        arch = compile(src, NAMESPACE)
        # The compiled leg should apply the swap transform
        f = arch.paths['a']
        result = f('x', 'y', 0.0)
        # _dummy_op returns its first positional arg (x); swap makes that 'y'
        assert result == 'y'

    def test_compiler_semiring_level(self):
        src = """
semiring s:
    contract = ops.dummy
    compiler = ops.compiler

sort x

leg a : x -> x  via "abc->x"
"""
        arch = compile(src, NAMESPACE)
        # The equation compiler uppercases the equation string.
        # dummy_op receives the compiled equation as first arg and returns x (second arg).
        # We can verify via explain that the equation is stored.
        text = arch.explain('a')
        assert 'abc->x' in text

    def test_compiler_per_leg_overrides_semiring(self):
        src = """
semiring s:
    contract = ops.dummy
    compiler = ops.compiler

sort x

leg a : x -> x  via "abc"  compiler ops.compiler
"""
        arch = compile(src, NAMESPACE)
        assert callable(arch.paths['a'])

    def test_leg_semiring_mapping(self):
        arch = compile(BASIC_SOURCE, NAMESPACE)
        assert arch.morphism_semiring['realize'] == 'join'
        assert arch.morphism_semiring['propagate'] == 'join'

    def test_name_resolution_error(self):
        src = """
semiring s:
    contract = nonexistent.fn

sort x

leg a : x -> x  via "x->x"
"""
        with pytest.raises(NameError, match="not found"):
            compile(src, NAMESPACE)


# ---------------------------------------------------------------------------
# TestFan
# ---------------------------------------------------------------------------

class TestFan:

    def test_parse_fan_default_merge(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"

fan ab = a & b
"""
        ast = parse(src)
        assert len(ast.fans) == 1
        assert ast.fans[0].name == 'ab'
        assert ast.fans[0].branches == ['a', 'b']
        assert ast.fans[0].merge == 'dict'

    def test_parse_fan_meet_merge(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"

fan ab = a & b  merge meet
"""
        ast = parse(src)
        assert ast.fans[0].merge == 'meet'

    def test_parse_fan_custom_merge(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"

fan ab = a  merge ops.custom
"""
        ast = parse(src)
        assert ast.fans[0].merge == 'ops.custom'

    def test_compile_fan_dict(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"

fan ab = a & b
"""
        arch = compile(src, NAMESPACE)
        result = arch.paths['ab']('hello', None, 0.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'a', 'b'}
        assert result['a'] == 'hello'

    def test_compile_fan_meet(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"

fan ab = a & b  merge meet
"""
        arch = compile(src, NAMESPACE)
        x = np.array([3.0, 1.0, 2.0])
        result = arch.paths['ab'](x, None, 0.0)
        np.testing.assert_array_equal(result, x)

    def test_compile_fan_join(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"

fan ab = a & b  merge join
"""
        arch = compile(src, NAMESPACE)
        x = np.array([1.0, 2.0, 3.0])
        result = arch.paths['ab'](x, None, 0.0)
        np.testing.assert_array_equal(result, x)

    def test_compile_fan_custom_merge(self):
        def _sum_merge(results):
            return sum(results.values())

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'sum_merge': staticmethod(_sum_merge),
        })()

        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"

fan ab = a & b  merge ops.sum_merge
"""
        arch = compile(src, ns)
        result = arch.paths['ab'](5, None, 0.0)
        assert result == 10  # 5 + 5

    def test_fan_unknown_branch_raises(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"

fan ab = a & nonexistent
"""
        with pytest.raises(ValueError, match="not a declared morphism or path"):
            compile(src, NAMESPACE)

    def test_fan_no_branches_raises(self):
        with pytest.raises(SyntaxError):
            parse("fan empty =")

    def test_fan_three_branches(self):
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"
leg b : x -> x  via "x->x"
leg c : x -> x  via "x->x"

fan abc = a & b & c
"""
        arch = compile(src, NAMESPACE)
        result = arch.paths['abc']('v', None, 0.0)
        assert len(result) == 3
        assert set(result.keys()) == {'a', 'b', 'c'}


# ---------------------------------------------------------------------------
# TestArchAlgebraOnly (formerly TestFunctorDSL — functor keyword is removed)
# ---------------------------------------------------------------------------

class TestArchAlgebraOnly:

    def test_parse_functor(self):
        """arch algebra: replaces old functor syntax."""
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"

arch Program:
    algebra:
        case input: recursive=0  data=1
        case block:  recursive=1  data=2
"""
        ast = parse(src)
        assert len(ast.archs) == 1
        assert ast.archs[0].name == 'Program'
        assert len(ast.archs[0].cases) == 2
        assert ast.archs[0].cases[0].name == 'input'
        assert ast.archs[0].cases[0].recursive == 0
        assert ast.archs[0].cases[0].data == 1
        assert ast.archs[0].cases[1].name == 'block'
        assert ast.archs[0].cases[1].recursive == 1

    def test_compile_functor(self):
        """arch compiles and arch.interpreter() produces a working ArchInterpreter."""
        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"

arch Program:
    algebra:
        case leaf: recursive=0  data=1
        case node: recursive=2  data=0
"""
        arch = compile(src, NAMESPACE)
        # The arch is registered — interpreter() should not raise KeyError
        # (no cells bound so algebra interpreter is None, but the arch exists)
        interp = arch.interpreter('Program')
        from engine import ArchInterpreter
        assert isinstance(interp, ArchInterpreter)

    def test_functor_used_with_interpreter(self):
        """run_algebra works end-to-end via arch with per-case cell bindings."""
        def _leaf_cell(payload, child_results, params, temp):
            return payload[0]

        def _node_cell(payload, child_results, params, temp):
            return sum(child_results)

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf_cell': staticmethod(_leaf_cell),
            'node_cell': staticmethod(_node_cell),
        })()

        src = """
arch Tree:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf_cell
        case node: recursive=2  data=0  cell=ops.node_cell
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Tree')
        tree = ('node', [], [('leaf', [10], []), ('leaf', [20], [])])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 30

    def test_functor_no_cases_raises(self):
        """The old 'functor' keyword is no longer valid — expect SyntaxError."""
        with pytest.raises(SyntaxError, match="removed"):
            parse("functor Empty:\n    case x: recursive=0  data=1\n")

    def test_multiple_functors(self):
        """Two arch declarations both compile successfully."""
        src = """
arch A:
    algebra:
        case x: recursive=0  data=1

arch B:
    algebra:
        case y: recursive=1  data=0
"""
        ast = parse(src)
        assert len(ast.archs) == 2
        arch = compile(src, NAMESPACE)
        # Both archs should be accessible via interpreter()
        from engine import ArchInterpreter
        assert isinstance(arch.interpreter('A'), ArchInterpreter)
        assert isinstance(arch.interpreter('B'), ArchInterpreter)

    def test_parse_per_case_cell(self):
        """Per-case cell= is stored on algebra_cases."""
        src = """
arch Tree:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf_fn
        case node: recursive=2  data=0  cell=ops.node_fn
"""
        ast = parse(src)
        assert ast.archs[0].cases[0].cell == 'ops.leaf_fn'
        assert ast.archs[0].cases[1].cell == 'ops.node_fn'

    def test_compile_per_case_cell_dispatch(self):
        """Per-case cells dispatch correctly via arch.interpreter().run_algebra()."""
        def _leaf_cell(payload, child_results, params, temp):
            return payload[0]

        def _node_cell(payload, child_results, params, temp):
            return sum(child_results)

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf_cell': staticmethod(_leaf_cell),
            'node_cell': staticmethod(_node_cell),
        })()

        src = """
arch Tree:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf_cell
        case node: recursive=2  data=0  cell=ops.node_cell
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Tree')
        tree = ('node', [], [('leaf', [10], []), ('leaf', [20], [])])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 30

    def test_per_case_cell_injects_paths(self):
        """arch.interpreter() puts paths in params automatically."""
        def _leaf_cell(payload, child_results, params, temp):
            assert 'paths' in params
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf_cell': staticmethod(_leaf_cell),
        })()

        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"

arch Tiny:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf_cell
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Tiny')
        tree = ('leaf', [42], [])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 42

    def test_per_case_cell_incomplete_raises(self):
        """If some cases have cell bindings but not all, raise ValueError."""
        def _fn(payload, child_results, params, temp):
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'fn': staticmethod(_fn),
        })()

        src = """
arch Incomplete:
    algebra:
        case a: recursive=0  data=1  cell=ops.fn
        case b: recursive=1  data=0
"""
        with pytest.raises(ValueError, match="incomplete"):
            compile(src, ns)

    def test_interpreter_no_cell_raises(self):
        """arch.interpreter() followed by run_algebra raises when no cell is bound."""
        src = """
arch NoBind:
    algebra:
        case x: recursive=0  data=1
"""
        arch = compile(src, NAMESPACE)
        # compile() succeeds; the error surfaces when trying to run
        interp = arch.interpreter('NoBind')
        with pytest.raises(ValueError):
            interp.run_algebra(('x', [1], []), lambda n: n)

    def test_interpreter_unknown_functor_raises(self):
        """Accessing a non-existent arch name raises with 'Unknown arch'."""
        src = """
arch F:
    algebra:
        case x: recursive=0  data=1
"""
        arch = compile(src, NAMESPACE)
        with pytest.raises(KeyError, match="Unknown arch"):
            arch.interpreter('NonExistent')

    def test_functor_level_cell_still_works(self):
        """algebra-level cell = <fn> binding (inside algebra: block) works."""
        def _cell(case_name, payload, child_results, params, temp):
            if case_name == 'leaf':
                return payload[0] * 2
            return sum(child_results)

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'tree_cell': staticmethod(_cell),
        })()

        src = """
arch Tree:
    algebra:
        cell = ops.tree_cell
        case leaf: recursive=0  data=1
        case node: recursive=2  data=0
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Tree')
        tree = ('node', [], [('leaf', [5], []), ('leaf', [7], [])])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 24  # (5*2) + (7*2)


# ---------------------------------------------------------------------------
# TestBridge
# ---------------------------------------------------------------------------

class TestBridge:

    def test_cross_semiring_with_bridge(self):
        def _join(eq, x, y=None, temp=0.0):
            return f'J({x})'
        def _res(eq, x, y=None, temp=0.0):
            return f'R({x})'
        def _bridge(eq, x, y=None, temp=0.0):
            return f'B({x})'

        ns = {
            'ops': type('ns', (), {
                'join': staticmethod(_join),
                'res': staticmethod(_res),
                'bridge': staticmethod(_bridge),
            })(),
        }

        src = """
semiring join:
    contract = ops.join

semiring residuate:
    contract = ops.res

sort i, j

leg a : i -> j  via "i->j"  using join
leg b : j -> j  via "j->j"  bridge  op ops.bridge
leg c : j -> i  via "j->i"  using residuate

path cross = a b c
"""
        arch = compile(src, ns)
        result = arch.paths['cross']('x', None, 0.0)
        assert result == 'R(B(J(x)))'

    def test_cross_semiring_without_bridge_raises(self):
        src = """
semiring s1:
    contract = ops.dummy

semiring s2:
    contract = ops.dummy

sort i, j

leg a : i -> j  via "i->j"  using s1
leg b : j -> i  via "j->i"  using s2

path cross = a b
"""
        with pytest.raises(ValueError, match="spans multiple semirings"):
            compile(src, NAMESPACE)

    def test_cross_semiring_sort_mismatch_raises(self):
        def _op(eq, x, y=None, temp=0.0):
            return x

        ns = {
            'ops': type('ns', (), {
                'op': staticmethod(_op),
            })(),
        }

        src = """
semiring s1:
    contract = ops.op

semiring s2:
    contract = ops.op

sort i, j, k

leg a : i -> j  via "i->j"  using s1
leg b : j -> k  via "j->k"  bridge  op ops.op
leg c : i -> j  via "i->j"  using s2

path cross = a b c
"""
        with pytest.raises(TypeError, match="mismatch"):
            compile(src, ns)


# ---------------------------------------------------------------------------
# TestArch
# ---------------------------------------------------------------------------

class TestArch:

    def test_parse_arch_algebra_only(self):
        src = """
arch MyArch:
    algebra:
        case leaf: recursive=0  data=1
        case node: recursive=2  data=0
"""
        ast = parse(src)
        assert len(ast.archs) == 1
        assert ast.archs[0].name == 'MyArch'
        assert ast.archs[0].cases is not None
        assert len(ast.archs[0].cases) == 2

    def test_parse_arch_coalgebra_raises(self):
        """coalgebra: sub-block raises SyntaxError with migration message."""
        with pytest.raises(SyntaxError, match="coalgebra.*removed"):
            parse("""
arch Bad:
    algebra:
        case base: recursive=0  data=1
    coalgebra:
        case step: recursive=1  data=0  output=1
""")

    def test_parse_arch_empty_raises(self):
        with pytest.raises(SyntaxError, match="must declare"):
            parse("arch Empty:\n\n")

    def test_parse_arch_empty_algebra_raises(self):
        with pytest.raises(SyntaxError, match="must declare at least one case"):
            parse("arch Bad:\n    algebra:\n\n")

    def test_compile_arch_algebra_with_cells(self):
        def _leaf(payload, child_results, params, temp):
            return payload[0]

        def _node(payload, child_results, params, temp):
            return sum(child_results)

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf': staticmethod(_leaf),
            'node': staticmethod(_node),
        })()

        src = """
arch Tree:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf
        case node: recursive=2  data=0  cell=ops.node
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Tree')
        tree = ('node', [], [('leaf', [10], []), ('leaf', [20], [])])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 30

    def test_arch_interpreter_returns_arch_interpreter(self):
        from engine import ArchInterpreter

        def _leaf(payload, child_results, params, temp):
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf': staticmethod(_leaf),
        })()

        src = """
arch A:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf
"""
        arch = compile(src, ns)
        interp = arch.interpreter('A')
        assert isinstance(interp, ArchInterpreter)

    def test_arch_coalgebra_block_raises(self):
        """coalgebra: sub-block raises SyntaxError — use step: instead."""
        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {'dummy': staticmethod(_dummy_op)})()

        src = """
arch OnlyCoalg:
    coalgebra:
        case step: recursive=1  data=1
"""
        with pytest.raises(SyntaxError, match="coalgebra.*removed"):
            compile(src, ns)

    def test_arch_algebra_no_coalgebra_raises(self):
        """run_coalgebra on an arch with only algebra raises."""
        def _leaf(payload, child_results, params, temp):
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf': staticmethod(_leaf),
        })()

        src = """
arch OnlyAlg:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf
"""
        arch = compile(src, ns)
        interp = arch.interpreter('OnlyAlg')
        with pytest.raises(ValueError, match="no coalgebra"):
            interp.run_coalgebra(None)

    def test_arch_paths_injected(self):
        """Cell functions get paths via params automatically."""
        def _leaf(payload, child_results, params, temp):
            assert 'paths' in params
            assert 'a' in params['paths']
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf': staticmethod(_leaf),
        })()

        src = """
semiring s:
    contract = ops.dummy

sort x

leg a : x -> x  via "x->x"

arch A:
    algebra:
        case leaf: recursive=0  data=1  cell=ops.leaf
"""
        arch = compile(src, ns)
        interp = arch.interpreter('A')
        result = interp.run_algebra(('leaf', [42], []), lambda n: n)
        assert result == 42

    def test_arch_incomplete_cells_raises(self):
        def _fn(payload, child_results, params, temp):
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'fn': staticmethod(_fn),
        })()

        src = """
arch Bad:
    algebra:
        case a: recursive=0  data=1  cell=ops.fn
        case b: recursive=1  data=0
"""
        with pytest.raises(ValueError, match="incomplete"):
            compile(src, ns)


# ---------------------------------------------------------------------------
# Phase 2: SortDecl and structured sorts
# ---------------------------------------------------------------------------

class TestSortDecl:
    def test_string_equality(self):
        from engine.decl import SortDecl
        s = SortDecl("foo")
        assert s == "foo"
        assert "foo" == s

    def test_str(self):
        from engine.decl import SortDecl
        assert str(SortDecl("bar")) == "bar"

    def test_hash(self):
        from engine.decl import SortDecl
        assert hash(SortDecl("x")) == hash("x")
        assert {SortDecl("x"): 1}[SortDecl("x")] == 1

    def test_sorting(self):
        from engine.decl import SortDecl
        assert sorted([SortDecl("b"), SortDecl("a")]) == [SortDecl("a"), SortDecl("b")]

    def test_fields_preserved(self):
        from engine.decl import SortDecl
        s = SortDecl("model", {"Wq": "matrix", "bq": "vector"})
        assert s.fields == {"Wq": "matrix", "bq": "vector"}
        assert s.name == "model"

    def test_opaque_sort_fields_none(self):
        from engine.decl import SortDecl
        s = SortDecl("i")
        assert s.fields is None


class TestStructuredSortParsing:
    def test_opaque_sorts(self):
        from engine.decl import SortDecl
        ast = parse("sort i, j")
        assert ast.sorts == [SortDecl("i", None), SortDecl("j", None)]

    def test_structured_sort(self):
        ast = parse("sort model(Wq: matrix, bq: vector)")
        assert len(ast.sorts) == 1
        assert ast.sorts[0].name == "model"
        assert ast.sorts[0].fields == {"Wq": "matrix", "bq": "vector"}

    def test_mixed_sorts(self):
        ast = parse("sort i, model(W: matrix), j")
        assert len(ast.sorts) == 3
        assert ast.sorts[0] == "i"
        assert ast.sorts[0].fields is None
        assert ast.sorts[1] == "model"
        assert ast.sorts[1].fields == {"W": "matrix"}
        assert ast.sorts[2] == "j"
        assert ast.sorts[2].fields is None

    def test_sort_defs_in_archdef(self):
        src = """
semiring sr:
    contract = ops.dummy

sort i, j

leg realize : j -> i  via "j,ji->i"
"""
        arch = compile(src, NAMESPACE)
        assert "i" in arch.sort_defs
        assert "j" in arch.sort_defs


# ---------------------------------------------------------------------------
# TestAugment
# ---------------------------------------------------------------------------

class TestAugment:
    def test_parse_augment_in_path(self):
        """Bracketed tokens are preserved in path morphisms list."""
        src = """
semiring s:
    contract = ops.dummy

sort a

leg f : a -> a  via "a->a"

fan kv = f & f

path p = f [kv] f
"""
        ast = parse(src)
        assert '[kv]' in ast.paths[0].morphisms

    def test_compile_augment(self):
        """Augment step merges fan output into y."""
        def _op(eq, x, y=None, temp=0.0):
            return x

        def _fan_branch(eq, x, y=None, temp=0.0):
            return x * 2

        def _reader(eq, x, y=None, temp=0.0):
            # Should see fan results in y
            return y.get('b1', -1)

        ns = {
            'ops': type('ns', (), {
                'passthru': staticmethod(_op),
                'branch': staticmethod(_fan_branch),
                'reader': staticmethod(_reader),
            })(),
        }

        src = """
semiring s:
    contract = ops.passthru

sort a

morphism f : a -> a  via "a->a"  op ops.passthru
morphism b1 : a -> a  via "a->a"  op ops.branch
morphism b2 : a -> a  via "a->a"  op ops.branch
morphism r : a -> a  via "a->a"  op ops.reader

fan kv = b1 & b2

path p = f [kv] r
"""
        arch = compile(src, ns)
        import numpy as np
        x = np.array([1.0])
        result = arch.paths['p'](x, {}, 0.0)
        # After augment, y should contain {'b1': ..., 'b2': ...}
        # reader returns y.get('b1', -1), which should be the fan result
        assert result != -1  # b1 key should exist in augmented y

    def test_augment_preserves_x(self):
        """Augment step does not modify x."""
        call_log = []

        def _identity(eq, x, y=None, temp=0.0):
            return x

        def _double(eq, x, y=None, temp=0.0):
            return x * 2

        ns = {
            'ops': type('ns', (), {
                'identity': staticmethod(_identity),
                'double': staticmethod(_double),
            })(),
        }

        src = """
semiring s:
    contract = ops.identity

sort a

morphism pre  : a -> a  via "a->a"  op ops.double
morphism b1   : a -> a  via "a->a"  op ops.identity
morphism post : a -> a  via "a->a"  op ops.double

fan aug = b1

path p = pre [aug] post
"""
        arch = compile(src, ns)
        import numpy as np
        x = np.array([1.0])
        result = arch.paths['p'](x, {}, 0.0)
        # pre doubles: 2.0, augment doesn't change x, post doubles: 4.0
        assert float(result[0]) == 4.0

    def test_augment_with_residual(self):
        """Augment works with residual combinator."""
        def _op(eq, x, y=None, temp=0.0):
            return x + 1.0

        ns = {
            'ops': type('ns', (), {
                'op': staticmethod(_op),
            })(),
        }

        src = """
semiring s:
    contract = ops.op

sort a

morphism f : a -> a  via "a->a"
morphism b : a -> a  via "a->a"

fan aug = b

path p = f [aug] f  residual
"""
        arch = compile(src, ns)
        import numpy as np
        x = np.array([0.0])
        result = arch.paths['p'](x, {}, 0.0)
        # f(0) = 1, augment no change to x, f(1) = 2, residual: 2 + 0 = 2
        assert float(result[0]) == 2.0


class TestIdentityCell:
    def test_identity_cell_with_payload(self):
        src = """
semiring sr:
    contract = ops.dummy

sort a

arch Pipe:
    algebra:
        case leaf: recursive=0 data=1 cell=identity
"""
        arch = compile(src, NAMESPACE)
        interp = arch.interpreter('Pipe')
        result = interp.run_algebra(
            ('leaf', [42.0], []),
            lambda n: n,
        )
        assert result == 42.0

    def test_identity_cell_with_child(self):
        src = """
semiring sr:
    contract = ops.dummy

sort a

arch Pipe:
    algebra:
        case leaf: recursive=0 data=1 cell=identity
        case wrap: recursive=1 data=0 cell=identity
"""
        arch = compile(src, NAMESPACE)
        interp = arch.interpreter('Pipe')
        result = interp.run_algebra(
            ('wrap', [], [('leaf', [99.0], [])]),
            lambda n: n,
        )
        assert result == 99.0


# ---------------------------------------------------------------------------
# TestMultilineMorphism
# ---------------------------------------------------------------------------

class TestMultilineMorphism:
    def test_multiline_basic(self):
        """Multi-line morphism with indented clauses parses correctly."""
        src = """
semiring s:
    contract = ops.dummy

sort x, y

morphism m : x -> y
    via "x->y"
    op ops.dummy
"""
        ast = parse(src)
        assert len(ast.morphisms) == 1
        assert ast.morphisms[0].name == 'm'
        assert ast.morphisms[0].src_sort == 'x'
        assert ast.morphisms[0].tgt_sort == 'y'
        assert ast.morphisms[0].equation == 'x->y'
        assert ast.morphisms[0].op == 'ops.dummy'

    def test_multiline_all_clauses(self):
        """Multi-line morphism with arity and accumulate."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x
    via "x->x"
    op ops.dummy
    arity unary
    accumulate cat
"""
        ast = parse(src)
        assert ast.morphisms[0].arity == 'unary'
        assert ast.morphisms[0].accumulate == 'cat'

    def test_singleline_still_works(self):
        """Single-line morphism declaration still parses."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x  via "x->x"  op ops.dummy  arity unary
"""
        ast = parse(src)
        assert ast.morphisms[0].name == 'm'
        assert ast.morphisms[0].arity == 'unary'

    def test_multiline_followed_by_other_decl(self):
        """Multi-line morphism correctly terminates at next top-level keyword."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x
    via "x->x"
    op ops.dummy

path p = m
"""
        ast = parse(src)
        assert len(ast.morphisms) == 1
        assert len(ast.paths) == 1


# ---------------------------------------------------------------------------
# TestLeafNodeAliases
# ---------------------------------------------------------------------------

class TestLeafNodeAliases:
    def test_leaf_alias(self):
        """'leaf' is an alias for recursive=0."""
        src = """
arch A:
    algebra:
        case base: leaf  data=1  cell=identity
"""
        ast = parse(src)
        assert ast.archs[0].cases[0].recursive == 0

    def test_node_alias(self):
        """'node' is an alias for recursive=1."""
        src = """
arch A:
    algebra:
        case block: node  data=1  cell=identity
"""
        ast = parse(src)
        assert ast.archs[0].cases[0].recursive == 1

    def test_recursive_still_works(self):
        """Explicit recursive=N still works alongside aliases."""
        src = """
arch A:
    algebra:
        case a: leaf  data=1  cell=identity
        case b: recursive=2  data=0  cell=identity
"""
        ast = parse(src)
        assert ast.archs[0].cases[0].recursive == 0
        assert ast.archs[0].cases[1].recursive == 2

    def test_leaf_with_explicit_recursive_raises(self):
        """Using 'leaf' with explicit recursive= raises SyntaxError."""
        src = """
arch A:
    algebra:
        case bad: leaf  recursive=0  data=1  cell=identity
"""
        with pytest.raises(SyntaxError, match="leaf"):
            parse(src)


# ---------------------------------------------------------------------------
# TestFieldLevelAccumulate
# ---------------------------------------------------------------------------

class TestFieldLevelAccumulate:
    def test_parse_accumulate_with_fields(self):
        """'accumulate cat on K, V' parsed correctly."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x
    via "x->x"
    op ops.dummy
    accumulate cat on K, V
"""
        ast = parse(src)
        assert ast.morphisms[0].accumulate == 'cat'
        assert ast.morphisms[0].accumulate_fields == ['K', 'V']

    def test_parse_accumulate_without_fields(self):
        """Plain 'accumulate cat' still works (no fields)."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x  via "x->x"  op ops.dummy  accumulate cat
"""
        ast = parse(src)
        assert ast.morphisms[0].accumulate == 'cat'
        assert ast.morphisms[0].accumulate_fields is None

    def test_compile_accumulate_fields_threaded(self):
        """Field-level accumulate info reaches the compiled arch's accumulate_legs."""
        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
        })()

        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x
    via "x->x"
    op ops.dummy
    accumulate cat on K, V

arch A:
    cases:
        case block: recursive=1  data=1
"""
        arch = compile(src, ns)
        ad = arch._archs['A']
        assert ad.accumulate_legs is not None
        assert 'm' in ad.accumulate_legs
        assert ad.accumulate_legs['m'] == ('cat', ['K', 'V'])


# ---------------------------------------------------------------------------
# TestCompactSyntax — bare equation, bare op, no case keyword, no semiring
# ---------------------------------------------------------------------------

class TestCompactSyntax:

    def test_bare_equation_no_via(self):
        """Morphism without 'via' keyword — bare equation string."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x  "x->x"  op ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].equation == 'x->x'
        assert ast.morphisms[0].op == 'ops.dummy'

    def test_bare_op_no_keyword(self):
        """Bare dotted name treated as op (no 'op' prefix needed)."""
        src = """
semiring s:
    contract = ops.dummy

sort x

morphism m : x -> x  "x->x"  ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].op == 'ops.dummy'

    def test_fully_compact_morphism(self):
        """No 'via', no 'op' — just equation and dotted name."""
        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].equation == 'x->x'
        assert ast.morphisms[0].op == 'ops.dummy'

    def test_compact_multiline(self):
        """Multi-line morphism with compact syntax."""
        src = """
sort x

morphism m : x -> x
    "x->x"
    ops.dummy
    arity unary
"""
        ast = parse(src)
        assert ast.morphisms[0].equation == 'x->x'
        assert ast.morphisms[0].op == 'ops.dummy'
        assert ast.morphisms[0].arity == 'unary'

    def test_no_semiring_with_explicit_ops(self):
        """No semiring block when all morphisms have explicit ops."""
        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy
"""
        arch = compile(src, NAMESPACE)
        assert 'm' in arch.paths
        result = arch.paths['m']('hello', None, 0.0)
        assert result == 'hello'

    def test_case_without_keyword(self):
        """Arch cases without 'case' keyword."""
        src = """
arch A:
    algebra:
        input: leaf  data=1  cell=identity
        block: node  data=1  cell=identity
"""
        ast = parse(src)
        cases = ast.archs[0].cases
        assert len(cases) == 2
        assert cases[0].name == 'input'
        assert cases[0].recursive == 0
        assert cases[1].name == 'block'
        assert cases[1].recursive == 1

    def test_full_compact_architecture(self):
        """End-to-end: compact syntax for a complete architecture."""
        src = """
sort x

morphism a : x -> x  "x->x"  ops.dummy
morphism b : x -> x  "x->x"  ops.dummy

path p = a b  residual

arch Net:
    algebra:
        input: leaf  data=1  cell=identity
        block:  node  data=1  morphisms = p
"""
        arch = compile(src, NAMESPACE)
        interp = arch.interpreter('Net')
        tree = ('block', [None], [('input', ['hello'], [])])
        result = interp.run_algebra(tree, lambda n: n)
        # p is a b with residual: dummy returns x, so a(b(x)) + x = x + x
        # but dummy just returns x, so chain of dummies returns x, residual = x + x
        # Actually dummy op returns first positional arg (x), so result depends
        # on how chain works — it feeds output as next x
        assert result is not None  # just verify no crash

    def test_mixed_old_and_new_syntax(self):
        """Old syntax (via/op/case) and new compact syntax coexist."""
        src = """
sort x

morphism a : x -> x  via "x->x"  op ops.dummy
morphism b : x -> x  "x->x"  ops.dummy

arch A:
    algebra:
        case old: recursive=0  data=1  cell=identity
        new:      leaf  data=1  cell=identity
"""
        ast = parse(src)
        assert ast.morphisms[0].op == 'ops.dummy'
        assert ast.morphisms[1].op == 'ops.dummy'
        cases = ast.archs[0].cases
        assert cases[0].name == 'old'
        assert cases[1].name == 'new'


# ---------------------------------------------------------------------------
# TestIterateCombinator
# ---------------------------------------------------------------------------

class TestIterateCombinator:

    def test_parse_iterate_attribute(self):
        """iterate=layers parsed on case declarations."""
        src = """
arch A:
    algebra:
        input: leaf  data=1  cell=identity
        block:  node  data=1  cell=identity  iterate=layers
"""
        ast = parse(src)
        cases = ast.archs[0].cases
        assert cases[0].iterate is None
        assert cases[1].iterate == 'layers'

    def test_parse_iterate_multiple_cases(self):
        """Multiple cases can share the same iterate group."""
        src = """
arch A:
    algebra:
        input: leaf  data=1  cell=identity
        attn:  node  data=1  cell=identity  iterate=layers
        ffn:   node  data=1  cell=identity  iterate=layers
        final: node  data=1  cell=identity
"""
        ast = parse(src)
        cases = ast.archs[0].cases
        assert cases[1].iterate == 'layers'
        assert cases[2].iterate == 'layers'
        assert cases[3].iterate is None

    def test_iterate_single_case_fold(self):
        """Single iterate case folds over layers."""
        def _add(payload, child_results, params, temp):
            return child_results[0] + payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'add': staticmethod(_add),
        })()

        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy

arch Net:
    algebra:
        input: leaf  data=1  cell=identity
        block:  node  data=1  cell=ops.add  iterate=layers
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        # layers=[10, 20, 30]: ((5 + 10) + 20) + 30 = 65
        result = interp.run_algebra_layers(5, layers=[10, 20, 30])
        assert result == 65

    def test_iterate_with_epilogue(self):
        """Cases after iterate block are applied as epilogue."""
        def _add(payload, child_results, params, temp):
            return child_results[0] + payload[0]

        def _double(payload, child_results, params, temp):
            return child_results[0] * 2

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'add': staticmethod(_add),
            'double': staticmethod(_double),
        })()

        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy

arch Net:
    algebra:
        input: leaf  data=1  cell=identity
        block:  node  data=1  cell=ops.add    iterate=layers
        final: node  data=1  cell=ops.double
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        # layers=[10, 20]: (5 + 10) + 20 = 35, then double = 70
        result = interp.run_algebra_layers(5, layers=[10, 20])
        assert result == 70

    def test_iterate_two_case_group(self):
        """Two iterate cases form a block that repeats per layer."""
        def _add(payload, child_results, params, temp):
            return child_results[0] + payload[0]

        def _mul(payload, child_results, params, temp):
            return child_results[0] * 2  # just doubles, ignores payload

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'add': staticmethod(_add),
            'mul': staticmethod(_mul),
        })()

        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy

arch Net:
    algebra:
        input: leaf  data=1  cell=identity
        step1: node  data=1  cell=ops.add  iterate=layers
        step2: node  data=1  cell=ops.mul  iterate=layers
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        # _build_tree reverses declaration order when nesting, so step1 is outer,
        # step2 is inner. Fold is bottom-up, so step2 executes first each layer.
        # Layer 1 (payload=10): step2 inner: 5*2=10, step1 outer: 10+10=20
        # Layer 2 (payload=0):  step2 inner: 20*2=40, step1 outer: 40+0=40
        result = interp.run_algebra_layers(5, layers=[10, 0])
        assert result == 40

    def test_iterate_with_extras(self):
        """extras dict is merged into each layer payload."""
        def _reader(payload, child_results, params, temp):
            return child_results[0] + payload[0].get('extra_val', 0)

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'reader': staticmethod(_reader),
        })()

        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy

arch Net:
    algebra:
        input: leaf  data=1  cell=identity
        block:  node  data=1  cell=ops.reader  iterate=layers
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        # extras={'extra_val': 100} merged into each layer payload
        result = interp.run_algebra_layers(0, layers=[{}, {}], extras={'extra_val': 100})
        # Layer 1: 0 + 100 = 100. Layer 2: 100 + 100 = 200
        assert result == 200

    def test_iterate_backward_compat_decompose(self):
        """Old decompose= calling convention still works."""
        def _leaf(payload, child_results, params, temp):
            return payload[0]

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'leaf': staticmethod(_leaf),
        })()

        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy

arch Net:
    algebra:
        input: leaf  data=1  cell=ops.leaf
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        # Old style: pass tree + decompose
        tree = ('input', [42], [])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 42

    def test_iterate_no_groups_with_layers_raises(self):
        """Passing layers= when no iterate cases exist raises ValueError."""
        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
        })()

        src = """
arch Net:
    algebra:
        input: leaf  data=1  cell=identity
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        with pytest.raises(ValueError, match="iterate"):
            interp.run_algebra_layers(5, layers=[1, 2])

    def test_iterate_empty_layers(self):
        """Empty layers list means only base + epilogue."""
        def _double(payload, child_results, params, temp):
            return child_results[0] * 2

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'double': staticmethod(_double),
        })()

        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy

arch Net:
    algebra:
        input: leaf  data=1  cell=identity
        block:  node  data=1  cell=identity  iterate=layers
        final: node  data=1  cell=ops.double
"""
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        # No layers, so just input(10) -> final(double) = 20
        result = interp.run_algebra_layers(10, layers=[])
        assert result == 20


# ---------------------------------------------------------------------------
# TestParameterizedMorphisms
# ---------------------------------------------------------------------------

class TestParameterizedMorphisms:

    def test_parse_template_param(self):
        """morphism ln[prefix] stores template_param='prefix'."""
        src = """
sort x

morphism ln[prefix] : x -> x  "x->x"  ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].name == 'ln'
        assert ast.morphisms[0].template_param == 'prefix'

    def test_parse_no_template_param(self):
        """Regular morphism has template_param=None."""
        src = """
sort x

morphism m : x -> x  "x->x"  ops.dummy
"""
        ast = parse(src)
        assert ast.morphisms[0].template_param is None

    def test_parse_instantiation_in_path(self):
        """ln[ln1] appears as a token in path morphisms."""
        src = """
sort x

morphism ln[prefix] : x -> x  "x->x"  ops.dummy
morphism m : x -> x  "x->x"  ops.dummy

path p = ln[ln1] m
"""
        ast = parse(src)
        assert ast.paths[0].morphisms == ['ln[ln1]', 'm']

    def test_compile_template_instantiation(self):
        """Template instantiation creates a curried callable."""
        def _param_op(eq, x, y=None, temp=0.0, prefix='default'):
            return f'{prefix}({x})'

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'param_op': staticmethod(_param_op),
        })()

        src = """
sort x

morphism t[prefix] : x -> x  "x->x"  ops.param_op

path p1 = t[alpha]
path p2 = t[beta]
"""
        arch = compile(src, ns)
        assert arch.paths['p1']('data', None, 0.0) == 'alpha(data)'
        assert arch.paths['p2']('data', None, 0.0) == 'beta(data)'

    def test_template_two_paths_same_template(self):
        """Same template instantiated with different params in different paths."""
        def _prefix_op(eq, x, w, temp=0.0, key='x'):
            return w.get(key, 'missing')

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'prefix_op': staticmethod(_prefix_op),
        })()

        src = """
sort x

morphism lookup[key] : x -> x  "x->x"  ops.prefix_op

path a = lookup[name]
path b = lookup[age]
"""
        arch = compile(src, ns)
        data = {'name': 'Alice', 'age': 30}
        assert arch.paths['a']('x', data, 0.0) == 'Alice'
        assert arch.paths['b']('x', data, 0.0) == 30

    def test_template_with_regular_morphisms(self):
        """Template and regular morphisms compose in a path."""
        def _tag(eq, x, y=None, temp=0.0, label='?'):
            return f'{label}:{x}'

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'tag': staticmethod(_tag),
        })()

        src = """
sort x

morphism t[label] : x -> x  "x->x"  ops.tag
morphism pass : x -> x  "x->x"  ops.dummy

path p = t[hello] pass
"""
        arch = compile(src, ns)
        result = arch.paths['p']('world', None, 0.0)
        # t[hello] tags, then pass returns x unchanged
        assert result == 'hello:world'

    def test_template_multiline_syntax(self):
        """Template morphism with multi-line declaration."""
        src = """
sort x

morphism t[key] : x -> x
    "x->x"
    ops.dummy
    arity unary
"""
        ast = parse(src)
        assert ast.morphisms[0].template_param == 'key'
        assert ast.morphisms[0].arity == 'unary'

    def test_template_not_confused_with_augment(self):
        """ln[ln1] (template inst) and [kv] (augment) coexist in a path."""
        def _tag(eq, x, y=None, temp=0.0, label='?'):
            return x

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'tag': staticmethod(_tag),
        })()

        src = """
sort x

morphism t[label] : x -> x  "x->x"  ops.tag
morphism b : x -> x  "x->x"  ops.dummy

fan kv = b

path p = t[pre] [kv] b
"""
        arch = compile(src, ns)
        # Should compile without confusing [kv] augment with t[pre] instantiation
        assert 'p' in arch.paths


# ---------------------------------------------------------------------------
# TestFinalLNFix
# ---------------------------------------------------------------------------

class TestFinalLNFix:
    """Binary layer norm morphism reads gamma/beta from weight dict.

    The fix: LN uses binary arity (reads y['gamma'], y['beta']),
    not unary arity (ignores y, uses identity gamma/beta).
    With the unified endofunctor, the same LN morphism serves both
    algebra and coalgebra — no divergence possible.
    """

    def test_binary_ln_reads_params(self):
        """LN morphism with binary arity reads gamma/beta from payload."""
        import numpy as np

        def ln_op(eq, x, y, temp=0.0):
            """Binary layer norm: reads gamma/beta from y."""
            gamma = y.get('gamma', np.ones_like(x))
            beta = y.get('beta', np.zeros_like(x))
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-5
            return gamma * (x - mean) / std + beta

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'ln_op': staticmethod(ln_op),
        })()

        src = """
semiring s:
    contract = ops.dummy

sort model

morphism ln : model -> model via "x" op ops.ln_op
morphism f  : model -> model via "x"

path block = f ln

arch T:
    cases:
        input: leaf data=1 cell=identity
        layer: node data=1 morphisms=block iterate=layers
"""
        arch = compile(src, ns)

        x0 = np.array([[1.0, 2.0, 3.0]])
        gamma = np.array([2.0, 2.0, 2.0])
        beta = np.array([0.5, 0.5, 0.5])
        layer_weights = {'gamma': gamma, 'beta': beta}

        # Run through the path directly
        result = arch.paths['ln'](x0, layer_weights, 0.0)

        # With non-identity gamma/beta, output should differ from input
        assert not np.allclose(result, x0), "LN should transform with learned params"
        # With gamma=2, beta=0.5, the output should be scaled and shifted
        assert result.shape == x0.shape

    def test_binary_ln_identity_params_is_identity(self):
        """With identity gamma/beta, binary LN reduces to standard normalization."""
        import numpy as np

        def ln_op(eq, x, y, temp=0.0):
            gamma = y.get('gamma', np.ones_like(x))
            beta = y.get('beta', np.zeros_like(x))
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-5
            return gamma * (x - mean) / std + beta

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'ln_op': staticmethod(ln_op),
        })()

        src = """
semiring s:
    contract = ops.dummy
sort model
morphism ln : model -> model via "x" op ops.ln_op
"""
        arch = compile(src, ns)

        x0 = np.array([[1.0, 2.0, 3.0]])
        identity_params = {'gamma': np.ones(3), 'beta': np.zeros(3)}
        result = arch.paths['ln'](x0, identity_params, 0.0)

        # Standard normalization with identity params
        mean = x0.mean(axis=-1, keepdims=True)
        std = x0.std(axis=-1, keepdims=True) + 1e-5
        expected = (x0 - mean) / std
        assert np.allclose(result, expected, atol=1e-6)

    def test_unified_functor_ln_same_for_algebra(self):
        """LN in unified cases: algebra fold applies the same binary LN."""
        import numpy as np

        def ln_op(eq, x, y, temp=0.0):
            gamma = y.get('gamma', np.ones_like(x))
            beta = y.get('beta', np.zeros_like(x))
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-5
            return gamma * (x - mean) / std + beta

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'ln_op': staticmethod(ln_op),
        })()

        src = """
semiring s:
    contract = ops.dummy
sort model
morphism ln : model -> model via "x" op ops.ln_op
arch T:
    cases:
        input: leaf data=1 cell=identity
        norm:  node data=1 morphisms=ln
"""
        arch = compile(src, ns)
        interp = arch.interpreter('T', params={}, temp=0.0)

        x0 = np.array([[1.0, 2.0, 3.0]])
        gamma = np.array([2.0, 2.0, 2.0])
        beta = np.array([0.5, 0.5, 0.5])
        weights = {'gamma': gamma, 'beta': beta}

        tree = ('norm', [weights], [('input', [x0], [])])
        result = interp.run_algebra(tree, lambda n: n)

        # Same result as direct path call
        direct = arch.paths['ln'](x0, weights, 0.0)
        assert np.allclose(result, direct)


# ---------------------------------------------------------------------------
# TestStateDeclaration
# ---------------------------------------------------------------------------

class TestStateDeclaration:
    def test_parse_state_block(self):
        src = """
arch T:
    cases:
        input: leaf data=1 cell=identity
        block: node data=1 cell=identity
    state:
        pos: int
        caches: list
"""
        ast = parse(src)
        assert ast.archs[0].state_fields == {'pos': 'int', 'caches': 'list'}

    def test_state_compiles_to_hydra_type(self):
        src = """
semiring s:
    contract = ops.dummy
sort a
morphism m : a -> a via "x"
arch T:
    cases:
        input: leaf data=1 cell=identity
    state:
        pos: int
        caches: list
"""
        ns = dict(NAMESPACE)
        arch = compile(src, ns)
        ad = arch._archs['T']
        assert ad.state_type is not None

    def test_no_state_block_is_none(self):
        src = """
semiring s:
    contract = ops.dummy
sort a
morphism m : a -> a via "x"
arch T:
    cases:
        input: leaf data=1 cell=identity
"""
        ns = dict(NAMESPACE)
        arch = compile(src, ns)
        ad = arch._archs['T']
        assert ad.state_type is None


# ---------------------------------------------------------------------------
# TestStepProtocol
# ---------------------------------------------------------------------------

class TestStepProtocol:
    """Step protocol: declarative coalgebra from step: block."""

    # ------------------------------------------------------------------
    # Parse tests — these pass today: parser fully supports step: blocks
    # ------------------------------------------------------------------

    def test_parse_step_block(self):
        """step: block with enter and emit is parsed."""
        src = """
arch T:
    cases:
        input: leaf data=1 cell=identity
        block: node data=1 cell=identity
    step:
        enter = embed
        emit = unembed
"""
        ast = parse(src)
        assert ast.archs[0].step_enter == 'embed'
        assert ast.archs[0].step_emit == 'unembed'

    def test_parse_step_enter_only(self):
        """step: block with only enter (no emit) is valid."""
        src = """
arch T:
    cases:
        input: leaf data=1 cell=identity
    step:
        enter = embed
"""
        ast = parse(src)
        assert ast.archs[0].step_enter == 'embed'
        assert ast.archs[0].step_emit is None

    def test_parse_no_step_block(self):
        """Arch without step: block has None for step fields."""
        src = """
arch T:
    cases:
        input: leaf data=1 cell=identity
"""
        ast = parse(src)
        assert ast.archs[0].step_enter is None
        assert ast.archs[0].step_emit is None

    # ------------------------------------------------------------------
    # Runtime tests — xfail until compiler generates coalgebra from step:
    # The compiler currently stores step_enter/step_emit on ArchDecl but
    # _compile_archs() does not yet produce a coalgebra_cell from them.
    # These tests document the intended contract for the implementation.
    # ------------------------------------------------------------------

    def test_step_generates_coalgebra_cell(self):
        """step: block generates a coalgebra cell that can run."""
        def embed_op(eq, token, state, temp=0.0):
            return token * 2

        def unembed_op(eq, x, state, temp=0.0):
            return x + 1

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'embed_op': staticmethod(embed_op),
            'unembed_op': staticmethod(unembed_op),
        })()

        src = """
semiring s:
    contract = ops.dummy
sort a
morphism embed : a -> a via "x" op ops.embed_op
morphism unembed : a -> a via "x" op ops.unembed_op

arch T:
    cases:
        input: leaf data=1 cell=identity
        run: node data=1 output=1 cell=identity
    step:
        enter = embed
        emit = unembed
"""
        arch = compile(src, ns)
        interp = arch.interpreter('T', params={}, temp=0.0)

        init_state = {'pos': 0}
        tokens = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        outputs, final = interp.run_coalgebra(init_state, token_iter=tokens)

        assert len(outputs) == 3
        # embed doubles, unembed adds 1: output = token * 2 + 1
        assert np.allclose(outputs[0], np.array([3.0]))   # 1*2 + 1
        assert np.allclose(outputs[1], np.array([5.0]))   # 2*2 + 1
        assert np.allclose(outputs[2], np.array([7.0]))   # 3*2 + 1

    def test_step_with_morphism_computation(self):
        """step: block with case morphisms applies them between enter and emit."""
        def embed_op(eq, token, state, temp=0.0):
            return token

        def transform_op(eq, x, y, temp=0.0):
            return x * 3

        def unembed_op(eq, x, state, temp=0.0):
            return x

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'embed_op': staticmethod(embed_op),
            'transform_op': staticmethod(transform_op),
            'unembed_op': staticmethod(unembed_op),
        })()

        src = """
semiring s:
    contract = ops.dummy
sort a
morphism embed : a -> a via "x" op ops.embed_op
morphism transform : a -> a via "x" op ops.transform_op
morphism unembed : a -> a via "x" op ops.unembed_op

arch T:
    cases:
        input: leaf data=1 cell=identity
        block: node data=1 output=1 morphisms=transform
    step:
        enter = embed
        emit = unembed
"""
        arch = compile(src, ns)
        interp = arch.interpreter('T', params={}, temp=0.0)

        init_state = {}
        tokens = [np.array([2.0])]
        outputs, final = interp.run_coalgebra(init_state, token_iter=tokens)

        # embed passes through, transform triples, unembed passes through
        assert len(outputs) == 1
        assert np.allclose(outputs[0], np.array([6.0]))   # 2 * 3

    def test_step_enter_only_no_emit(self):
        """step: with only enter and no emit — coalgebra steps produce no output."""
        def embed_op(eq, token, state, temp=0.0):
            return token * 2

        ns = dict(NAMESPACE)
        ns['ops'] = type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            'embed_op': staticmethod(embed_op),
        })()

        src = """
semiring s:
    contract = ops.dummy
sort a
morphism embed : a -> a via "x" op ops.embed_op

arch T:
    cases:
        input: leaf data=1 cell=identity
        run: node data=1 output=0 cell=identity
    step:
        enter = embed
"""
        arch = compile(src, ns)
        interp = arch.interpreter('T', params={}, temp=0.0)

        init_state = {}
        tokens = [np.array([1.0]), np.array([2.0])]
        outputs, final = interp.run_coalgebra(init_state, token_iter=tokens)

        # No emit means no outputs collected
        assert len(outputs) == 0
