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

def _dsl(body: str, *, semiring: bool = True) -> str:
    """Wrap body in the standard single-semiring+sort-x preamble."""
    if semiring:
        return "semiring s:\n    contract = ops.dummy\n\nsort x\n\n" + body
    return "sort x\n\n" + body


def _coerce_dsl(coerce_decls: str, threshold: float, sorts: str = "a, b",
                morphisms: str = None, path: str = "path p = f g\n") -> str:
    """Build a minimal coercion DSL for TestSortCoercion tests."""
    if morphisms is None:
        morphisms = 'morphism f : a -> a  via "a->a"\nmorphism g : b -> b  via "b->b"\n\n'
    return (
        "semiring s:\n    contract = ops.dummy\n\n"
        f"sort {sorts}\n\n{coerce_decls}\n"
        f"sort_threshold {threshold}\n\n"
        + morphisms
        + path
    )


BASIC_SOURCE = """
semiring join:
    contract = ops.dummy

sort i, j

morphism realize   : j -> i  via "j,ji->i"
morphism propagate : i -> j  via "i,ij->j"

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

morphism a : x -> x  via "x->x"
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

morphism a : i -> j  via "i->j"  using s1
morphism b : j -> i  via "j->i"  using s2
"""
        ast = parse(src)
        assert ast.morphisms[0].semiring == 's1'
        assert ast.morphisms[1].semiring == 's2'

    def test_op_clause(self):
        src = _dsl('morphism a : x -> x  via "x->x"  op ops.dummy\n')
        ast = parse(src)
        assert ast.morphisms[0].op == 'ops.dummy'

    def test_bridge_clause(self):
        src = _dsl('morphism a : x -> x  via "x->x"  bridge  op ops.dummy\n')
        ast = parse(src)
        assert ast.morphisms[0].semiring is None

    def test_transform_clause(self):
        src = """
semiring s:
    contract = ops.dummy

sort i, j

morphism a : i -> j  via "ij,i->j"  transform ops.swap
"""
        ast = parse(src)
        assert ast.morphisms[0].transform == 'ops.swap'

    def test_compiler_on_semiring(self):
        src = """
semiring s:
    contract = ops.dummy
    compiler = ops.compiler

sort x

morphism a : x -> x  via "x->x"
"""
        ast = parse(src)
        assert ast.semirings[0].compiler == 'ops.compiler'

    def test_compiler_on_leg(self):
        src = _dsl('morphism a : x -> x  via "x->x"  compiler ops.compiler\n')
        ast = parse(src)
        assert ast.morphisms[0].compiler == 'ops.compiler'

    def test_unrecognised_clause_raises(self):
        src = _dsl('morphism a : x -> x  via "x->x"  badclause foo\n')
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
        src = _dsl('morphism a : x -> x  via "x->x"  using nonexistent\n')
        with pytest.raises(ValueError, match="undeclared semiring"):
            compile(src, NAMESPACE)

    def test_no_op_no_contract_raises(self):
        src = _dsl('morphism a : x -> x  via "x->x"\n', semiring=False)
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

morphism a : i -> j  via "ij,i->j"  transform ops.swap
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

morphism a : x -> x  via "abc->x"
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

morphism a : x -> x  via "abc"  compiler ops.compiler
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

morphism a : x -> x  via "x->x"
"""
        with pytest.raises(NameError, match="not found"):
            compile(src, NAMESPACE)

    def test_multi_error_accumulation(self):
        """Both a bad semiring ref and a bad path morphism ref appear in one ValueError."""
        src = _dsl(
            'morphism a : x -> x  via "x->x"  using nonexistent_semiring\n\n'
            'path p = does_not_exist\n'
        )
        with pytest.raises(ValueError) as exc_info:
            compile(src, NAMESPACE)
        msg = str(exc_info.value)
        assert "nonexistent_semiring" in msg
        assert "does_not_exist" in msg

    def test_phase3_multi_error_accumulation(self):
        """Two morphisms with bad op refs both appear in one ValueError."""
        src = _dsl(
            'morphism a : x -> x  via "x->x"  op ops.nonexistent_a\n'
            'morphism b : x -> x  via "x->x"  op ops.nonexistent_b\n'
        )
        with pytest.raises(ValueError) as exc_info:
            compile(src, NAMESPACE)
        msg = str(exc_info.value)
        assert "nonexistent_a" in msg
        assert "nonexistent_b" in msg


# ---------------------------------------------------------------------------
# TestFan
# ---------------------------------------------------------------------------

class TestFan:

    def test_parse_fan_default_merge(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b\n'
        )
        ast = parse(src)
        assert len(ast.fans) == 1
        assert ast.fans[0].name == 'ab'
        assert ast.fans[0].branches == ['a', 'b']
        assert ast.fans[0].merge == 'dict'

    def test_parse_fan_meet_merge(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b  merge meet\n'
        )
        ast = parse(src)
        assert ast.fans[0].merge == 'meet'

    def test_parse_fan_custom_merge(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n\n'
            'fan ab = a  merge ops.custom\n'
        )
        ast = parse(src)
        assert ast.fans[0].merge == 'ops.custom'

    def test_compile_fan_dict(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b\n'
        )
        arch = compile(src, NAMESPACE)
        result = arch.paths['ab']('hello', None, 0.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'a', 'b'}
        assert result['a'] == 'hello'

    def test_compile_fan_meet(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b  merge meet\n'
        )
        arch = compile(src, NAMESPACE)
        x = np.array([3.0, 1.0, 2.0])
        result = arch.paths['ab'](x, None, 0.0)
        np.testing.assert_array_equal(result, x)

    def test_compile_fan_join(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b  merge join\n'
        )
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

        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b  merge ops.sum_merge\n'
        )
        arch = compile(src, ns)
        result = arch.paths['ab'](5, None, 0.0)
        assert result == 10  # 5 + 5

    def test_fan_unknown_branch_raises(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n\n'
            'fan ab = a & nonexistent\n'
        )
        with pytest.raises(ValueError, match="not a declared morphism or path"):
            compile(src, NAMESPACE)

    def test_fan_no_branches_raises(self):
        with pytest.raises(SyntaxError):
            parse("fan empty =")

    def test_fan_three_branches(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n'
            'morphism c : x -> x  via "x->x"\n\n'
            'fan abc = a & b & c\n'
        )
        arch = compile(src, NAMESPACE)
        result = arch.paths['abc']('v', None, 0.0)
        assert len(result) == 3
        assert set(result.keys()) == {'a', 'b', 'c'}


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

morphism a : i -> j  via "i->j"  using join
morphism b : j -> j  via "j->j"  bridge  op ops.bridge
morphism c : j -> i  via "j->i"  using residuate

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

morphism a : i -> j  via "i->j"  using s1
morphism b : j -> i  via "j->i"  using s2

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

morphism a : i -> j  via "i->j"  using s1
morphism b : j -> k  via "j->k"  bridge  op ops.op
morphism c : i -> j  via "i->j"  using s2

path cross = a b c
"""
        with pytest.raises(ValueError, match="mismatch"):
            compile(src, ns)


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

morphism f : a -> a  via "a->a"

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


# ---------------------------------------------------------------------------
# TestAccumulateStateFeedback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestSortCoercion — T6: Graded Sort Compatibility
# ---------------------------------------------------------------------------

class TestSortCoercion:
    """Tests for the graded coercion system (T6).

    DSL syntax:
        coerce A -> B : 0.9        # directional, grade in [0.0, 1.0]
        sort_threshold 0.7         # grades below this warn; 0.0 always errors

    Rules:
        - No coercions declared  -> binary pass/fail (old behaviour)
        - A == B (exact match)   -> always pass (grade 1.0)
        - grade == 0.0 OR no declaration for pair -> ValueError
        - grade < threshold      -> warnings.warn (no error)
        - grade >= threshold     -> silent pass
        - Transitive closure     -> grade(A->C) = grade(A->B) * grade(B->C)
        - Direct declaration overrides derived grade
        - Coercions are directional (A->B != B->A)
    """

    # ------------------------------------------------------------------
    # Baseline — old behaviour preserved when no coerce declarations exist
    # ------------------------------------------------------------------

    def test_no_coercions_exact_match(self):
        """No coerce declarations. Matching sorts compile without error."""
        src = _dsl(
            'morphism f : x -> x  via "x->x"\n'
            'morphism g : x -> x  via "x->x"\n\n'
            'path p = f g\n'
        )
        arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    def test_no_coercions_mismatch_raises(self):
        """No coerce declarations. Sort mismatch raises ValueError."""
        src = _coerce_dsl("", 0.5)
        with pytest.raises(ValueError, match="mismatch"):
            compile(src, NAMESPACE)

    # ------------------------------------------------------------------
    # Grade at or above threshold — silent pass
    # ------------------------------------------------------------------

    def test_coerce_grade_above_threshold_passes(self):
        """grade 0.9 >= threshold 0.5 -> silent pass, no warning."""
        import warnings

        src = _coerce_dsl("coerce a -> b : 0.9", 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # any warning becomes an error here
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    # ------------------------------------------------------------------
    # Grade below threshold — warning but no error
    # ------------------------------------------------------------------

    def test_coerce_grade_below_threshold_warns(self):
        """grade 0.9 < threshold 0.95 -> warning issued, no TypeError."""
        src = _coerce_dsl("coerce a -> b : 0.9", 0.95)
        with pytest.warns(UserWarning, match=r"coercion grade .* < threshold"):
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    # ------------------------------------------------------------------
    # Grade exactly 0.0 — always a hard error regardless of threshold
    # ------------------------------------------------------------------

    def test_coerce_zero_grade_raises(self):
        """grade 0.0 is a hard error regardless of threshold."""
        src = _coerce_dsl("coerce a -> b : 0.0", 0.0)
        with pytest.raises(ValueError):
            compile(src, NAMESPACE)

    # ------------------------------------------------------------------
    # Undeclared pair — TypeError even when other coercions exist
    # ------------------------------------------------------------------

    def test_undeclared_pair_raises(self):
        """coerce A->B declared but path has C->D mismatch -> ValueError."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9", 0.5,
            sorts="a, b, c, d",
            morphisms='morphism f : c -> c  via "c->c"\nmorphism g : d -> d  via "d->d"\n\n',
        )
        with pytest.raises(ValueError):
            compile(src, NAMESPACE)

    # ------------------------------------------------------------------
    # Transitive closure — auto-derived grade
    # ------------------------------------------------------------------

    def test_transitive_closure_derives_grade(self):
        """A->B: 0.9, B->C: 0.8 -> auto-derived A->C: 0.72. threshold 0.5 -> passes."""
        import warnings

        src = _coerce_dsl(
            "coerce a -> b : 0.9\ncoerce b -> c : 0.8", 0.5,
            sorts="a, b, c",
            morphisms='morphism f : a -> a  via "a->a"\nmorphism g : c -> c  via "c->c"\n\n',
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    def test_transitive_closure_compounds_below_threshold(self):
        """A->B: 0.9, B->C: 0.8 -> derived A->C: 0.72. threshold 0.8 -> warning."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9\ncoerce b -> c : 0.8", 0.8,
            sorts="a, b, c",
            morphisms='morphism f : a -> a  via "a->a"\nmorphism g : c -> c  via "c->c"\n\n',
        )
        with pytest.warns(UserWarning, match=r"coercion grade .* < threshold"):
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    # ------------------------------------------------------------------
    # Direct declaration overrides transitive derivation
    # ------------------------------------------------------------------

    def test_direct_declaration_overrides_derived(self):
        """Direct coerce A->C: 0.5 overrides derived 0.72. threshold 0.6 -> warning."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9\ncoerce b -> c : 0.8\ncoerce a -> c : 0.5", 0.6,
            sorts="a, b, c",
            morphisms='morphism f : a -> a  via "a->a"\nmorphism g : c -> c  via "c->c"\n\n',
        )
        # grade 0.5 < threshold 0.6 -> must warn (not silently pass)
        with pytest.warns(UserWarning, match=r"coercion grade .* < threshold"):
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    # ------------------------------------------------------------------
    # Directionality — A->B does not imply B->A
    # ------------------------------------------------------------------

    def test_coercion_is_directional(self):
        """coerce A->B declared but B->A path -> ValueError."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9", 0.5,
            morphisms='morphism f : b -> b  via "b->b"\nmorphism g : a -> a  via "a->a"\n\n',
        )
        with pytest.raises(ValueError):
            compile(src, NAMESPACE)


class TestAccumulateStateFeedback:
    """P1: accumulated values are merged back into state after each step.

    Tests use Interpreter directly (engine.functor) with hand-built functors
    and cells, passing accumulate_specs explicitly. This isolates the P1
    state-merge behavior from DSL compilation concerns.
    """

    def _make_interp(self, cell):
        """Build a minimal single-case Interpreter for coalgebra runs."""
        from engine.functor import Functor, Case, Interpreter
        functor = Functor([Case(name='step', recursive=1, data=1, output=1)])
        return Interpreter(functor=functor, cell=cell, params={}, temp=0.0)

    def test_accumulate_visible_in_next_step(self):
        """At step 2, params['accumulated'] contains the payload from step 1."""
        from engine.functor import UnfoldStep

        seen_accumulated = []

        def cell(state, token, params, temp):
            # Record what accumulated looks like at call time
            seen_accumulated.append(dict(params.get('accumulated', {})))
            payload = token  # np.array([...])
            next_state = dict(state) if isinstance(state, dict) else state
            return UnfoldStep('step', [payload], [next_state], output=payload)

        interp = self._make_interp(cell)
        tokens = [np.array([1.0]), np.array([2.0])]
        accumulate_specs = {'key': ('cat', None)}

        outputs, final = interp.run_coalgebra(
            {}, token_iter=tokens, accumulate_specs=accumulate_specs
        )

        assert len(outputs) == 2
        # Step 1 sees empty accumulated
        assert seen_accumulated[0] == {}
        # Step 2 sees the result from step 1 merged in
        assert 'key' in seen_accumulated[1]
        assert np.allclose(seen_accumulated[1]['key'], np.array([1.0]))

    def test_accumulate_grows_across_steps(self):
        """accumulated['key'] concatenates across 3 steps: length 1, 2, 3."""
        from engine.functor import UnfoldStep

        acc_sizes = []

        def cell(state, token, params, temp):
            acc = params.get('accumulated', {})
            key_val = acc.get('key')
            acc_sizes.append(len(key_val) if key_val is not None else 0)
            payload = token
            next_state = dict(state) if isinstance(state, dict) else state
            return UnfoldStep('step', [payload], [next_state], output=payload)

        interp = self._make_interp(cell)
        tokens = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        accumulate_specs = {'key': ('cat', None)}

        outputs, final = interp.run_coalgebra(
            {}, token_iter=tokens, accumulate_specs=accumulate_specs
        )

        assert len(outputs) == 3
        # At step N the cell sees N-1 accumulated items (previous steps)
        assert acc_sizes == [0, 1, 2]
        # Final state carries all 3 accumulated values
        assert 'key' in final
        assert len(final['key']) == 3
        assert np.allclose(final['key'], np.array([1.0, 2.0, 3.0]))

    def test_no_accumulate_state_unchanged(self):
        """Without accumulate_specs, no spurious keys appear in state across steps."""
        from engine.functor import UnfoldStep

        state_keys_per_step = []

        def cell(state, token, params, temp):
            state_keys_per_step.append(set(state.keys()) if isinstance(state, dict) else None)
            next_state = dict(state) if isinstance(state, dict) else state
            return UnfoldStep('step', [token], [next_state], output=token)

        interp = self._make_interp(cell)
        tokens = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

        outputs, final = interp.run_coalgebra(
            {'pos': 0}, token_iter=tokens
            # No accumulate_specs
        )

        assert len(outputs) == 3
        # State keys must be identical at every step — no accumulate injection
        for keys in state_keys_per_step:
            assert keys == {'pos'}, f"unexpected keys: {keys}"
        assert set(final.keys()) == {'pos'}


# ---------------------------------------------------------------------------
# TestBackend
# ---------------------------------------------------------------------------

class TestBackend:
    """Verify that compile(..., backend=NUMPY_BACKEND) is identical to compile(...)."""

    def test_backend_torch_explicit(self):
        torch = pytest.importorskip('torch')
        from engine.runtime import Backend

        TORCH_BACKEND = Backend(
            minimum=torch.minimum,
            maximum=torch.maximum,
            concatenate=lambda arrays, axis=0: torch.cat(arrays, dim=axis),
            abs=torch.abs,
            max=lambda x: torch.max(x).item(),
        )

        dsl = """
sort a
morphism f : a -> a  via "i->i"  op ops.f  arity unary
morphism g : a -> a  via "i->i"  op ops.g  arity unary
fan fg = f & g  merge join
"""

        class Ops:
            def f(self, eq, x, y=None, temp=0.0):
                return x * 2.0  # x is a torch tensor; scalar multiply preserves type
            def g(self, eq, x, y=None, temp=0.0):
                return x * 3.0

        source = parse(dsl)
        ad = compile(source, {'ops': Ops()}, backend=TORCH_BACKEND)

        x = torch.tensor([1.0, 2.0, 3.0])
        result = ad.paths['fg'](x, {}, temp=0.0)
        # join merge uses backend.maximum — result must be a torch Tensor
        assert isinstance(result, torch.Tensor), f"expected Tensor, got {type(result)}"
        expected = torch.tensor([3.0, 6.0, 9.0])  # max(2x, 3x) = 3x
        assert torch.allclose(result, expected)



