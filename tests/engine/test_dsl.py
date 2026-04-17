"""Tests for engine — parse + compile."""

import warnings
import numpy as np
import pytest
from engine import parse, compile, DSLSource, ArchDef
from engine.compiler import _tterm_fields, _str_val, _int_val, _str_list_val, _bool_val, _opt_str_val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mf(m) -> dict:
    """Extract morphism fields as Python dict with camelCase → snake_case mapping."""
    tf = _tterm_fields(m)
    return {
        'name':            _str_val(tf['name']),
        'src_sort':        _str_val(tf['srcSort']),
        'tgt_sort':        _str_val(tf['tgtSort']),
        'arity':           _str_val(tf['arity']),
        'semiring':        _str_val(tf['semiring']) or None,
        'equation':        _str_val(tf['equation']),
        'op':              _str_val(tf['op']) or None,
        'transform':       _str_val(tf['transform']) or None,
        'compiler':        _str_val(tf['compiler']) or None,
        'accumulate':      _str_val(tf['accumulate']) or None,
        'accumulate_fields': _str_list_val(tf['accumulateFields']) or None,
        'template_param':  _str_val(tf['templateParam']) or None,
    }


def _pf(p) -> dict:
    """Extract path fields."""
    tf = _tterm_fields(p)
    return {
        'name':      _str_val(tf['name']),
        'morphisms': _str_list_val(tf['morphisms']),
        'residual':  _bool_val(tf['residual']),
        'normed':    _str_val(tf['normed']) or None,
    }


def _ff(f) -> dict:
    """Extract fan fields."""
    tf = _tterm_fields(f)
    return {
        'name':     _str_val(tf['name']),
        'branches': _str_list_val(tf['branches']),
        'merge':    _str_val(tf['merge']) or 'dict',
    }


def _af(a) -> dict:
    """Extract arch fields (top-level only, not case internals)."""
    from hydra.core import TermList
    tf = _tterm_fields(a)
    cases_term = tf['cases']
    case_list = list(cases_term.value) if isinstance(cases_term, TermList) else []
    return {
        'name':                _str_val(tf['name']),
        'cases':               [_cf(c) for c in case_list],
        'step_enter':          _opt_str_val(tf['stepEnter']),
        'step_emit':           _opt_str_val(tf['stepEmit']),
        'algebra_cell':        _str_val(tf['algebraCell']) or None,
        'observer_convergence': _str_val(tf['observerConvergence']) or None,
        'observer_loss':       _str_val(tf['observerLoss']) or None,
        'step_compute':        _str_val(tf['stepCompute']) or None,
        'state_fields':        dict(zip(_str_list_val(tf['stateFieldNames']),
                                        _str_list_val(tf['stateFieldTypes'])))
                               if _str_list_val(tf['stateFieldNames']) else None,
    }


def _cf(c) -> dict:
    """Extract case fields from a bare Term (as stored inside arch.cases list)."""
    from hydra.core import TermRecord, Record
    if isinstance(c, TermRecord):
        rec = c.value
    elif isinstance(c, Record):
        rec = c
    else:
        raise TypeError(f"_cf: expected TermRecord or Record, got {type(c)}")
    tf = {f.name.value: f.term for f in rec.fields}
    return {
        'name':      _str_val(tf['name']),
        'recursive': _int_val(tf['recursive']),
        'data':      _int_val(tf['data']),
        'output':    _int_val(tf['output']),
        'cell':      _str_val(tf['cell']) or None,
        'morphisms': _str_list_val(tf['caseMorphisms']) or None,
        'iterate':   _str_val(tf['iterate']) or None,
    }


def _dummy_op(eq, x, y=None):
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

# Shared DSL for iterate combinator tests — minimal arch with sort + morphism stubs.
_ITERATE_PREAMBLE = (
    "sort x\n\n"
    'morphism m : x -> x  "x->x"  ops.dummy\n\n'
)

# Shared DSL skeleton for parameterized morphism tests.
_PARAM_PREAMBLE = "sort x\n\n"

def _make_ns(**extra_ops):
    return {
        'ops': type('ns', (), {
            'dummy': staticmethod(_dummy_op),
            **{k: staticmethod(v) for k, v in extra_ops.items()},
        })(),
    }

_iterate_ns = _make_ns
_param_ns = _make_ns


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
        realize = _mf(ast.morphisms[0])
        assert realize['name'] == 'realize'
        assert realize['src_sort'] == 'j'
        assert realize['tgt_sort'] == 'i'
        assert realize['equation'] == 'j,ji->i'
        assert _pf(ast.paths[0])['name'] == 'attend'
        assert _pf(ast.paths[0])['morphisms'] == ['realize', 'propagate']

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
        with pytest.raises(SyntaxError, match="declare 'contract'"):
            parse("semiring s:\n    # no contract\n\nsort x")

    def test_using_clause(self):
        src = (
            "semiring s1:\n    contract = ops.dummy\n\n"
            "semiring s2:\n    contract = ops.dummy\n\n"
            "sort i, j\n\n"
            "morphism a : i -> j  via \"i->j\"  using s1\n"
            "morphism b : j -> i  via \"j->i\"  using s2\n"
        )
        ast = parse(src)
        assert _mf(ast.morphisms[0])['semiring'] == 's1'
        assert _mf(ast.morphisms[1])['semiring'] == 's2'

    @pytest.mark.parametrize("src,obj,attr,expected", [
        (_dsl('morphism a : x -> x  via "x->x"  op ops.dummy\n'),
         'morphisms', 'op', 'ops.dummy'),
        (_dsl('morphism a : x -> x  via "x->x"  bridge  op ops.dummy\n'),
         'morphisms', 'semiring', None),
        (_dsl('morphism a : x -> x  via "x->x"  transform ops.swap\n'),
         'morphisms', 'transform', 'ops.swap'),
        (_dsl('morphism a : x -> x  via "x->x"  compiler ops.compiler\n'),
         'morphisms', 'compiler', 'ops.compiler'),
        ("semiring s:\n    contract = ops.dummy\n    compiler = ops.compiler\n\n"
         "sort x\n\nmorphism a : x -> x  via \"x->x\"\n",
         'semirings', 'compiler', 'ops.compiler'),
    ])
    def test_morphism_clause_parsed(self, src, obj, attr, expected):
        ast = parse(src)
        node = getattr(ast, obj)[0]
        if obj == 'morphisms':
            assert _mf(node)[attr] == expected
        else:
            assert getattr(node, attr) == expected

    def test_unrecognised_clause_raises(self):
        src = _dsl('morphism a : x -> x  via "x->x"  badclause foo\n')
        with pytest.raises(SyntaxError, match="unrecognised clause"):
            parse(src)

    def test_auto_assign_single_semiring(self):
        """Legs without 'using' auto-assign when exactly one semiring exists."""
        ast = parse(BASIC_SOURCE)
        assert _mf(ast.morphisms[0])['semiring'] == '_default'
        assert _mf(ast.morphisms[1])['semiring'] == '_default'


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
        assert arch.morphism_semiring['realize'] == 'join'
        assert arch.morphism_semiring['propagate'] == 'join'
        assert callable(arch.paths['realize'])
        assert arch.paths['realize']('x', 'y') == 'x'
        text = arch.explain('attend')
        assert 'realize' in text
        assert 'propagate' in text

    def test_undeclared_semiring_raises(self):
        src = _dsl('morphism a : x -> x  via "x->x"  using nonexistent\n')
        with pytest.raises(ValueError, match="undeclared semiring"):
            compile(src, NAMESPACE)

    def test_no_op_no_contract_raises(self):
        src = _dsl('morphism a : x -> x  via "x->x"\n', semiring=False)
        ast = parse(src)
        assert len(ast.semirings) == 0
        with pytest.raises(ValueError, match="no 'op' clause"):
            compile(ast, NAMESPACE)

    def test_transform_resolved(self):
        src = _dsl('morphism a : x -> x  via "x->x"  transform ops.swap\n')
        arch = compile(src, NAMESPACE)
        result = arch.paths['a']('x', 'y')
        assert result == 'y'

    def test_compiler_semiring_level(self):
        src = ("semiring s:\n    contract = ops.dummy\n    compiler = ops.compiler\n\n"
               "sort x\n\nmorphism a : x -> x  via \"abc->x\"\n")
        arch = compile(src, NAMESPACE)
        text = arch.explain('a')
        assert 'abc->x' in text

    def test_name_resolution_error(self):
        src = _dsl('morphism a : x -> x  via "x->x"\n').replace(
            'contract = ops.dummy', 'contract = nonexistent.fn'
        )
        with pytest.raises((NameError, ValueError), match="not found"):
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

@pytest.mark.parametrize("merge_kw,expected_merge", [
    ("",                 "dict"),
    ("  merge meet",     "meet"),
    ("  merge ops.custom", "ops.custom"),
])
def test_parse_fan_merge(merge_kw, expected_merge):
    src = _dsl(
        'morphism a : x -> x  via "x->x"\n\n'
        f'fan ab = a{merge_kw}\n'
    )
    ast = parse(src)
    assert _ff(ast.fans[0])['name'] == 'ab'
    assert _ff(ast.fans[0])['merge'] == expected_merge


class TestFan:

    def test_compile_fan_dict(self):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b\n'
        )
        arch = compile(src, NAMESPACE)
        result = arch.paths['ab']('hello', None)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'a', 'b'}
        assert result['a'] == 'hello'

    @pytest.mark.parametrize("merge,op", [
        ("meet", np.minimum),
        ("join", np.maximum),
    ])
    def test_compile_fan_elementwise_merge(self, merge, op):
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            f'fan ab = a & b  merge {merge}\n'
        )
        arch = compile(src, NAMESPACE)
        x = np.array([3.0, 1.0, 2.0])
        result = arch.paths['ab'](x, None)
        np.testing.assert_array_equal(result, op(x, x))

    def test_compile_fan_custom_merge(self):
        ns = _make_ns(sum_merge=lambda results: sum(results.values()))
        src = _dsl(
            'morphism a : x -> x  via "x->x"\n'
            'morphism b : x -> x  via "x->x"\n\n'
            'fan ab = a & b  merge ops.sum_merge\n'
        )
        assert compile(src, ns).paths['ab'](5, None) == 10  # 5 + 5

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


# ---------------------------------------------------------------------------
# TestBridge
# ---------------------------------------------------------------------------

def _make_bridge_ns():
    def _join(eq, x, y=None):
        return f'J({x})'
    def _res(eq, x, y=None):
        return f'R({x})'
    def _bridge(eq, x, y=None):
        return f'B({x})'
    return {
        'ops': type('ns', (), {
            'join': staticmethod(_join),
            'res': staticmethod(_res),
            'bridge': staticmethod(_bridge),
        })(),
    }

_CROSS_SEMIRING_SRC = """
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

class TestBridge:

    def test_cross_semiring_with_bridge(self):
        arch = compile(_CROSS_SEMIRING_SRC, _make_bridge_ns())
        result = arch.paths['cross']('x', None)
        assert result == 'R(B(J(x)))'

    def test_cross_semiring_without_bridge_raises(self):
        src = (
            "semiring s1:\n    contract = ops.dummy\n\n"
            "semiring s2:\n    contract = ops.dummy\n\n"
            "sort i, j\n\n"
            "morphism a : i -> j  via \"i->j\"  using s1\n"
            "morphism b : j -> i  via \"j->i\"  using s2\n\n"
            "path cross = a b\n"
        )
        with pytest.raises(ValueError, match="spans multiple semirings"):
            compile(src, NAMESPACE)

    def test_cross_semiring_sort_mismatch_raises(self):
        src = (
            "semiring s1:\n    contract = ops.dummy\n\n"
            "semiring s2:\n    contract = ops.dummy\n\n"
            "sort i, j, k\n\n"
            "morphism a : i -> j  via \"i->j\"  using s1\n"
            "morphism b : j -> k  via \"j->k\"  bridge  op ops.dummy\n"
            "morphism c : i -> j  via \"i->j\"  using s2\n\n"
            "path cross = a b c\n"
        )
        with pytest.raises(ValueError, match="mismatch"):
            compile(src, NAMESPACE)


# ---------------------------------------------------------------------------
# TestAugment
# ---------------------------------------------------------------------------

class TestAugment:
    def test_parse_augment_in_path(self):
        """Bracketed tokens are preserved in path morphisms list."""
        src = _dsl(
            'morphism f : x -> x  via "x->x"\n\n'
            'fan kv = f & f\n\n'
            'path p = f [kv] f\n'
        )
        ast = parse(src)
        assert '[kv]' in _pf(ast.paths[0])['morphisms']

    def test_compile_augment(self):
        """Augment step merges fan output into y."""
        ns = _make_ns(
            branch=lambda eq, x, y=None: x * 2,
            reader=lambda eq, x, y=None: y.get('b1', -1),
        )
        src = _dsl(
            'morphism f : x -> x  via "x->x"\n'
            'morphism b1 : x -> x  via "x->x"  op ops.branch\n'
            'morphism b2 : x -> x  via "x->x"  op ops.branch\n'
            'morphism r : x -> x  via "x->x"  op ops.reader\n\n'
            'fan kv = b1 & b2\n\n'
            'path p = f [kv] r\n'
        )
        arch = compile(src, ns)
        result = arch.paths['p'](np.array([1.0]), {})
        assert result != -1  # b1 key should exist in augmented y

    def test_augment_preserves_x(self):
        """Augment step does not modify x."""
        ns = _make_ns(double=lambda eq, x, y=None: x * 2)
        src = _dsl(
            'morphism pre  : x -> x  via "x->x"  op ops.double\n'
            'morphism b1   : x -> x  via "x->x"\n'
            'morphism post : x -> x  via "x->x"  op ops.double\n\n'
            'fan aug = b1\n\n'
            'path p = pre [aug] post\n'
        )
        arch = compile(src, ns)
        x = np.array([1.0])
        result = arch.paths['p'](x, {})
        # pre doubles: 2.0, augment doesn't change x, post doubles: 4.0
        assert float(result[0]) == 4.0

    def test_augment_with_residual(self):
        """Augment works with residual combinator."""
        ns = _make_ns(op=lambda eq, x, y=None: x + 1.0)
        src = _dsl(
            'morphism f : x -> x  via "x->x"  op ops.op\n'
            'morphism b : x -> x  via "x->x"  op ops.op\n\n'
            'fan aug = b\n\n'
            'path p = f [aug] f  residual\n'
        )
        arch = compile(src, ns)
        result = arch.paths['p'](np.array([0.0]), {})
        # f(0) = 1, augment no change to x, f(1) = 2, residual: 2 + 0 = 2
        assert float(result[0]) == 2.0


# ---------------------------------------------------------------------------
# TestIterateCombinator
# ---------------------------------------------------------------------------

class TestIterateCombinator:

    def test_parse_iterate_attribute(self):
        """iterate=layers parsed on case declarations; multiple cases can share a group."""
        src = """
arch A:
    algebra:
        input: leaf  data=1  cell=identity
        attn:  node  data=1  cell=identity  iterate=layers
        ffn:   node  data=1  cell=identity  iterate=layers
        final: node  data=1  cell=identity
"""
        ast = parse(src)
        cases = _af(ast.archs[0])['cases']
        assert cases[0]['iterate'] is None
        assert cases[1]['iterate'] == 'layers'
        assert cases[2]['iterate'] == 'layers'
        assert cases[3]['iterate'] is None

    @pytest.mark.parametrize("layers,expected", [
        ([10, 20, 30], 65),   # ((5+10)+20)+30
        ([10, 20],     35),   # (5+10)+20
        ([],           5),    # no layers, identity
    ])
    def test_iterate_single_case_fold(self, layers, expected):
        """Single iterate case folds over layers."""
        def _add(payload, child_results, params):
            return child_results[0] + payload[0]

        ns = _iterate_ns(add=_add)
        src = _ITERATE_PREAMBLE + (
            "arch Net:\n"
            "    algebra:\n"
            "        input: leaf  data=1  cell=identity\n"
            "        block:  node  data=1  cell=ops.add  iterate=layers\n"
        )
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        result = interp.run_algebra_layers(5, layers=layers)
        assert result == expected

    @pytest.mark.parametrize("base,layers,expected", [
        (5,  [10, 20], 70),   # (5+10+20)*2 = 70
        (10, [],       20),   # base=10, no layers, double=20
    ])
    def test_iterate_with_epilogue(self, base, layers, expected):
        """Cases after iterate block are applied as epilogue; empty layers skips to epilogue."""
        def _add(payload, child_results, params):
            return child_results[0] + payload[0]

        def _double(payload, child_results, params):
            return child_results[0] * 2

        ns = _iterate_ns(add=_add, double=_double)
        src = _ITERATE_PREAMBLE + (
            "arch Net:\n"
            "    algebra:\n"
            "        input: leaf  data=1  cell=identity\n"
            "        block:  node  data=1  cell=ops.add    iterate=layers\n"
            "        final: node  data=1  cell=ops.double\n"
        )
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        result = interp.run_algebra_layers(base, layers=layers)
        assert result == expected

    def test_iterate_two_case_group(self):
        """Two iterate cases form a block that repeats per layer."""
        def _add(payload, child_results, params):
            return child_results[0] + payload[0]

        def _mul(payload, child_results, params):
            return child_results[0] * 2

        ns = _iterate_ns(add=_add, mul=_mul)
        src = _ITERATE_PREAMBLE + (
            "arch Net:\n"
            "    algebra:\n"
            "        input: leaf  data=1  cell=identity\n"
            "        step1: node  data=1  cell=ops.add  iterate=layers\n"
            "        step2: node  data=1  cell=ops.mul  iterate=layers\n"
        )
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        result = interp.run_algebra_layers(5, layers=[10, 0])
        assert result == 40

    def test_iterate_with_extras(self):
        """extras dict is merged into each layer payload."""
        def _reader(payload, child_results, params):
            return child_results[0] + payload[0].get('extra_val', 0)

        ns = _iterate_ns(reader=_reader)
        src = _ITERATE_PREAMBLE + (
            "arch Net:\n"
            "    algebra:\n"
            "        input: leaf  data=1  cell=identity\n"
            "        block:  node  data=1  cell=ops.reader  iterate=layers\n"
        )
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        result = interp.run_algebra_layers(0, layers=[{}, {}], extras={'extra_val': 100})
        assert result == 200

    def test_iterate_backward_compat_decompose(self):
        """Old decompose= calling convention still works."""
        def _leaf(payload, child_results, params):
            return payload[0]

        ns = _iterate_ns(leaf=_leaf)
        src = _ITERATE_PREAMBLE + (
            "arch Net:\n"
            "    algebra:\n"
            "        input: leaf  data=1  cell=ops.leaf\n"
        )
        arch = compile(src, ns)
        interp = arch.interpreter('Net')
        tree = ('input', [42], [])
        result = interp.run_algebra(tree, lambda n: n)
        assert result == 42

    def test_iterate_no_groups_with_layers_raises(self):
        """Passing layers= when no iterate cases exist raises ValueError."""
        src = "arch Net:\n    algebra:\n        input: leaf  data=1  cell=identity\n"
        arch = compile(src, _iterate_ns())
        interp = arch.interpreter('Net')
        with pytest.raises(ValueError, match="iterate"):
            interp.run_algebra_layers(5, layers=[1, 2])


# ---------------------------------------------------------------------------
# TestParameterizedMorphisms
# ---------------------------------------------------------------------------

class TestParameterizedMorphisms:

    @pytest.mark.parametrize("src_suffix,attr,expected", [
        ('morphism ln[prefix] : x -> x  "x->x"  ops.dummy\n', 'template_param', 'prefix'),
        ('morphism m : x -> x  "x->x"  ops.dummy\n',          'template_param', None),
    ])
    def test_parse_template_param(self, src_suffix, attr, expected):
        ast = parse(_PARAM_PREAMBLE + src_suffix)
        assert _mf(ast.morphisms[0])[attr] == expected

    def test_parse_instantiation_in_path(self):
        """ln[ln1] appears as a token in path morphisms."""
        src = (
            _PARAM_PREAMBLE
            + 'morphism ln[prefix] : x -> x  "x->x"  ops.dummy\n'
            + 'morphism m : x -> x  "x->x"  ops.dummy\n\n'
            + 'path p = ln[ln1] m\n'
        )
        ast = parse(src)
        assert _pf(ast.paths[0])['morphisms'] == ['ln[ln1]', 'm']

    def test_compile_template_instantiation(self):
        """Template instantiation creates a curried callable."""
        ns = _param_ns(param_op=lambda eq, x, y=None, prefix='default': f'{prefix}({x})')
        src = (
            _PARAM_PREAMBLE
            + 'morphism t[prefix] : x -> x  "x->x"  ops.param_op\n\n'
            + 'path p1 = t[alpha]\n'
            + 'path p2 = t[beta]\n'
        )
        arch = compile(src, ns)
        assert arch.paths['p1']('data', None) == 'alpha(data)'
        assert arch.paths['p2']('data', None) == 'beta(data)'

    def test_template_with_regular_morphisms(self):
        """Template and regular morphisms compose in a path."""
        ns = _param_ns(tag=lambda eq, x, y=None, label='?': f'{label}:{x}')
        src = (
            _PARAM_PREAMBLE
            + 'morphism t[label] : x -> x  "x->x"  ops.tag\n'
            + 'morphism pass : x -> x  "x->x"  ops.dummy\n\n'
            + 'path p = t[hello] pass\n'
        )
        assert compile(src, ns).paths['p']('world', None) == 'hello:world'

    def test_template_multiline_syntax(self):
        """Template morphism with multi-line declaration."""
        src = (
            _PARAM_PREAMBLE
            + "morphism t[key] : x -> x\n    \"x->x\"\n    ops.dummy\n    arity unary\n"
        )
        ast = parse(src)
        assert _mf(ast.morphisms[0])['template_param'] == 'key'
        assert _mf(ast.morphisms[0])['arity'] == 'unary'

    def test_template_not_confused_with_augment(self):
        """ln[ln1] (template inst) and [kv] (augment) coexist in a path."""
        ns = _param_ns(tag=lambda eq, x, y=None, label='?': x)
        src = (
            _PARAM_PREAMBLE
            + 'morphism t[label] : x -> x  "x->x"  ops.tag\n'
            + 'morphism b : x -> x  "x->x"  ops.dummy\n\n'
            + 'fan kv = b\n\n'
            + 'path p = t[pre] [kv] b\n'
        )
        assert 'p' in compile(src, ns).paths


# ---------------------------------------------------------------------------
# TestStateDeclaration
# ---------------------------------------------------------------------------

_STATE_ARCH_BASE = """
semiring s:
    contract = ops.dummy
sort a
morphism m : a -> a via "x"
arch T:
    cases:
        input: leaf data=1 cell=identity
"""

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
        assert _af(ast.archs[0])['state_fields'] == {'pos': 'int', 'caches': 'list'}

    def test_state_compiles_to_hydra_type(self):
        src = _STATE_ARCH_BASE + "    state:\n        pos: int\n        caches: list\n"
        arch = compile(src, dict(NAMESPACE))
        assert arch._archs['T'].state_type is not None

    def test_no_state_block_is_none(self):
        arch = compile(_STATE_ARCH_BASE, dict(NAMESPACE))
        assert arch._archs['T'].state_type is None


# ---------------------------------------------------------------------------
# TestStepProtocol
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step_body,enter,emit", [
    ("    step:\n        enter = embed\n        emit = unembed\n", "embed", "unembed"),
    ("    step:\n        enter = embed\n",                          "embed", None),
    ("",                                                            None,    None),
])
def test_parse_step_protocol(step_body, enter, emit):
    src = (
        "arch T:\n"
        "    cases:\n"
        "        input: leaf data=1 cell=identity\n"
        "        block: node data=1 cell=identity\n"
        + step_body
    )
    ast = parse(src)
    arch_fields = _af(ast.archs[0])
    assert arch_fields['step_enter'] == enter
    assert arch_fields['step_emit'] == emit


# ---------------------------------------------------------------------------
# TestSortCoercion — T6: Graded Sort Compatibility
# ---------------------------------------------------------------------------

class TestSortCoercion:
    """Tests for the graded coercion system (T6)."""

    def test_no_coercions_mismatch_raises(self):
        """No coerce declarations. Sort mismatch raises ValueError."""
        src = _coerce_dsl("", 0.5)
        with pytest.raises(ValueError, match="mismatch"):
            compile(src, NAMESPACE)

    def test_coerce_grade_above_threshold_passes(self):
        """grade 0.9 >= threshold 0.5 -> silent pass, no warning."""
        src = _coerce_dsl("coerce a -> b : 0.9", 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    def test_coerce_grade_below_threshold_warns(self):
        """grade 0.9 < threshold 0.95 -> warning issued, no TypeError."""
        src = _coerce_dsl("coerce a -> b : 0.9", 0.95)
        with pytest.warns(UserWarning, match=r"coercion grade .* < threshold"):
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    def test_coerce_zero_grade_raises(self):
        """grade 0.0 is a hard error regardless of threshold."""
        src = _coerce_dsl("coerce a -> b : 0.0", 0.0)
        with pytest.raises(ValueError):
            compile(src, NAMESPACE)

    def test_undeclared_pair_raises(self):
        """coerce A->B declared but path has C->D mismatch -> ValueError."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9", 0.5,
            sorts="a, b, c, d",
            morphisms='morphism f : c -> c  via "c->c"\nmorphism g : d -> d  via "d->d"\n\n',
        )
        with pytest.raises(ValueError):
            compile(src, NAMESPACE)

    @pytest.mark.parametrize("threshold,warns", [
        (0.5,  False),   # derived grade 0.72 >= 0.5 -> silent
        (0.8,  True),    # derived grade 0.72 < 0.8  -> warn
    ])
    def test_transitive_closure(self, threshold, warns):
        """A->B: 0.9, B->C: 0.8 -> auto-derived A->C: 0.72."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9\ncoerce b -> c : 0.8", threshold,
            sorts="a, b, c",
            morphisms='morphism f : a -> a  via "a->a"\nmorphism g : c -> c  via "c->c"\n\n',
        )
        if warns:
            with pytest.warns(UserWarning, match=r"coercion grade .* < threshold"):
                arch = compile(src, NAMESPACE)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    def test_direct_declaration_overrides_derived(self):
        """Direct coerce A->C: 0.5 overrides derived 0.72. threshold 0.6 -> warning."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9\ncoerce b -> c : 0.8\ncoerce a -> c : 0.5", 0.6,
            sorts="a, b, c",
            morphisms='morphism f : a -> a  via "a->a"\nmorphism g : c -> c  via "c->c"\n\n',
        )
        with pytest.warns(UserWarning, match=r"coercion grade .* < threshold"):
            arch = compile(src, NAMESPACE)
        assert 'p' in arch.paths

    def test_coercion_is_directional(self):
        """coerce A->B declared but B->A path -> ValueError."""
        src = _coerce_dsl(
            "coerce a -> b : 0.9", 0.5,
            morphisms='morphism f : b -> b  via "b->b"\nmorphism g : a -> a  via "a->a"\n\n',
        )
        with pytest.raises(ValueError):
            compile(src, NAMESPACE)


class TestAccumulateStateFeedback:
    """P1: accumulated values are merged back into state after each step."""

    def _make_interp(self, cell):
        """Build a minimal single-case Interpreter for coalgebra runs."""
        from engine.functor import Functor, Case, Interpreter
        functor = Functor([Case(name='step', recursive=1, data=1, output=1)])
        return Interpreter(functor=functor, cell=cell, params={}, temp=0.0)

    def test_accumulate_visible_in_next_step(self):
        """At step 2, params['accumulated'] contains the payload from step 1."""
        from engine.functor import UnfoldStep

        seen_accumulated = []

        def cell(state, token, params):
            seen_accumulated.append(dict(params.get('accumulated', {})))
            payload = token
            next_state = dict(state) if isinstance(state, dict) else state
            return UnfoldStep('step', [payload], [next_state], output=payload)

        interp = self._make_interp(cell)
        tokens = [np.array([1.0]), np.array([2.0])]
        accumulate_specs = {'key': ('cat', None)}

        outputs, final = interp.run_coalgebra(
            {}, token_iter=tokens, accumulate_specs=accumulate_specs
        )

        assert len(outputs) == 2
        assert seen_accumulated[0] == {}
        assert 'key' in seen_accumulated[1]
        assert np.allclose(seen_accumulated[1]['key'], np.array([1.0]))

    def test_accumulate_grows_across_steps(self):
        """accumulated['key'] concatenates across 3 steps: length 1, 2, 3."""
        from engine.functor import UnfoldStep

        acc_sizes = []

        def cell(state, token, params):
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
        assert acc_sizes == [0, 1, 2]
        assert 'key' in final
        assert len(final['key']) == 3
        assert np.allclose(final['key'], np.array([1.0, 2.0, 3.0]))

    def test_no_accumulate_state_unchanged(self):
        """Without accumulate_specs, no spurious keys appear in state across steps."""
        from engine.functor import UnfoldStep

        state_keys_per_step = []

        def cell(state, token, params):
            state_keys_per_step.append(set(state.keys()) if isinstance(state, dict) else None)
            next_state = dict(state) if isinstance(state, dict) else state
            return UnfoldStep('step', [token], [next_state], output=token)

        interp = self._make_interp(cell)
        tokens = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

        outputs, final = interp.run_coalgebra(
            {'pos': 0}, token_iter=tokens
        )

        assert len(outputs) == 3
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
            add=torch.add,
            multiply=torch.mul,
            sum=lambda x, axis=None: torch.sum(x, dim=axis) if axis is not None else torch.sum(x),
            min=lambda x: torch.min(x).item(),
        )

        dsl = """
sort a
morphism f : a -> a  via "i->i"  op ops.f  arity unary
morphism g : a -> a  via "i->i"  op ops.g  arity unary
fan fg = f & g  merge join
"""

        class Ops:
            def f(self, x):
                return x * 2.0
            def g(self, x):
                return x * 3.0

        source = parse(dsl)
        ad = compile(source, {'ops': Ops()}, backend=TORCH_BACKEND)

        x = torch.tensor([1.0, 2.0, 3.0])
        result = ad.paths['fg'](x, {})
        assert isinstance(result, torch.Tensor), f"expected Tensor, got {type(result)}"
        expected = torch.tensor([3.0, 6.0, 9.0])
        assert torch.allclose(result, expected)
