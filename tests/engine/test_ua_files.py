"""Tests for .ua file loading."""

import types as pytypes
from pathlib import Path

import pytest
import engine

_FIXTURE = Path(__file__).parent / "fixtures" / "test_arch.ua"

_DSL_INLINE = _FIXTURE.read_text()


def _make_namespace():
    ops = pytypes.SimpleNamespace(
        join_fn=lambda eq, x, y, temp=0.0: x,
        compile_einsum=lambda eq: eq,
    )
    return {'ops': ops}


class TestCompileFromPath:
    def test_compile_path_object(self):
        arch = engine.compile(_FIXTURE, _make_namespace())
        assert "read" in arch.paths
        assert "attn" in arch.paths
        assert "Transformer" in arch._arch_terms

    def test_compile_string_path(self):
        arch = engine.compile(str(_FIXTURE), _make_namespace())
        assert "read" in arch.paths

    def test_load_function(self):
        arch = engine.load(_FIXTURE, _make_namespace())
        assert "read" in arch.paths
        assert "kv" in arch._fan_terms

    def test_load_string_path(self):
        arch = engine.load(str(_FIXTURE), _make_namespace())
        assert "read" in arch.paths

    def test_equivalent_to_inline(self):
        arch_file = engine.compile(_FIXTURE, _make_namespace())
        arch_inline = engine.compile(_DSL_INLINE, _make_namespace())
        assert set(arch_file.paths.keys()) == set(arch_inline.paths.keys())
        assert set(arch_file._morphism_terms.keys()) == set(arch_inline._morphism_terms.keys())
        assert set(arch_file._path_terms.keys()) == set(arch_inline._path_terms.keys())

    def test_module_from_file(self):
        arch = engine.load(_FIXTURE, _make_namespace())
        mod = arch.module
        assert mod.namespace.value == "ua.engine.compiled"
        assert len(mod.definitions) > 0

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            engine.compile("nonexistent.ua", _make_namespace())

    def test_inline_string_still_works(self):
        dsl = '''
semiring s:
    contract = ops.join_fn
sort a, b
morphism f : a -> b via "x"
'''
        arch = engine.compile(dsl, _make_namespace())
        assert "f" in arch.paths
