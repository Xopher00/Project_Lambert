"""Tests for engine/terms.py (phantom-typed DSL).

Requires Python 3.12+ — skipped on older versions.
"""

import sys
import pytest

if sys.version_info < (3, 12):
    pytest.skip("engine.terms requires Python 3.12+", allow_module_level=True)

from engine import terms  # noqa: E402
from engine.runtime import MorphismSpec  # noqa: E402
from engine.sorts import morphism_to_term  # noqa: E402
from hydra.dsl.meta.phantoms import TTerm  # noqa: E402
from hydra.core import TermRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(t: TTerm):
    """Unwrap TTerm -> TermRecord -> Record."""
    assert isinstance(t, TTerm), f"expected TTerm, got {type(t)}"
    tr = t.value
    assert isinstance(tr, TermRecord), f"expected TermRecord, got {type(tr)}"
    return tr.value


def _field_names(t: TTerm) -> list[str]:
    rec = _record(t)
    return [f.name.value for f in rec.fields]


# ---------------------------------------------------------------------------
# TestTermsModule
# ---------------------------------------------------------------------------

class TestTermsModule:

    def test_morphism_returns_tterm(self):
        result = terms.morphism("proj", "i", "j", "ij->j")
        assert isinstance(result, TTerm)
        assert isinstance(result.value, TermRecord)

    def test_morphism_fields(self):
        result = terms.morphism("proj", "i", "j", "ij->j")
        field_names = _field_names(result)
        assert "name" in field_names
        assert "equation" in field_names
        assert "srcSort" in field_names
        assert "tgtSort" in field_names
        assert "arity" in field_names

    def test_path_returns_tterm(self):
        result = terms.path("read", ["proj", "score"])
        assert isinstance(result, TTerm)
        assert isinstance(result.value, TermRecord)

    def test_path_with_residual(self):
        result = terms.path("attn", ["proj"], residual=True)
        field_names = _field_names(result)
        assert "residual" in field_names
        rec = _record(result)
        fields = {f.name.value: f.term for f in rec.fields}
        from hydra.core import TermLiteral, LiteralBoolean
        assert isinstance(fields["residual"], TermLiteral)
        assert isinstance(fields["residual"].value, LiteralBoolean)
        assert fields["residual"].value.value is True

    def test_fan_returns_tterm(self):
        result = terms.fan("kv", ["k", "v"])
        assert isinstance(result, TTerm)
        field_names = _field_names(result)
        assert "branches" in field_names

    def test_case_returns_tterm(self):
        result = terms.case("leaf", recursive=0, data=1)
        assert isinstance(result, TTerm)
        field_names = _field_names(result)
        assert "recursive" in field_names
        assert "data" in field_names

    def test_arch_with_algebra(self):
        leaf = terms.case("leaf", 0, 1)
        result = terms.arch("T", algebra_cases=[leaf])
        assert isinstance(result, TTerm)
        field_names = _field_names(result)
        assert "algebraCases" in field_names


# ---------------------------------------------------------------------------
# TestTermsEquivalence
# ---------------------------------------------------------------------------

class TestTermsEquivalence:

    def test_morphism_equivalent_to_bridge(self):
        """terms.morphism() must produce a Term identical to morphism_to_term(spec)."""
        spec = MorphismSpec(
            name="proj",
            op=lambda eq, *a: None,
            equation="ij->j",
            src_sort="i",
            tgt_sort="j",
        )
        term_via_bridge = morphism_to_term(spec)
        tterm_via_terms = terms.morphism("proj", "i", "j", "ij->j")
        assert term_via_bridge == tterm_via_terms.value
