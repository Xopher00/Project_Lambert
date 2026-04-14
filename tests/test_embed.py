"""
Tests for lattice/embed.py (Embed class and Learner class).

Sprint 1 — Fixpoint Abstraction:
  TestFixpointStep:           direct tests for _fixpoint_step (RED before implementation)
  TestConjointCompanionHom:   _conjoint_hom and _companion_hom delegate correctly (RED before implementation)
  TestAttendRegression:       Attend output is bit-identical to reference at temp=0.0 and temp=1.0
  TestRecallRegression:       Recall output is bit-identical to reference at temp=0.0 and temp=1.0

The Learner class tests live in tests/test_learn.py.
"""
import numpy as np
import numpy.testing as npt
import pytest

from legacy.lattice.embed import Embed, Learner

# Re-export so pytest collects Learner tests when running test_embed directly.
from test_learn import (
    TestLearnNewAttractor,
    TestLearnPreservesOld,
    TestLearnMergeIsMax,
    TestLearnShapeValidation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _seed42_inputs(n=5, k=3):
    """
    Return a fixed-seed (n, k) embedding matrix and a (k,) query vector and
    a (n,) entity vector, all in [0, 1].  Used across all regression tests.
    """
    rng = np.random.default_rng(42)
    emb = rng.random((n, k))
    q   = rng.random((k,))
    a   = rng.random((n,))
    return emb, q, a


# ---------------------------------------------------------------------------
# Reference implementations (independent of the class under test)
# ---------------------------------------------------------------------------

def _attend_reference(embed, q, emb, temp):
    """Attend computed directly from Join — the pre-refactoring formula."""
    q2d    = q.reshape(1, -1)
    scores = embed.Join(q2d, emb.T, temp)
    out    = embed.Join(scores, emb, temp)
    return out.squeeze()


def _recall_reference(embed, a, emb, temp):
    """Recall computed directly from Residuate — the pre-refactoring formula."""
    a2d = a.reshape(-1, 1)
    b   = embed.Residuate(emb, a2d, temp).reshape(-1, 1)
    return embed.Residuate(emb.T, b, temp).reshape(-1)


# ---------------------------------------------------------------------------
# T1 (RED): TestFixpointStep — must FAIL before _fixpoint_step is added
# ---------------------------------------------------------------------------

class TestFixpointStep:
    """Direct tests for Embed._fixpoint_step."""

    def test_fixpoint_step_sigma_matches_join_join(self):
        """
        _fixpoint_step with Join-based callables must equal
        the raw Join/Join computation (i.e. the Σ direction).
        """
        embed = Embed()
        emb, q, _ = _seed42_inputs()

        encode = lambda x, e, t: embed.Join(x.reshape(1, -1), e.T, t)
        decode = lambda s, e, t: embed.Join(s, e, t).squeeze()

        result   = embed._fixpoint_step(q, emb, encode, decode, temp=0.0)
        expected = _attend_reference(embed, q, emb, temp=0.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="_fixpoint_step(Σ) must match direct Join/Join at temp=0.0")

    def test_fixpoint_step_pi_matches_residuate_residuate(self):
        """
        _fixpoint_step with Residuate-based callables must equal
        the raw Residuate/Residuate computation (i.e. the Π direction).
        """
        embed = Embed()
        emb, _, a = _seed42_inputs()

        encode = lambda x, e, t: embed.Residuate(e, x.reshape(-1, 1), t)
        decode = lambda b, e, t: embed.Residuate(e.T, b, t).reshape(-1)

        result   = embed._fixpoint_step(a, emb, encode, decode, temp=0.0)
        expected = _recall_reference(embed, a, emb, temp=0.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="_fixpoint_step(Π) must match direct Residuate/Residuate at temp=0.0")

    def test_fixpoint_step_performs_no_reshaping(self):
        """
        _fixpoint_step must pass the output of encode directly to decode,
        without any additional reshaping inside the method.
        Verified by using callables that track their input/output shapes.
        """
        embed = Embed()
        emb, q, _ = _seed42_inputs()

        shapes = {}

        def encode(x, e, t):
            out = embed.Join(x.reshape(1, -1), e.T, t)   # (1, n)
            shapes['encode_out'] = out.shape
            return out

        def decode(s, e, t):
            shapes['decode_in'] = s.shape
            return embed.Join(s, e, t).squeeze()

        embed._fixpoint_step(q, emb, encode, decode, temp=0.0)

        assert shapes['encode_out'] == shapes['decode_in'], (
            "_fixpoint_step must pass encode output directly to decode with no reshaping"
        )

    def test_fixpoint_step_docstring_names_cql_directions(self):
        """_fixpoint_step docstring must mention 'Σ' and 'Π'."""
        embed = Embed()
        doc = embed._fixpoint_step.__doc__ or ""
        assert "Σ" in doc, "_fixpoint_step docstring must mention Σ direction"
        assert "Π" in doc, "_fixpoint_step docstring must mention Π direction"


# ---------------------------------------------------------------------------
# T1 (RED): TestConjointCompanionHom — must FAIL before the wrappers are added
# ---------------------------------------------------------------------------

class TestConjointCompanionHom:
    """_conjoint_hom and _companion_hom are pure delegation wrappers."""

    def test_conjoint_hom_equals_residuate(self):
        """
        _conjoint_hom(emb, x, temp) == Residuate(emb, x, temp)
        for seed-42 inputs.
        """
        embed = Embed()
        emb, _, a = _seed42_inputs()
        x = a.reshape(-1, 1)     # (n, 1) — conjoint expects column vector

        result   = embed._conjoint_hom(emb, x, temp=0.0)
        expected = embed.Residuate(emb, x, temp=0.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="_conjoint_hom must equal Residuate(emb, x)")

    def test_conjoint_hom_at_temp_1(self):
        """_conjoint_hom works at temp=1.0 (soft regime)."""
        embed = Embed()
        emb, _, a = _seed42_inputs()
        x = a.reshape(-1, 1)

        result   = embed._conjoint_hom(emb, x, temp=1.0)
        expected = embed.Residuate(emb, x, temp=1.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="_conjoint_hom(temp=1.0) must equal Residuate(emb, x, temp=1.0)")

    def test_companion_hom_equals_residuate_emb_T(self):
        """
        _companion_hom(x, emb, temp) == Residuate(x, emb.T, temp)
        for seed-42 inputs.
        """
        embed = Embed()
        emb, q, _ = _seed42_inputs()
        x = q.reshape(-1, 1)     # (k, 1) — companion takes concept vector

        result   = embed._companion_hom(x, emb, temp=0.0)
        expected = embed.Residuate(x, emb.T, temp=0.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="_companion_hom must equal Residuate(x, emb.T)")

    def test_companion_hom_at_temp_1(self):
        """_companion_hom works at temp=1.0 (soft regime)."""
        embed = Embed()
        emb, q, _ = _seed42_inputs()
        x = q.reshape(-1, 1)

        result   = embed._companion_hom(x, emb, temp=1.0)
        expected = embed.Residuate(x, emb.T, temp=1.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="_companion_hom(temp=1.0) must equal Residuate(x, emb.T, temp=1.0)")

    def test_conjoint_hom_docstring_names_pi_direction(self):
        """_conjoint_hom docstring must mention 'Π' and 'F̃'."""
        embed = Embed()
        doc = embed._conjoint_hom.__doc__ or ""
        assert "Π" in doc, "_conjoint_hom docstring must mention Π direction"
        assert "F̃" in doc, "_conjoint_hom docstring must mention conjoint bimodule F̃"

    def test_companion_hom_docstring_names_delta_direction(self):
        """_companion_hom docstring must mention 'Δ' and 'F̂'."""
        embed = Embed()
        doc = embed._companion_hom.__doc__ or ""
        assert "Δ" in doc, "_companion_hom docstring must mention Δ direction"
        assert "F̂" in doc, "_companion_hom docstring must mention companion bimodule F̂"


# ---------------------------------------------------------------------------
# TestAttendRegression — bit-identical output before and after refactoring
# ---------------------------------------------------------------------------

class TestAttendRegression:
    """Attend produces bit-identical output to the reference at temp=0 and temp=1."""

    def test_attend_temp0(self):
        embed = Embed()
        emb, q, _ = _seed42_inputs()

        result   = embed.Attend(q, emb, temp=0.0)
        expected = _attend_reference(embed, q, emb, temp=0.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="Attend(temp=0.0) must match reference Join/Join result")

    def test_attend_temp1(self):
        embed = Embed()
        emb, q, _ = _seed42_inputs()

        result   = embed.Attend(q, emb, temp=1.0)
        expected = _attend_reference(embed, q, emb, temp=1.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="Attend(temp=1.0) must match reference Join/Join result")

    def test_attend_output_shape(self):
        """Attend output shape is (k,), matching the input query shape."""
        embed = Embed()
        emb, q, _ = _seed42_inputs()
        result = embed.Attend(q, emb, temp=0.0)
        assert result.shape == q.shape, (
            f"Attend output shape {result.shape} must equal q.shape {q.shape}"
        )


# ---------------------------------------------------------------------------
# TestRecallRegression — bit-identical output before and after refactoring
# ---------------------------------------------------------------------------

class TestRecallRegression:
    """Recall produces bit-identical output to the reference at temp=0 and temp=1."""

    def test_recall_temp0(self):
        embed = Embed()
        emb, _, a = _seed42_inputs()

        result   = embed.Recall(a, emb, temp=0.0)
        expected = _recall_reference(embed, a, emb, temp=0.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="Recall(temp=0.0) must match reference Residuate/Residuate result")

    def test_recall_temp1(self):
        embed = Embed()
        emb, _, a = _seed42_inputs()

        result   = embed.Recall(a, emb, temp=1.0)
        expected = _recall_reference(embed, a, emb, temp=1.0)

        npt.assert_allclose(result, expected, atol=1e-9,
            err_msg="Recall(temp=1.0) must match reference Residuate/Residuate result")

    def test_recall_output_shape(self):
        """Recall output shape is (n,), matching the input entity vector shape."""
        embed = Embed()
        emb, _, a = _seed42_inputs()
        result = embed.Recall(a, emb, temp=0.0)
        assert result.shape == a.shape, (
            f"Recall output shape {result.shape} must equal a.shape {a.shape}"
        )
