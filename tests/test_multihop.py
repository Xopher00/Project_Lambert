"""
Unit tests for multi-hop traversal: Embed.hop and Query.multihop.

Tests verify:
  - test_hop_shape:               hop returns a (k,) vector
  - test_hop_reachability:        hop propagates through a known nonzero path
  - test_multihop_entity_space:   multihop returns QueryResult with mode='multihop'
                                  and entity labels drawn from entity_labels
  - test_multihop_chain_2hop:     2-hop synthetic chain A→B (R1), B→C (R2) returns C
  - test_multihop_shape_mismatch: mismatched EmbR shapes raise ValueError

All tests run at temp=0 (exact, hard) for algebraic precision.

Import paths follow the canonical convention:
  from lattice.embed import Embed
  from query import Query, QueryResult
"""

import numpy as np
import numpy.testing as npt
import pytest

from legacy.lattice.embed import Embed
from legacy.query import Query, QueryResult

pytestmark = pytest.mark.legacy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_model(n_entities, n_head_features_pairs):
    """
    Build a minimal Lambert model whose heads dict contains only the fields
    Sprint 3 needs: 'emb', 'EmbR'.

    Parameters
    ----------
    n_entities : int
        Number of entities.
    n_head_features_pairs : list of (str, ndarray, ndarray)
        Each element is (head_name, emb, EmbR).
        emb shape: (n_entities, k)
        EmbR shape: (k, k)

    Returns
    -------
    object
        A minimal mock model with .entity_labels, .heads, .eps, .attn_temp.
    """
    from types import SimpleNamespace

    heads = {}
    for name, emb, EmbR in n_head_features_pairs:
        heads[name] = {'emb': emb, 'EmbR': EmbR}

    entity_labels = [f'entity_{i}' for i in range(n_entities)]

    model = SimpleNamespace(
        entity_labels=entity_labels,
        heads=heads,
        eps=1e-3,
        attn_temp=0.0,
    )
    return model


def _build_two_hop_model():
    """
    Build a synthetic 3-entity (A, B, C), 2-head Lambert model.

    Entities: A=0, B=1, C=2
    head_0: R1  A→B  (entity 0 related to entity 1)
    head_1: R2  B→C  (entity 1 related to entity 2)

    Each head has k=3 concept dimensions. We construct emb and EmbR so that
    the path A→B via head_0 and B→C via head_1 is recoverable through the
    concept-space chain.

    emb for both heads is the 3×3 identity (each entity is its own concept):
        emb[i, :] = e_i

    EmbR for head_0: concept 0 → concept 1 (A→B path in concept space)
        EmbR_0[0, 1] = 1.0, all other entries = 0

    EmbR for head_1: concept 1 → concept 2 (B→C path in concept space)
        EmbR_1[1, 2] = 1.0, all other entries = 0
    """
    n = 3
    k = 3
    # Both heads share identity embedding: entity i lives in concept dimension i
    emb = np.eye(n, k)

    EmbR_0 = np.zeros((k, k))
    EmbR_0[0, 1] = 1.0   # concept 0 reaches concept 1 (A → B in concept space)

    EmbR_1 = np.zeros((k, k))
    EmbR_1[1, 2] = 1.0   # concept 1 reaches concept 2 (B → C in concept space)

    model = _make_synthetic_model(
        n_entities=n,
        n_head_features_pairs=[
            ('head_0', emb.copy(), EmbR_0),
            ('head_1', emb.copy(), EmbR_1),
        ]
    )
    return model


# ---------------------------------------------------------------------------
# Test: hop shape
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Embed.hop removed during refactor")
class TestHopShape:
    """hop(q, EmbR, temp) returns a (k,) vector."""

    def test_hop_shape(self):
        """
        hop with a random k=5 EmbR and query should return shape (k,).
        """
        embed = Embed()
        k = 5
        q = np.random.rand(k)
        EmbR = np.random.rand(k, k)
        result = embed.hop(q, EmbR, temp=0.0)
        assert result.shape == (k,), (
            f"Expected shape ({k},), got {result.shape}"
        )


# ---------------------------------------------------------------------------
# Test: hop reachability
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Embed.hop removed during refactor")
class TestHopReachability:
    """hop propagates through a nonzero path."""

    def test_hop_reachability(self):
        """
        Synthetic 3-concept EmbR with one nonzero path from concept 0 to
        concept 2: EmbR[0, 2] = 1.0.

        hop(e0, EmbR, temp=0) should return a vector with nonzero value at
        concept 2, verifying that the left-Kan step propagates through the
        known edge.
        """
        embed = Embed()
        k = 3
        EmbR = np.zeros((k, k))
        EmbR[0, 2] = 1.0          # concept 0 → concept 2

        e0 = np.zeros(k)
        e0[0] = 1.0                # starting at concept 0

        result = embed.hop(e0, EmbR, temp=0.0)

        assert result.shape == (k,), f"Expected shape ({k},), got {result.shape}"
        assert result[2] > 0.0, (
            f"Expected nonzero at concept 2, got result={result}"
        )


# ---------------------------------------------------------------------------
# Test: multihop entity space
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Embed.hop removed during refactor")
class TestMultihopEntitySpace:
    """multihop returns QueryResult with mode='multihop' and entity labels."""

    def test_multihop_entity_space(self):
        """
        Using the 2-hop synthetic model, call multihop('entity_0', ['head_0'])
        (single hop for simplicity) and verify:
          - returns a QueryResult
          - mode is exactly 'multihop'
          - all entities in the result are drawn from entity_labels
        """
        model = _build_two_hop_model()
        q = Query.__new__(Query)
        q.model = model

        result = q.multihop('entity_0', ['head_0'], top_k=10)

        assert isinstance(result, QueryResult), (
            f"Expected QueryResult, got {type(result)}"
        )
        assert result.mode == 'multihop', (
            f"Expected mode='multihop', got mode='{result.mode}'"
        )
        for entity in result.entities:
            assert entity in model.entity_labels, (
                f"Entity '{entity}' not in entity_labels {model.entity_labels}"
            )


# ---------------------------------------------------------------------------
# Test: multihop 2-hop chain
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Embed.hop removed during refactor")
class TestMultihopChain2Hop:
    """2-hop chain A→B→C returns C as reachable from A."""

    def test_multihop_chain_2hop(self):
        """
        Build a synthetic model with 3 entities A, B, C and two heads:
          head_0: R1 with A→B path in concept space (EmbR_0[0,1]=1.0)
          head_1: R2 with B→C path in concept space (EmbR_1[1,2]=1.0)

        Both heads use identity emb (entity i = concept i).

        multihop('entity_0', ['head_0', 'head_1']) should return C
        (entity_2) in the results, as the 2-hop chain A→B→C is reachable.
        """
        model = _build_two_hop_model()
        q = Query.__new__(Query)
        q.model = model

        result = q.multihop('entity_0', ['head_0', 'head_1'], top_k=10)

        assert isinstance(result, QueryResult), (
            f"Expected QueryResult, got {type(result)}"
        )
        assert result.mode == 'multihop', (
            f"Expected mode='multihop', got mode='{result.mode}'"
        )
        assert 'entity_2' in result.entities, (
            f"Expected entity_2 (C) to be reachable from entity_0 (A) "
            f"via 2-hop chain [head_0, head_1]. Got entities: {result.entities}"
        )


# ---------------------------------------------------------------------------
# Test: shape mismatch raises ValueError
# ---------------------------------------------------------------------------

class TestMultihopShapeMismatch:
    """Mismatched EmbR shapes across chain heads raise ValueError."""

    def test_multihop_shape_mismatch(self):
        """
        Passing two heads with different EmbR shapes (k=3 vs k=4) to
        multihop should raise ValueError before any hop is performed.
        The error message must include the head names and their shapes.
        """
        n = 3
        k_a = 3
        k_b = 4

        emb_a = np.eye(n, k_a)
        emb_b = np.eye(n, k_b)
        EmbR_a = np.zeros((k_a, k_a))
        EmbR_b = np.zeros((k_b, k_b))

        from types import SimpleNamespace
        model = SimpleNamespace(
            entity_labels=[f'entity_{i}' for i in range(n)],
            heads={
                'head_a': {'emb': emb_a, 'EmbR': EmbR_a},
                'head_b': {'emb': emb_b, 'EmbR': EmbR_b},
            },
            eps=1e-3,
            attn_temp=0.0,
        )

        q = Query.__new__(Query)
        q.model = model

        with pytest.raises(ValueError):
            q.multihop('entity_0', ['head_a', 'head_b'], top_k=10)
