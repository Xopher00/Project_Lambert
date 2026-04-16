"""
Registers engine tensor operations as Hydra Primitive objects.

Follows the pattern in hydra.sources.libraries (qname, primN constructors).
hydra_bridge is imported lazily inside register_tensor_primitives so that
setup_hydra_path() runs before any Hydra import at call site.

Public API
----------
qname(ns, local)                     -> Name
register_tensor_primitives(...)      -> dict[Name, Primitive]
build_engine_graph(primitives)       -> Graph

References
----------
Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
PODS 2007, pp. 31–40.  cite{green2007}

Domingos, P. (2025). Tensor logic: The language of AI.
arXiv:2510.12269.  cite{domingos2025}
"""

from __future__ import annotations

from typing import Any, Callable

from engine.decl import MorphismDecl, SemiringDecl
from engine.sorts import setup_hydra_path as _setup  # ensures Hydra on sys.path

_setup()

from hydra.core import Name  # noqa: E402 — after path setup

NS = "ua.lib.tensor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qname(namespace: str, local_name: str) -> Name:
    """Qualified Hydra Name following libraries.py convention."""
    return Name(f"{namespace}.{local_name}")


from engine.utils import resolve as _resolve


# ---------------------------------------------------------------------------
# Main registration
# ---------------------------------------------------------------------------

def register_tensor_primitives(
    semirings: dict[str, SemiringDecl],
    morphisms: dict[str, MorphismDecl],
    namespace: dict[str, Any],
) -> dict:
    """Build a dict of Hydra Primitives for semiring contracts and morphism op overrides.

    Parameters
    ----------
    semirings:  parsed SemiringDecl map (name -> decl)
    morphisms:  parsed MorphismDecl map (name -> decl)
    namespace:  Python namespace for resolving dotted callable names
                (e.g. {'ops': <module>})

    Returns
    -------
    dict[Name, Primitive]  ready to pass to build_engine_graph

    References
    ----------
    Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
    PODS 2007, pp. 31–40.  cite{green2007}
    """
    from engine.sorts import ndarray_coder
    from hydra.dsl import prims

    primitives: dict = {}
    nd = ndarray_coder()

    # --- semiring contracts ---------------------------------------------------
    # Register each semiring contract as a prim2: (x, y) -> result.
    # The equation and temp are defaulted (equation pre-compiled at morphism
    # compile time; temp=0.0 default). Phase 2 will wrap these into per-morphism
    # primitives with the equation baked in.
    for sr_name, sr_decl in semirings.items():
        try:
            contract = _resolve(sr_decl.contract, namespace)
        except (KeyError, AttributeError) as exc:
            raise ImportError(
                f"Cannot resolve semiring contract '{sr_decl.contract}': {exc}"
            ) from exc

        prim_name = qname(NS, sr_name)

        # Wrap: (x, y) -> contract(compiled_eq=None, x, y, temp=0.0)
        # compiled_eq=None signals the contract to skip equation dispatch
        # (Phase 2 will supply a real compiled equation).
        def _make_contract_fn(fn):
            return lambda x, y: fn(None, x, y, temp=0.0)

        primitives[prim_name] = prims.prim2(
            prim_name,
            _make_contract_fn(contract),
            [],
            nd, nd, nd,
        )

    # --- morphism op overrides -----------------------------------------------
    # For morphisms with an explicit op= override, register as a prim1 or prim2
    # depending on arity.
    for m_name, m_decl in morphisms.items():
        if m_decl.op is None:
            continue
        try:
            op_fn = _resolve(m_decl.op, namespace)
        except (KeyError, AttributeError) as exc:
            raise ImportError(
                f"Cannot resolve morphism op '{m_decl.op}' for '{m_name}': {exc}"
            ) from exc

        prim_name = qname(NS, m_name)

        if m_decl.arity in ("unary", "pointwise"):
            primitives[prim_name] = prims.prim1(
                prim_name, op_fn, [], nd, nd
            )
        else:
            # binary / ternary — expose as prim2 (x, y) -> result
            primitives[prim_name] = prims.prim2(
                prim_name, op_fn, [], nd, nd, nd
            )

    return primitives


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_engine_graph(primitives: dict, bound_terms: dict | None = None):
    """Assemble a Hydra Graph containing engine primitives and optional bound terms."""
    from hydra.dsl.python import FrozenDict
    from hydra.graph import Graph

    return Graph(
        bound_terms=FrozenDict(bound_terms or {}),
        bound_types=FrozenDict({}),
        class_constraints=FrozenDict({}),
        lambda_variables=frozenset(),
        metadata=FrozenDict({}),
        primitives=FrozenDict(primitives),
        schema_types=FrozenDict({}),
        type_variables=frozenset(),
    )
