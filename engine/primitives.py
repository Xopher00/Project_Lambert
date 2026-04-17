"""
Registers engine tensor operations as Hydra Primitive objects.

Follows the pattern in hydra.sources.libraries (qname, primN constructors).
hydra_bridge is imported lazily inside register_primitives so that
setup_hydra_path() runs before any Hydra import at call site.

Public API
----------
qname(ns, local)                     -> Name
register_primitives(...)             -> dict[Name, Primitive]
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

from engine.sorts import setup_hydra_path as _setup  # ensures Hydra on sys.path
from engine.runtime import _resolve

_setup()

from hydra.core import Name  # noqa: E402 — after path setup

NS = "ua.lib.tensor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qname(namespace: str, local_name: str) -> Name:
    """Qualified Hydra Name following libraries.py convention."""
    return Name(f"{namespace}.{local_name}")


# ---------------------------------------------------------------------------
# Main registration
# ---------------------------------------------------------------------------

def register_primitives(
    semiring_contracts: dict,
    morphism_to_semiring: dict,
    compiled_equations: dict,
    morphism_decls: dict[str, Any],
    namespace: dict[str, Any],
) -> dict:
    """Build Hydra Primitives for op-override morphisms and semiring-contract morphisms.

    Returns hydra_primitives: Name -> Primitive

    References
    ----------
    Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
    PODS 2007, pp. 31–40.  cite{green2007}
    """
    from engine.sorts import ndarray_coder
    from hydra.dsl import prims

    primitives: dict = {}
    nd = ndarray_coder()

    # --- morphism op overrides -----------------------------------------------
    for m_name, m_decl in morphism_decls.items():
        if m_decl['op'] is None:
            continue
        try:
            op_fn = _resolve(m_decl['op'], namespace)
        except (KeyError, AttributeError) as exc:
            raise ImportError(
                f"Cannot resolve morphism op '{m_decl['op']}' for '{m_name}': {exc}"
            ) from exc

        prim_name = qname(NS, m_name)
        eq = compiled_equations.get(m_name, "")

        if m_decl['arity'] in ("unary", "pointwise"):
            bound_op = lambda x, _op=op_fn: _op(x)
            primitives[prim_name] = prims.prim1(
                prim_name, bound_op, [], nd, nd
            )
        else:
            bound_op = lambda x, y, _op=op_fn, _eq=eq: _op(_eq, x, y)
            primitives[prim_name] = prims.prim2(
                prim_name, bound_op, [], nd, nd, nd
            )

    # --- semiring-contract morphisms (equation baked in) ---------------------
    for morph_name, eq in compiled_equations.items():
        # Skip morphisms already registered with an op override in group 1.
        if qname(NS, morph_name) in primitives:
            continue
        sr_name = morphism_to_semiring.get(morph_name)
        if sr_name is None or sr_name == '_bridge':
            continue
        contract = semiring_contracts.get(sr_name)
        if contract is None:
            continue

        prim_name = qname(NS, morph_name)

        primitives[prim_name] = prims.prim3(
            prim_name,
            lambda eq_str, x, y, _c=contract: _c(eq_str, x, y),
            [],
            prims.string(), nd, nd, nd,
        )

    return primitives


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_engine_graph(primitives: dict, bound_terms: dict | None = None):
    """Assemble a Hydra Graph containing engine primitives and optional bound terms."""
    from hydra.dsl.python import FrozenDict
    from hydra.graph import Graph
    from hydra.lexical import build_graph

    # Convert dict of primitives to FrozenDict for build_graph
    prim_dict = FrozenDict(primitives)

    # build_graph returns a Graph with all non-primitive fields filtered/empty
    # Pass empty tuple for elements, empty FrozenDict for environment
    graph = build_graph((), FrozenDict({}), prim_dict)

    # hydra.lexical.build_graph does not accept bound_terms; reconstruct Graph manually
    # to splice in bound_terms alongside the built primitives and environment.
    # Revisit if hydra.lexical adds a bound_terms parameter in a future kernel version.
    if bound_terms:
        return Graph(
            bound_terms=FrozenDict(bound_terms),
            bound_types=graph.bound_types,
            class_constraints=graph.class_constraints,
            lambda_variables=graph.lambda_variables,
            metadata=graph.metadata,
            primitives=graph.primitives,
            schema_types=graph.schema_types,
            type_variables=graph.type_variables,
        )

    return graph
