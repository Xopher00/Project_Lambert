"""Bridge between the engine DSL and Hydra's type/term system.

Maps engine sorts to Hydra Types, provides TermCoders for numpy arrays,
bundles, and scalars, and builds Hydra function types for morphisms.
Also provides the shared dotted-name resolver used by the compiler and primitives.

References
----------
Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
*Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
"""

import sys
from pathlib import Path
from typing import Any

_hydra_initialized = False


def setup_hydra_path():
    global _hydra_initialized
    if _hydra_initialized:
        return
    root = Path(__file__).resolve().parent.parent / "hydra"
    paths = [
        str(root / "heads" / "python" / "src" / "main" / "python"),
        str(root / "dist" / "python" / "hydra-kernel" / "src" / "main" / "python"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    _hydra_initialized = True


setup_hydra_path()

from hydra.core import Name, Term, Type  # noqa: E402
from hydra.dsl import prims  # noqa: E402
from hydra.dsl.python import Right  # noqa: E402
from hydra.graph import TermCoder  # noqa: E402

import hydra.dsl.types as types  # noqa: E402

from engine.decl import SortDecl  # noqa: E402


# ---------------------------------------------------------------------------
# Dotted-name resolver (shared by compiler.py and primitives.py)
# ---------------------------------------------------------------------------

def resolve(dotted: str, namespace: dict[str, Any]) -> Any:
    """Walk a dotted name through a namespace dict."""
    parts = dotted.split('.')
    obj = namespace.get(parts[0])
    if obj is None:
        raise NameError(f"Name {parts[0]!r} not found in provided namespace")
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# Field type mapping for structured sorts
# ---------------------------------------------------------------------------

def _field_type(type_name: str) -> Type:
    if type_name == "tensor":
        return types.variable("ua.tensor.NDArray")
    if type_name == "int":
        return types.int32()
    if type_name == "list":
        return types.list_(types.variable("ua.tensor.NDArray"))
    return types.variable("ua.sort." + type_name)


# ---------------------------------------------------------------------------
# Sort → Type
# ---------------------------------------------------------------------------


def sort_to_type(
    name: str, sort_defs: dict[str, SortDecl] | None = None
) -> Type:
    """Map an engine sort to a Hydra Type, encoding the hom-tensor adjunction.

    References
    ----------
    Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
    *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
    """
    if sort_defs and name in sort_defs:
        decl = sort_defs[name]
        if decl.fields:
            fields = [
                types.field(fname, _field_type(ftype))
                for fname, ftype in decl.fields.items()
            ]
            return types.record_with_name(Name("ua.sort." + name), fields)
    return types.variable("ua.sort." + name)


# ---------------------------------------------------------------------------
# TermCoders
# ---------------------------------------------------------------------------


def ndarray_coder() -> TermCoder:
    return TermCoder(
        type=types.variable("ua.tensor.NDArray"),
        encode=lambda cx, g, t: Right(t),
        decode=lambda cx, v: Right(v),
    )


def bundle_coder() -> TermCoder:
    return TermCoder(
        type=types.variable("ua.tensor.Bundle"),
        encode=lambda cx, g, t: Right(t),
        decode=lambda cx, v: Right(v),
    )


def temp_coder() -> TermCoder:
    return prims.float64()


def equation_coder() -> TermCoder:
    return prims.string()



# ---------------------------------------------------------------------------
# Bulk sort → type mapping
# ---------------------------------------------------------------------------


def sort_types_from_defs(
    sort_defs: dict[str, SortDecl], morphism_sorts: set[str]
) -> dict[str, Type]:
    all_names = set(sort_defs.keys()) | morphism_sorts
    return {name: sort_to_type(name, sort_defs) for name in sorted(all_names)}


# ---------------------------------------------------------------------------
# Term constructors for architecture metadata
# ---------------------------------------------------------------------------


def morphism_to_term(spec) -> Term:
    """Encode a MorphismSpec as a Hydra record term."""
    import hydra.dsl.terms as Terms

    fields = [
        Terms.field("name", Terms.string(spec.name)),
        Terms.field("equation", Terms.string(spec.equation)),
        Terms.field("srcSort", Terms.string(spec.src_sort)),
        Terms.field("tgtSort", Terms.string(spec.tgt_sort)),
        Terms.field("arity", Terms.string(spec.arity)),
    ]
    if spec.accumulate:
        fields.append(Terms.field("accumulate", Terms.string(spec.accumulate)))
    return Terms.record(Name("ua.engine.Morphism"), fields)


def path_to_term(
    name: str,
    morphism_names: list[str],
    residual: bool = False,
    normed: str | None = None,
) -> Term:
    """Encode a path as a Hydra record term."""
    import hydra.dsl.terms as Terms

    fields = [
        Terms.field("name", Terms.string(name)),
        Terms.field("morphisms", Terms.list_([Terms.string(m) for m in morphism_names])),
        Terms.field("residual", Terms.boolean(residual)),
    ]
    if normed:
        fields.append(Terms.field("normed", Terms.string(normed)))
    return Terms.record(Name("ua.engine.Path"), fields)


def fan_to_term(name: str, branches: list[str], merge: str) -> Term:
    """Encode a fan as a Hydra record term."""
    import hydra.dsl.terms as Terms

    return Terms.record(Name("ua.engine.Fan"), [
        Terms.field("name", Terms.string(name)),
        Terms.field("branches", Terms.list_([Terms.string(b) for b in branches])),
        Terms.field("merge", Terms.string(merge)),
    ])


def functor_to_union_type(cases) -> Type:
    """Convert a list of CaseDecl objects into a Hydra union type.

    Each case becomes a variant with a payload record encoding
    its data count and recursive children count.
    """
    import hydra.dsl.types as T

    ndarray_type = T.variable("ua.tensor.NDArray")
    fields = []
    for c in cases:
        case_fields = [
            T.field("data", T.list_(ndarray_type)),
        ]
        if c.recursive > 0:
            case_fields.append(
                T.field("children", T.list_(T.variable("ua.engine.TreeNode")))
            )
        if getattr(c, 'output', 0) > 0:
            case_fields.append(
                T.field("output", ndarray_type)
            )
        fields.append(T.field(c.name, T.record_with_name(Name("ua.engine.case." + c.name), case_fields)))
    return T.union(fields)


def arch_to_term(name: str, algebra_cases=None,
                 observer_convergence: str | None = None,
                 observer_loss: str | None = None) -> Term:
    """Encode an arch declaration as a Hydra record term."""
    import hydra.dsl.terms as Terms

    fields = [Terms.field("name", Terms.string(name))]

    if algebra_cases is not None:
        case_terms = [
            Terms.record(Name("ua.engine.Case"), [
                Terms.field("name", Terms.string(c.name)),
                Terms.field("recursive", Terms.int32(c.recursive)),
                Terms.field("data", Terms.int32(c.data)),
                Terms.field("output", Terms.int32(getattr(c, 'output', 0))),
            ])
            for c in algebra_cases
        ]
        fields.append(Terms.field("algebraCases", Terms.list_(case_terms)))

    if observer_convergence:
        fields.append(Terms.field("observerConvergence", Terms.string(observer_convergence)))
    if observer_loss:
        fields.append(Terms.field("observerLoss", Terms.string(observer_loss)))

    return Terms.record(Name("ua.engine.Arch"), fields)
