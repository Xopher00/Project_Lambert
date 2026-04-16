"""Bridge between the engine DSL and Hydra's type/term system.

Maps engine sorts to Hydra Types, provides TermCoders for numpy arrays,
bundles, and scalars, and builds Hydra function types for morphisms.

References
----------
Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
*Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
"""

import sys
from pathlib import Path
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

from hydra.core import Name, Type  # noqa: E402
from hydra.dsl import prims  # noqa: E402
from hydra.dsl.python import Right  # noqa: E402
from hydra.graph import TermCoder  # noqa: E402

import hydra.dsl.types as types  # noqa: E402

from engine.decl import SortDecl  # noqa: E402


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


def functor_to_union_type(cases) -> Type:
    """Convert a list of CaseDecl objects into a Hydra union type.

    Each case becomes a variant with a payload record encoding
    its data count and recursive children count.

    Note: hydra.rewriting (fold_over_term, rewrite_term) operates on existing
    Hydra terms and is not applicable here — this function constructs a new
    Type from DSL data rather than traversing an existing term.
    """
    ndarray_type = types.variable("ua.tensor.NDArray")

    def _case_fields(c):
        fs = [types.field("data", types.list_(ndarray_type))]
        if c.recursive > 0:
            fs.append(types.field("children", types.list_(types.variable("ua.engine.TreeNode"))))
        if getattr(c, 'output', 0) > 0:
            fs.append(types.field("output", ndarray_type))
        return types.field(c.name, types.record_with_name(Name("ua.engine.case." + c.name), fs))

    return types.union([_case_fields(c) for c in cases])


