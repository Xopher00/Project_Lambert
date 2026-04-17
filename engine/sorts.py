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

from hydra.core import Name, Type, TypeScheme, TypeVariable as _TypeVariable  # noqa: E402
from hydra.dsl import prims  # noqa: E402
from hydra.dsl.python import FrozenDict as _FD, Nothing, Right  # noqa: E402
from hydra.graph import Graph as _Graph, TermCoder  # noqa: E402
from hydra.checking import types_effectively_equal as _types_equal  # noqa: E402

import hydra.dsl.types as types  # noqa: E402

from engine.parser import SortDecl as _SortDecl, SortCoercion as _SortCoercion  # noqa: E402


def _graph_for_sorts(sort_types: dict) -> _Graph:
    """Build a Hydra Graph with sort types registered as schema_types.

    Registering sort types prevents types_effectively_equal from treating
    sort TypeVariable names as free-variable wildcards.
    """
    scheme_map = {
        Name("ua.sort." + k): TypeScheme(variables=(), type=v, constraints=Nothing())
        for k, v in sort_types.items()
        if isinstance(k, str)
    }
    return _Graph(
        bound_terms=_FD({}),
        bound_types=_FD({}),
        class_constraints=_FD({}),
        lambda_variables=frozenset(),
        metadata=_FD({}),
        primitives=_FD({}),
        schema_types=_FD(scheme_map),
        type_variables=frozenset(),
    )


_EMPTY_GRAPH = _Graph(
    bound_terms=_FD({}),
    bound_types=_FD({}),
    class_constraints=_FD({}),
    lambda_variables=frozenset(),
    metadata=_FD({}),
    primitives=_FD({}),
    schema_types=_FD({}),
    type_variables=frozenset(),
)


def _sort_types_match(graph: _Graph, t1: Type, t2: Type) -> bool:
    """Check whether two sort types are compatible.

    TypeVariable sorts are compared by name (nominal equality).
    Structured record sorts use types_effectively_equal so typedef
    aliases are normalized before comparison.
    """
    if isinstance(t1, _TypeVariable) or isinstance(t2, _TypeVariable):
        return t1 == t2
    return _types_equal(graph, t1, t2)


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
    name: str, sort_defs: dict | None = None
) -> Type:
    """Map an engine sort to a Hydra Type, encoding the hom-tensor adjunction.

    sort_defs values may be SortDecl objects or plain dicts with a 'fields' key.

    References
    ----------
    Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
    *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
    """
    if sort_defs and name in sort_defs:
        decl = sort_defs[name]
        fields_map = decl.fields
        if fields_map:
            fields = [
                types.field(fname, _field_type(ftype))
                for fname, ftype in fields_map.items()
            ]
            return types.record_with_name(Name("ua.sort." + name), fields)
    return types.variable("ua.sort." + name)


# ---------------------------------------------------------------------------
# TermCoders
# ---------------------------------------------------------------------------


def _ndarray_encode(cx, g, t):
    import numpy as np
    from hydra.core import LiteralBinary, TermLiteral
    raw = t.value.value  # TermLiteral -> LiteralBinary -> bytes
    sep = raw.index(0)
    sep2 = raw.index(0, sep + 1)
    dtype = np.dtype(raw[:sep].decode())
    shape = tuple(int(x) for x in raw[sep + 1:sep2].decode().split(",") if x)
    data = raw[sep2 + 1:]
    return Right(np.frombuffer(data, dtype=dtype).reshape(shape))


def _ndarray_decode(cx, v):
    from hydra.core import LiteralBinary, TermLiteral
    import numpy as np
    arr = np.asarray(v)
    dtype_b = arr.dtype.str.encode()
    shape_b = ",".join(str(s) for s in arr.shape).encode()
    hdr = dtype_b + b"\x00" + shape_b + b"\x00"
    return Right(TermLiteral(LiteralBinary(hdr + arr.tobytes())))


def ndarray_coder() -> TermCoder:
    return TermCoder(
        type=types.variable("ua.tensor.NDArray"),
        encode=_ndarray_encode,
        decode=_ndarray_decode,
    )


def _bundle_encode(cx, g, t):
    import io
    import numpy as np
    raw = t.value.value  # TermLiteral -> LiteralBinary -> bytes
    buf = io.BytesIO(raw)
    buf.seek(0)
    npz = np.load(buf, allow_pickle=False)
    return Right(dict(npz))


def _bundle_decode(cx, v):
    import io
    import numpy as np
    from hydra.core import LiteralBinary, TermLiteral
    buf = io.BytesIO()
    np.savez(buf, **v)
    return Right(TermLiteral(LiteralBinary(buf.getvalue())))


def bundle_coder() -> TermCoder:
    return TermCoder(
        type=types.variable("ua.tensor.Bundle"),
        encode=_bundle_encode,
        decode=_bundle_decode,
    )


# ---------------------------------------------------------------------------
# Bulk sort → type mapping
# ---------------------------------------------------------------------------


def sort_types_from_defs(
    sort_defs: dict, morphism_sorts: set[str]
) -> dict[str, Type]:
    all_names = set(sort_defs.keys()) | morphism_sorts
    return {name: sort_to_type(name, sort_defs) for name in sorted(all_names)}



# ---------------------------------------------------------------------------
# Engine construct type declarations (roadmap 3.4 step 3)
# ---------------------------------------------------------------------------


def morphism_type() -> Type:
    """ua.engine.Morphism: V-functor type signature (Lawvere 1973, Gavranovic §3).

    Fields are type-level only. equation/op/compiler are representations,
    not type properties, and are excluded.
    templateParams is essential: ln[prefix] has a different type than ln.
    """
    return types.record_with_name(Name("ua.engine.Morphism"), [
        types.field("name", types.string()),
        types.field("srcSort", types.string()),
        types.field("tgtSort", types.string()),
        types.field("arity", types.string()),
        types.field("templateParams", types.list_(types.string())),
        types.field("semiring", types.string()),
        types.field("equation", types.string()),
        types.field("op", types.string()),
        types.field("transform", types.string()),
        types.field("compiler", types.string()),
        types.field("accumulate", types.string()),
        types.field("accumulateFields", types.list_(types.string())),
        types.field("templateParam", types.string()),
    ])


def path_type() -> Type:
    """ua.engine.Path: V-functor composition (Lawvere 1973).

    src/tgt sorts inferred from chain at step 4. merge excluded (runtime detail).
    """
    return types.record_with_name(Name("ua.engine.Path"), [
        types.field("name", types.string()),
        types.field("morphisms", types.list_(types.string())),
        types.field("residual", types.boolean()),
        types.field("normed", types.string()),
    ])


def fan_type() -> Type:
    """ua.engine.Fan: V-category product (Lawvere 1973, Shen & Tang 2022).

    merge excluded: runtime aggregation strategy, not a type property.
    """
    return types.record_with_name(Name("ua.engine.Fan"), [
        types.field("name", types.string()),
        types.field("branches", types.list_(types.string())),
        types.field("merge", types.string()),
    ])


def case_type() -> Type:
    """ua.engine.Case: endofunctor F variant (Gavranovic §5).

    recursive + data declare the F-algebra structure.
    iterate/output/cell excluded: catamorphism hints, not type properties.
    """
    return types.record_with_name(Name("ua.engine.Case"), [
        types.field("name", types.string()),
        types.field("recursive", types.int32()),
        types.field("data", types.int32()),
        types.field("output", types.int32()),
        types.field("cell", types.string()),
        types.field("caseMorphisms", types.list_(types.string())),
        types.field("iterate", types.string()),
    ])


def arch_type() -> Type:
    """ua.engine.Arch: initial algebra / final coalgebra (Gavranovic §5).

    cases declare the endofunctor F. stepEnter/stepEmit are essential
    coalgebra morphisms. Observers and step_compute are excluded.
    """
    return types.record_with_name(Name("ua.engine.Arch"), [
        types.field("name", types.string()),
        types.field("cases", types.list_(types.string())),
        types.field("stepEnter", types.maybe(types.string())),
        types.field("stepEmit", types.maybe(types.string())),
        types.field("algebraCell", types.string()),
        types.field("observerConvergence", types.string()),
        types.field("observerLoss", types.string()),
        types.field("stepCompute", types.string()),
        types.field("stateFieldNames", types.list_(types.string())),
        types.field("stateFieldTypes", types.list_(types.string())),
    ])


# ---------------------------------------------------------------------------
# Sort coercion algebra
# ---------------------------------------------------------------------------


def _close_coercions(coercions) -> dict:
    """Compute transitive closure of sort coercions via Floyd-Warshall.

    Input: list[SortCoercion]
    Returns: dict[(src, tgt), grade] — closed under composition (min, *, 1.0).

    Direct declarations take precedence: transitive paths only fill in pairs
    that were not explicitly declared.
    """
    grades: dict[tuple, float] = {}
    direct: set = set()
    for c in coercions:
        grades[(c.src, c.tgt)] = c.grade
        direct.add((c.src, c.tgt))

    sorts = {s for c in coercions for s in (c.src, c.tgt)}
    for k in sorts:
        for i in sorts:
            for j in sorts:
                if (i, j) in direct:
                    continue  # direct declaration wins; do not overwrite
                via = grades.get((i, k), 0.0) * grades.get((k, j), 0.0)
                if via > grades.get((i, j), 0.0):
                    grades[(i, j)] = via
    return grades


def grade_sorts(
    morphism_specs: dict,
    names: list[str],
    sort_types: dict,
    coercion_grades: dict,
    threshold: float = 1.0,
) -> tuple[list[str], list[str]]:
    """Check sort compatibility using graded coercions.

    Returns (errors, warnings).
    Hard errors: adjacent sorts with no declared coercion (grade 0.0).
    Warnings: adjacent sorts whose coercion grade is below threshold.
    """
    errors: list[str] = []
    warnings: list[str] = []
    _graph = _graph_for_sorts(sort_types)
    for i in range(1, len(names)):
        prev_tgt = morphism_specs[names[i - 1]].tgt_sort
        curr_src = morphism_specs[names[i]].src_sort
        if prev_tgt == curr_src:
            continue
        prev_type = sort_types.get(prev_tgt)
        curr_type = sort_types.get(curr_src)
        if prev_type is not None and curr_type is not None and _sort_types_match(_graph, prev_type, curr_type):
            continue
        grade = coercion_grades.get((prev_tgt, curr_src), 0.0)
        msg = (
            f"Type mismatch at step {i}: "
            f"{names[i - 1]!r} outputs {prev_tgt!r} but "
            f"{names[i]!r} expects {curr_src!r}"
        )
        if grade == 0.0:
            errors.append(msg)
        elif grade < threshold:
            warnings.append(f"{msg} (coercion grade {grade:.3f} < threshold {threshold:.3f})")
    return errors, warnings
