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

from engine.parser import SortDecl as _SortDecl, SortCoercion as _SortCoercion  # noqa: E402


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
    ])


def path_type() -> Type:
    """ua.engine.Path: V-functor composition (Lawvere 1973).

    src/tgt sorts inferred from chain at step 4. merge excluded (runtime detail).
    """
    return types.record_with_name(Name("ua.engine.Path"), [
        types.field("name", types.string()),
        types.field("morphisms", types.list_(types.string())),
        types.field("residual", types.boolean()),
    ])


def fan_type() -> Type:
    """ua.engine.Fan: V-category product (Lawvere 1973, Shen & Tang 2022).

    merge excluded: runtime aggregation strategy, not a type property.
    """
    return types.record_with_name(Name("ua.engine.Fan"), [
        types.field("name", types.string()),
        types.field("branches", types.list_(types.string())),
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
    ])


def arch_type() -> Type:
    """ua.engine.Arch: initial algebra / final coalgebra (Gavranovic §5).

    cases declare the endofunctor F. stepEnter/stepEmit are essential
    coalgebra morphisms. Observers and step_compute are excluded.
    """
    return types.record_with_name(Name("ua.engine.Arch"), [
        types.field("name", types.string()),
        types.field("cases", types.maybe(types.list_(types.string()))),
        types.field("stepEnter", types.maybe(types.string())),
        types.field("stepEmit", types.maybe(types.string())),
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
    for i in range(1, len(names)):
        prev_tgt = morphism_specs[names[i - 1]].tgt_sort
        curr_src = morphism_specs[names[i]].src_sort
        if prev_tgt == curr_src:
            continue
        prev_type = sort_types.get(prev_tgt)
        curr_type = sort_types.get(curr_src)
        if prev_type is not None and curr_type is not None and prev_type == curr_type:
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


# ---------------------------------------------------------------------------
# Hydra term bridge
# ---------------------------------------------------------------------------


def morphism_to_term(spec) -> "object":
    """Map a MorphismSpec to a TTerm (Hydra bridging)."""
    from engine.terms import morphism as _morphism_term
    template_params = [spec.template_param] if getattr(spec, 'template_param', None) else []
    return _morphism_term(spec.name, spec.src_sort, spec.tgt_sort, spec.arity, template_params).value
