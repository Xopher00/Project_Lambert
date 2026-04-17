"""Phantom-typed DSL constructors for engine architecture specifications.

Provides typed wrappers around Hydra's phantom DSL for constructing
engine morphism, path, fan, and arch terms with type safety.
"""

from __future__ import annotations

from engine.sorts import setup_hydra_path

setup_hydra_path()

from hydra.core import Name  # noqa: E402
from hydra.dsl.meta.phantoms import (  # noqa: E402
    TTerm,
    boolean,
    field,
    int32,
    just,
    list_,
    nothing,
    record,
    string,
)


# Nominal type names for engine constructs
_MORPHISM = Name("ua.engine.Morphism")
_PATH = Name("ua.engine.Path")
_FAN = Name("ua.engine.Fan")
_CASE = Name("ua.engine.Case")
_ARCH = Name("ua.engine.Arch")

# Field name constants
_NAME = Name("name")
_SRC_SORT = Name("srcSort")
_TGT_SORT = Name("tgtSort")
_ARITY = Name("arity")
_TEMPLATE_PARAMS = Name("templateParams")
_SEMIRING = Name("semiring")
_EQUATION = Name("equation")
_MORPHISMS = Name("morphisms")
_RESIDUAL = Name("residual")
_BRANCHES = Name("branches")
_RECURSIVE = Name("recursive")
_DATA = Name("data")
_CASES = Name("cases")
_STEP_ENTER = Name("stepEnter")
_STEP_EMIT = Name("stepEmit")
# Morphism extended fields
_OP = Name("op")
_TRANSFORM = Name("transform")
_COMPILER = Name("compiler")
_ACCUMULATE = Name("accumulate")
_ACCUMULATE_FIELDS = Name("accumulateFields")
_TEMPLATE_PARAM = Name("templateParam")
# Path extended fields
_NORMED = Name("normed")
# Fan extended fields
_MERGE = Name("merge")
# Case extended fields
_OUTPUT = Name("output")
_CELL = Name("cell")
_CASE_MORPHISMS = Name("caseMorphisms")
_ITERATE = Name("iterate")
# Arch extended fields
_ALGEBRA_CELL = Name("algebraCell")
_OBSERVER_CONVERGENCE = Name("observerConvergence")
_OBSERVER_LOSS = Name("observerLoss")
_STEP_COMPUTE = Name("stepCompute")
_STATE_FIELD_NAMES = Name("stateFieldNames")
_STATE_FIELD_TYPES = Name("stateFieldTypes")


def morphism(name: str, src: str, tgt: str, arity: str = "binary",
             template_params: list[str] | None = None,
             semiring: str = "", equation: str = "",
             op: str = "", transform: str = "", compiler_name: str = "",
             accumulate: str = "",
             accumulate_fields: list[str] | None = None,
             template_param: str = "") -> TTerm:
    """Construct a typed morphism term."""
    fields = [
        field(_NAME, string(name)),
        field(_SRC_SORT, string(src)),
        field(_TGT_SORT, string(tgt)),
        field(_ARITY, string(arity)),
        field(_TEMPLATE_PARAMS, list_([string(p) for p in (template_params or [])])),
        field(_SEMIRING, string(semiring)),
        field(_EQUATION, string(equation)),
        field(_OP, string(op)),
        field(_TRANSFORM, string(transform)),
        field(_COMPILER, string(compiler_name)),
        field(_ACCUMULATE, string(accumulate)),
        field(_ACCUMULATE_FIELDS, list_([string(f) for f in (accumulate_fields or [])])),
        field(_TEMPLATE_PARAM, string(template_param)),
    ]
    return record(_MORPHISM, fields)


def path(name: str, morphisms: list[str], residual: bool = False,
         normed: str = "") -> TTerm:
    """Construct a typed path term."""
    return record(_PATH, [
        field(_NAME, string(name)),
        field(_MORPHISMS, list_([string(m) for m in morphisms])),
        field(_RESIDUAL, boolean(residual)),
        field(_NORMED, string(normed)),
    ])


def fan(name: str, branches: list[str], merge: str = "dict") -> TTerm:
    """Construct a typed fan term."""
    return record(_FAN, [
        field(_NAME, string(name)),
        field(_BRANCHES, list_([string(b) for b in branches])),
        field(_MERGE, string(merge)),
    ])


def case(name: str, recursive: int, data: int,
         output: int = 0, cell: str = "",
         morphisms: list[str] | None = None,
         iterate: str = "") -> TTerm:
    """Construct a typed case term."""
    return record(_CASE, [
        field(_NAME, string(name)),
        field(_RECURSIVE, int32(recursive)),
        field(_DATA, int32(data)),
        field(_OUTPUT, int32(output)),
        field(_CELL, string(cell)),
        field(_CASE_MORPHISMS, list_([string(m) for m in (morphisms or [])])),
        field(_ITERATE, string(iterate)),
    ])


def arch(name: str, cases: list | None = None,
         step_enter: str | None = None,
         step_emit: str | None = None,
         algebra_cell: str = "",
         observer_convergence: str = "",
         observer_loss: str = "",
         step_compute: str = "",
         state_field_names: list[str] | None = None,
         state_field_types: list[str] | None = None) -> TTerm:
    """Construct a typed arch term.

    cases may be a list of TTerm objects (case TTerms) or strings (legacy).
    """
    if cases is None:
        cases = []
    # Accept either case TTerms or plain strings
    case_terms = []
    for c in cases:
        if isinstance(c, str):
            case_terms.append(string(c))
        else:
            # Assume it's a TTerm — store its inner value
            case_terms.append(c)
    fields = [
        field(_NAME, string(name)),
        field(_CASES, list_(case_terms)),
        field(_STEP_ENTER, just(string(step_enter)) if step_enter else nothing()),
        field(_STEP_EMIT, just(string(step_emit)) if step_emit else nothing()),
        field(_ALGEBRA_CELL, string(algebra_cell)),
        field(_OBSERVER_CONVERGENCE, string(observer_convergence)),
        field(_OBSERVER_LOSS, string(observer_loss)),
        field(_STEP_COMPUTE, string(step_compute)),
        field(_STATE_FIELD_NAMES, list_([string(n) for n in (state_field_names or [])])),
        field(_STATE_FIELD_TYPES, list_([string(t) for t in (state_field_types or [])])),
    ]
    return record(_ARCH, fields)


def morphism_call(prim_name: str, eq_str: str) -> "TTerm":
    """Hydra term: primitive partially applied to its equation argument.

    Returns apply(primitive(ua.lib.tensor.<prim_name>), string(eq_str)).
    The result has type ndarray -> ndarray -> ndarray and is ready to accept x and y.
    Used for step-2 end-to-end reduction proofs and step-4 TTerm emission.
    """
    from hydra.dsl.meta import phantoms as P
    return P.apply(P.primitive(Name(f"ua.lib.tensor.{prim_name}")), P.string(eq_str))


