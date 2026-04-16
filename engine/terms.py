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
_MORPHISMS = Name("morphisms")
_RESIDUAL = Name("residual")
_BRANCHES = Name("branches")
_RECURSIVE = Name("recursive")
_DATA = Name("data")
_CASES = Name("cases")
_STEP_ENTER = Name("stepEnter")
_STEP_EMIT = Name("stepEmit")


def morphism(name: str, src: str, tgt: str, arity: str = "binary",
             template_params: list[str] | None = None) -> TTerm:
    """Construct a typed morphism term."""
    fields = [
        field(_NAME, string(name)),
        field(_SRC_SORT, string(src)),
        field(_TGT_SORT, string(tgt)),
        field(_ARITY, string(arity)),
        field(_TEMPLATE_PARAMS, list_([string(p) for p in (template_params or [])])),
    ]
    return record(_MORPHISM, fields)


def path(name: str, morphisms: list[str], residual: bool = False) -> TTerm:
    """Construct a typed path term."""
    return record(_PATH, [
        field(_NAME, string(name)),
        field(_MORPHISMS, list_([string(m) for m in morphisms])),
        field(_RESIDUAL, boolean(residual)),
    ])


def fan(name: str, branches: list[str]) -> TTerm:
    """Construct a typed fan term."""
    return record(_FAN, [
        field(_NAME, string(name)),
        field(_BRANCHES, list_([string(b) for b in branches])),
    ])


def case(name: str, recursive: int, data: int) -> TTerm:
    """Construct a typed case term."""
    return record(_CASE, [
        field(_NAME, string(name)),
        field(_RECURSIVE, int32(recursive)),
        field(_DATA, int32(data)),
    ])


def arch(name: str, cases: list[str] | None = None,
         step_enter: str | None = None,
         step_emit: str | None = None) -> TTerm:
    """Construct a typed arch term."""
    fields = [
        field(_NAME, string(name)),
        field(_CASES, list_([string(n) for n in (cases or [])])),
        field(_STEP_ENTER, just(string(step_enter)) if step_enter else nothing()),
        field(_STEP_EMIT, just(string(step_emit)) if step_emit else nothing()),
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


def path_to_term(name: str, morphism_names: list[str],
                 residual: bool = False) -> "TTerm":
    return path(name, morphism_names, residual).value


def fan_to_term(name: str, branches: list[str]) -> "TTerm":
    return fan(name, branches).value


def arch_to_term(name: str, cases=None,
                 step_enter: str | None = None,
                 step_emit: str | None = None) -> "TTerm":
    case_names = [c['name'] if isinstance(c, dict) else c.name for c in cases] if cases is not None else []
    return arch(name, case_names, step_enter, step_emit).value
