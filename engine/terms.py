"""Phantom-typed DSL constructors for engine architecture specifications.

Provides typed wrappers around Hydra's phantom DSL for constructing
engine morphism, path, fan, and arch terms with type safety.

Requires Python 3.12+ (hydra.dsl.meta.phantoms uses `type` syntax).
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
    list_,
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
_EQUATION = Name("equation")
_SRC_SORT = Name("srcSort")
_TGT_SORT = Name("tgtSort")
_ARITY = Name("arity")
_ACCUMULATE = Name("accumulate")
_MORPHISMS = Name("morphisms")
_RESIDUAL = Name("residual")
_NORMED = Name("normed")
_BRANCHES = Name("branches")
_MERGE = Name("merge")
_RECURSIVE = Name("recursive")
_DATA = Name("data")
_OUTPUT = Name("output")
_ALGEBRA_CASES = Name("algebraCases")
_OBSERVER_CONVERGENCE = Name("observerConvergence")
_OBSERVER_LOSS = Name("observerLoss")


def morphism(name: str, src: str, tgt: str, equation: str,
             arity: str = "binary", accumulate: str | None = None) -> TTerm:
    """Construct a typed morphism term."""
    fields = [
        field(_NAME, string(name)),
        field(_EQUATION, string(equation)),
        field(_SRC_SORT, string(src)),
        field(_TGT_SORT, string(tgt)),
        field(_ARITY, string(arity)),
    ]
    if accumulate:
        fields.append(field(_ACCUMULATE, string(accumulate)))
    return record(_MORPHISM, fields)


def path(name: str, morphisms: list[str],
         residual: bool = False, normed: str | None = None) -> TTerm:
    """Construct a typed path term."""
    fields = [
        field(_NAME, string(name)),
        field(_MORPHISMS, list_([string(m) for m in morphisms])),
        field(_RESIDUAL, boolean(residual)),
    ]
    if normed:
        fields.append(field(_NORMED, string(normed)))
    return record(_PATH, fields)


def fan(name: str, branches: list[str], merge: str = "dict") -> TTerm:
    """Construct a typed fan term."""
    return record(_FAN, [
        field(_NAME, string(name)),
        field(_BRANCHES, list_([string(b) for b in branches])),
        field(_MERGE, string(merge)),
    ])


def case(name: str, recursive: int, data: int, output: int = 0) -> TTerm:
    """Construct a typed case term."""
    return record(_CASE, [
        field(_NAME, string(name)),
        field(_RECURSIVE, int32(recursive)),
        field(_DATA, int32(data)),
        field(_OUTPUT, int32(output)),
    ])


def arch(name: str, cases: list[TTerm] | None = None,
         observer_convergence: str | None = None,
         observer_loss: str | None = None) -> TTerm:
    """Construct a typed arch term."""
    fields = [field(_NAME, string(name))]
    if cases is not None:
        fields.append(field(_ALGEBRA_CASES, list_(cases)))
    if observer_convergence:
        fields.append(field(_OBSERVER_CONVERGENCE, string(observer_convergence)))
    if observer_loss:
        fields.append(field(_OBSERVER_LOSS, string(observer_loss)))
    return record(_ARCH, fields)
