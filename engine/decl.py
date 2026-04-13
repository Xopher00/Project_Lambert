"""
ast.py — AST nodes for the DSL.

Pure data declarations with no logic or external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SemiringDecl:
    name:     str
    contract: str               # dotted name: (compiled_eq, x, y, temp) -> tensor
    compiler: str | None = None # dotted name: equation string -> compiled form


@dataclass
class LegDecl:
    name:      str
    src_sort:  str
    tgt_sort:  str
    equation:  str
    semiring:  str | None         # None = bridge; '_default' = no using-clause
    op:        str | None = None  # dotted name; None = use semiring contract
    transform: str | None = None  # dotted name: (x, y) -> (x', y')
    compiler:  str | None = None  # dotted name: per-leg equation compiler override


@dataclass
class PathDecl:
    name: str
    legs: list[str]


@dataclass
class FanDecl:
    name:     str
    branches: list[str]   # leg/path names
    merge:    str = 'dict' # 'dict', 'meet', 'join', or dotted.name


@dataclass
class CaseDecl:
    name:      str
    recursive: int
    data:      int
    output:    int = 0
    cell:      str | None = None  # dotted name for per-case cell function


@dataclass
class FunctorDecl:
    name:  str
    cases: list[CaseDecl]
    cell:  str | None = None  # dotted name for cell function


@dataclass
class ArchDecl:
    """An architecture declaration containing algebra and/or coalgebra sub-blocks.

    Each sub-block declares functor cases and cell bindings. One arch block,
    one name, two functors.
    """
    name:              str
    algebra_cases:     list[CaseDecl] | None = None
    coalgebra_cases:   list[CaseDecl] | None = None
    algebra_cell:      str | None = None   # functor-level cell for algebra
    coalgebra_cell:    str | None = None   # functor-level cell for coalgebra


@dataclass
class DSLSource:
    semirings: list[SemiringDecl]
    sorts:     list[str]
    legs:      list[LegDecl]
    paths:     list[PathDecl]
    fans:      list[FanDecl]
    functors:  list[FunctorDecl]
    archs:     list[ArchDecl]       = field(default_factory=list)
