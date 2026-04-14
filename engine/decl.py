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
    arity:    str = 'binary'    # 'binary' | 'ternary'


@dataclass
class MorphismDecl:
    name:      str
    src_sort:  str
    tgt_sort:  str
    equation:  str
    semiring:  str | None         # None = bridge; '_default' = no using-clause
    op:        str | None = None  # dotted name; None = use semiring contract
    transform: str | None = None  # dotted name: (x, y) -> (x', y')
    compiler:  str | None = None  # dotted name: per-morphism equation compiler override
    arity:     str = 'binary'     # 'binary' | 'unary' | 'pointwise' | 'ternary'
    accumulate: str | None = None  # 'cat' | None — coalgebra state accumulation


@dataclass
class PathDecl:
    name:      str
    morphisms: list[str]
    residual:  bool = False
    normed:    str | None = None   # morphism name to apply as norm after path


@dataclass
class FanDecl:
    name:     str
    branches: list[str]   # morphism/path names
    merge:    str = 'dict' # 'dict', 'meet', 'join', or dotted.name


@dataclass
class CaseDecl:
    name:      str
    recursive: int
    data:      int
    output:    int = 0
    cell:      str | None = None       # dotted name for per-case cell function
    morphisms: list[str] | None = None  # DSL-derived cell: compose these morphisms



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
    observer_convergence: str | None = None  # path name for convergence check
    observer_loss:        str | None = None  # path name for loss computation


@dataclass
class SortDecl:
    """Sort declaration — optionally carries named fields for structured sorts."""
    name:   str
    fields: dict[str, str] | None = None  # None = opaque (today's behavior)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, SortDecl) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        if isinstance(other, str):
            return self.name < other
        if isinstance(other, SortDecl):
            return self.name < other.name
        return NotImplemented


@dataclass
class DSLSource:
    semirings: list[SemiringDecl]
    sorts:     list[SortDecl]
    morphisms: list[MorphismDecl]
    paths:     list[PathDecl]
    fans:      list[FanDecl]
    archs:     list[ArchDecl]       = field(default_factory=list)
