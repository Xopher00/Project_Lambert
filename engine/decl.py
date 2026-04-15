"""
AST node declarations for the engine DSL.

Pure data structures with no logic or external dependencies. Each dataclass
represents one syntactic construct in the DSL source language. The parser
(parser.py) creates these; the compiler (compiler.py) consumes them.

Declarations:
  SemiringDecl  — quantale V = (V, ⊗, k) with contract, compiler, arity
  SortDecl      — named type, optionally with structured fields
  MorphismDecl  — V-functor f: src_sort → tgt_sort via an equation string
  PathDecl      — sequential composition of morphisms with combinators
  FanDecl       — parallel fan-out with merge strategy
  CaseDecl      — one case of a recursive endofunctor (for arch blocks)
  ArchDecl      — endofunctor F declaration with algebra and coalgebra config
  DSLSource     — top-level container holding all declarations from one source block

Depends on: nothing (leaf module in the engine stack)
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
    accumulate_fields: list[str] | None = None  # field names for field-level accumulate, e.g. ['K', 'V']
    template_param: str | None = None    # parameter name for template morphisms (e.g. 'prefix')


@dataclass
class PathDecl:
    name:      str
    # Each token is one of: plain name ("q_proj"), augment bracket ("[kv]"),
    # or template instantiation ("ln[ln1]"). The compiler re-parses these with regex.
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
    iterate:   str | None = None        # payload sequence name for iteration (e.g. 'layers')



@dataclass
class ArchDecl:
    """An architecture declaration: one endofunctor F with dual evaluation.

    The `cases` list declares F — shared by both algebra (catamorphism) and
    coalgebra (anamorphism). Morphism bindings on cases serve both evaluations.
    `state:` and `step:` provide coalgebra-specific configuration (enter/emit).
    """
    name:              str
    cases:             list[CaseDecl] | None = None    # unified endofunctor F
    algebra_cases:     list[CaseDecl] | None = None    # legacy alias for cases (algebra: block)
    algebra_cell:      str | None = None   # functor-level cell override for algebra
    observer_convergence: str | None = None  # path name for convergence check
    observer_loss:        str | None = None  # path name for loss computation
    state_fields:         dict[str, str] | None = None  # coalgebra state shape: field_name -> type_name
    step_enter:           str | None = None  # morphism/path for coalgebra input binding
    step_emit:            str | None = None  # morphism/path for coalgebra output


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
