"""
dsl.py — Backward-compatibility shim.

All DSL functionality has been split into:
  - engine.decl      — AST node dataclasses
  - engine.parser    — parse() function
  - engine.compiler  — compile() function
  - engine.arch      — ArchDef, ArchInterpreter
"""

# Re-export everything so `from engine.dsl import ...` still works.
from .decl import (                                       # noqa: F401
    SemiringDecl, LegDecl, PathDecl, FanDecl,
    CaseDecl, FunctorDecl, ArchDecl, DSLSource,
)
from .parser import parse                                 # noqa: F401
from .arch import ArchDef, ArchInterpreter                # noqa: F401
from .compiler import compile                             # noqa: F401
