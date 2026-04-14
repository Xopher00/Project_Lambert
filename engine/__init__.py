"""
Engine DSL — compile declarative architecture specifications into executable interpreters.

Public API:
  compile(source, namespace) → ArchDef    compile DSL text + Python namespace into an architecture
  parse(source) → DSLSource              parse DSL text into an AST (for inspection or modification)
  ArchDef                                compiled architecture with explain/trace/loss/interpreter methods

The engine layer sits above the core stack (algebra → activations → tensor → fixpoint)
and provides a domain-specific language for expressing tensor architectures as compositions
of V-functors over arbitrary semirings.
"""

from engine.functor import Case, Functor, CoalgResult, Interpreter
from engine.path_engine import MorphismSpec, LegSpec, compile_morphism, chain, fan, check_sorts
from engine.decl import (
    SemiringDecl, MorphismDecl, PathDecl, FanDecl, CaseDecl,
    ArchDecl, DSLSource, SortDecl,
)
from engine.parser import parse
from engine.arch import ArchDef, ArchInterpreter
from engine.compiler import compile
