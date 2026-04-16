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

__all__ = [
    # compilation
    "compile", "parse", "load",
    # arch
    "ArchDef", "ArchInterpreter",
    # functor
    "Case", "Functor", "UnfoldStep", "NO_OUTPUT", "Interpreter",
    # runtime
    "MorphismSpec", "compile_morphism", "chain", "fan",
    # decl nodes
    "SemiringDecl", "DSLSource", "SortDecl",
]

# public API
from engine.functor import Case, Functor, UnfoldStep, NO_OUTPUT, Interpreter
from engine.runtime import MorphismSpec, compile_morphism, chain, fan
from engine.parser import (
    SemiringDecl, DSLSource, SortDecl,
)
from engine.parser import parse
from engine.arch import ArchDef, ArchInterpreter
from engine.compiler import compile

from pathlib import Path as _Path

def load(path, namespace: dict) -> ArchDef:
    """Load and compile a .ua file."""
    return compile(_Path(path).read_text(), namespace)

# internal — Hydra bridge
from engine.sorts import (
    sort_to_type, sort_types_from_defs,
    ndarray_coder, bundle_coder,
    morphism_type, path_type, fan_type, case_type, arch_type,
)
from engine.sorts import morphism_to_term
from engine.terms import path_to_term, fan_to_term, arch_to_term
from engine.primitives import (
    register_primitives, build_engine_graph, qname,
)
