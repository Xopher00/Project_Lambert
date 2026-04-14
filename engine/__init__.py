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
from engine.runtime import MorphismSpec, compile_morphism, chain, fan, check_sorts
from engine.decl import (
    SemiringDecl, MorphismDecl, PathDecl, FanDecl, CaseDecl,
    ArchDecl, DSLSource, SortDecl,
)
from engine.parser import parse
from engine.arch import ArchDef, ArchInterpreter
from engine.compiler import compile

from pathlib import Path as _Path

def load(path, namespace: dict) -> ArchDef:
    """Load and compile a .ua file."""
    return compile(_Path(path).read_text(), namespace)

from engine.sorts import (
    sort_to_type, sort_types_from_defs,
    morphism_to_term, path_to_term, fan_to_term,
    functor_to_union_type, arch_to_term,
    ndarray_coder, bundle_coder, temp_coder, equation_coder,
)
from engine.primitives import (
    register_tensor_primitives, build_engine_graph, qname,
)
