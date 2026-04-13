"""
compiler.py — DSL compiler.

Compiles a DSLSource AST into an ArchDef: resolves namespace names,
compiles legs/paths/fans/functors/archs into callables.
"""

from __future__ import annotations

from typing import Callable
from functools import reduce

from .path_engine import LegSpec, compile_leg, chain, fan, check_sorts, explain, trace
from .functor import Case, Functor
from .decl import DSLSource, LegDecl
from .arch import ArchDef, _ArchData
from .parser import parse


# ---------------------------------------------------------------------------
# Name resolver
# ---------------------------------------------------------------------------

def _resolve(dotted: str, namespace: dict) -> object:
    parts = dotted.split('.')
    obj = namespace.get(parts[0])
    if obj is None:
        raise NameError(f"Name {parts[0]!r} not found in provided namespace")
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# Cell dispatch generator
# ---------------------------------------------------------------------------

def _make_dispatch_cell(case_cells: dict[str, Callable]) -> Callable:
    """Generate a dispatch cell from per-case cell bindings.

    Each per-case cell has signature:
        (payload, child_results, params, temp) -> result

    The dispatch cell matches the Interpreter's algebra cell signature:
        (case_name, payload, child_results, params, temp) -> result
    """
    def dispatch(case_name, payload, child_results, params, temp,
                 _cells=case_cells):
        if case_name not in _cells:
            raise KeyError(f"No cell bound for case '{case_name}'")
        return _cells[case_name](payload, child_results, params, temp)
    return dispatch


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile(source: str | DSLSource, namespace: dict) -> ArchDef:
    """
    Compile DSL source into an ArchDef.

    Parameters
    ----------
    source    : str or DSLSource
    namespace : dict  — Python namespace for resolving dotted names
                        e.g. {'numpy': numpy, 'ops': ops_module}
    """
    ast = parse(source) if isinstance(source, str) else source

    # 1. Resolve semiring contracts and compilers
    semiring_contract: dict[str, Callable] = {
        sr.name: _resolve(sr.contract, namespace)
        for sr in ast.semirings
    }
    semiring_compiler: dict[str, Callable] = {
        sr.name: _resolve(sr.compiler, namespace)
        for sr in ast.semirings if sr.compiler is not None
    }

    # 2. Determine effective semiring group for each leg.
    sole = ast.semirings[0].name if len(ast.semirings) == 1 else None

    def _group(leg: LegDecl) -> str:
        if leg.semiring is None:
            return '_bridge'
        if leg.semiring == '_default':
            return sole if sole else '_default'
        if leg.semiring not in semiring_contract:
            raise ValueError(
                f"Leg '{leg.name}' references undeclared semiring '{leg.semiring}'"
            )
        return leg.semiring

    leg_to_semiring: dict[str, str] = {}
    for leg in ast.legs:
        leg_to_semiring[leg.name] = _group(leg)

    # 3. Build LegSpecs and compile each leg
    compiled:   dict[str, Callable]  = {}
    leg_specs:  dict[str, LegSpec]   = {}
    equations:  dict[str, str]       = {}

    for ld in ast.legs:
        group = leg_to_semiring[ld.name]
        if ld.op is not None:
            op_fn = _resolve(ld.op, namespace)
        elif group in semiring_contract:
            op_fn = semiring_contract[group]
        else:
            raise ValueError(
                f"Leg '{ld.name}': no 'op' clause and semiring '{group}' "
                f"has no 'contract'. Add 'op <fn>' to the leg or declare a "
                f"semiring contract."
            )
        tf = _resolve(ld.transform, namespace) if ld.transform else lambda x, y: (x, y)
        if ld.compiler is not None:
            eq_compiler = _resolve(ld.compiler, namespace)
        elif group in semiring_compiler:
            eq_compiler = semiring_compiler[group]
        else:
            eq_compiler = lambda eq: eq

        spec = LegSpec(
            name=ld.name, op=op_fn, equation=ld.equation,
            src_sort=ld.src_sort, tgt_sort=ld.tgt_sort,
            equation_compiler=eq_compiler, transform=tf,
        )
        leg_specs[ld.name] = spec
        equations[ld.name] = ld.equation
        compiled[ld.name]  = compile_leg(spec)

    # 4. Validate and compile paths
    path_legs: dict[str, list[str]] = {}
    for path in ast.paths:
        # Sort validation
        err = check_sorts(leg_specs, path.legs)
        if err:
            raise TypeError(f"Path '{path.name}': {err}")
        # Semiring validation
        used = {leg_to_semiring[l] for l in path.legs}
        if len(used) > 1 and '_bridge' not in used:
            raise ValueError(
                f"Path '{path.name}' spans multiple semirings: {used}. "
                f"Use a bridge leg to cross semiring boundaries."
            )
        compiled[path.name] = chain([compiled[l] for l in path.legs])
        path_legs[path.name] = path.legs

    # 5. Compile fans
    for f in ast.fans:
        branch_callables = {}
        for b in f.branches:
            if b not in compiled:
                raise ValueError(
                    f"Fan '{f.name}': branch '{b}' is not a declared leg or path"
                )
            branch_callables[b] = compiled[b]
        if f.merge == 'dict':
            merge_fn = lambda results: results
        elif f.merge == 'meet':
            import numpy as _np
            merge_fn = lambda results, _np=_np: reduce(_np.minimum, results.values())
        elif f.merge == 'join':
            import numpy as _np
            merge_fn = lambda results, _np=_np: reduce(_np.maximum, results.values())
        else:
            merge_fn = _resolve(f.merge, namespace)
        compiled[f.name] = fan(branch_callables, merge_fn)

    # 6. Compile functors and resolve cell bindings
    compiled_functors: dict[str, Functor] = {}
    compiled_cells:    dict[str, Callable] = {}
    for fd in ast.functors:
        cases = [Case(c.name, c.recursive, c.data, c.output) for c in fd.cases]
        compiled_functors[fd.name] = Functor(cases)
        if fd.cell is not None:
            compiled_cells[fd.name] = _resolve(fd.cell, namespace)
        else:
            case_cells = {}
            for c in fd.cases:
                if c.cell is not None:
                    case_cells[c.name] = _resolve(c.cell, namespace)
            if case_cells:
                missing = [c.name for c in fd.cases if c.name not in case_cells]
                if missing:
                    raise ValueError(
                        f"Functor '{fd.name}': per-case cell binding is incomplete. "
                        f"Missing cells for: {', '.join(missing)}"
                    )
                compiled_cells[fd.name] = _make_dispatch_cell(case_cells)

    # 7. Compile arch declarations
    compiled_archs: dict[str, _ArchData] = {}
    for ad in ast.archs:
        data = _ArchData()
        for mode, case_list, functor_cell in [
            ('algebra',   ad.algebra_cases,   ad.algebra_cell),
            ('coalgebra', ad.coalgebra_cases,  ad.coalgebra_cell),
        ]:
            if case_list is None:
                continue
            cases = [Case(c.name, c.recursive, c.data, c.output) for c in case_list]
            functor = Functor(cases)
            cell = None
            if functor_cell is not None:
                cell = _resolve(functor_cell, namespace)
            else:
                case_cells = {}
                for c in case_list:
                    if c.cell is not None:
                        case_cells[c.name] = _resolve(c.cell, namespace)
                if case_cells:
                    missing = [c.name for c in case_list if c.name not in case_cells]
                    if missing:
                        raise ValueError(
                            f"Arch '{ad.name}' {mode}: per-case cell binding is "
                            f"incomplete. Missing cells for: {', '.join(missing)}"
                        )
                    cell = _make_dispatch_cell(case_cells)
            if mode == 'algebra':
                data.algebra_functor = functor
                data.algebra_cell = cell
            else:
                data.coalgebra_functor = functor
                data.coalgebra_cell = cell
        compiled_archs[ad.name] = data

    return ArchDef(
        paths        = compiled,
        leg_semiring = leg_to_semiring,
        functors     = compiled_functors,
        _leg_specs   = leg_specs,
        _equations   = equations,
        _path_legs   = path_legs,
        _cells       = compiled_cells,
        _archs       = compiled_archs,
    )
