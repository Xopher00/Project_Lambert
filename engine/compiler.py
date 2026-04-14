"""
compiler.py — DSL compiler.

Compiles a DSLSource AST into an ArchDef: resolves namespace names,
compiles morphisms/paths/fans/functors/archs into callables.
"""

from __future__ import annotations

from typing import Callable
from functools import reduce

from .path_engine import MorphismSpec, compile_morphism, chain, chain_with_augments, fan, check_sorts, explain, trace
from .functor import Case, Functor
from .decl import DSLSource, MorphismDecl
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

def _make_morphism_derived_cell(case_name: str, path_fn: Callable) -> Callable:
    """Derive a per-case cell from a morphism composition.

    The derived cell applies path_fn(child_results[0], payload[0], temp).
    Signature matches per-case cell: (payload, child_results, params, temp) -> result
    """
    def derived(payload, child_results, params, temp,
                _fn=path_fn, _name=case_name):
        x = child_results[0] if child_results else None
        y = payload[0] if payload else None
        return _fn(x, y, temp)
    return derived


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
# Compiler phases
# ---------------------------------------------------------------------------

def _compile_morphisms(
    ast: DSLSource,
    morphism_to_semiring: dict,
    semiring_contract: dict,
    semiring_compiler: dict,
    semiring_arity: dict,
    namespace: dict,
):
    """Phase 3: Build MorphismSpecs and compile each morphism.

    Returns (compiled, morphism_specs, equations).
    """
    compiled:       dict[str, Callable]     = {}
    morphism_specs: dict[str, MorphismSpec] = {}
    equations:      dict[str, str]          = {}

    for md in ast.morphisms:
        group = morphism_to_semiring[md.name]
        if md.op is not None:
            op_fn = _resolve(md.op, namespace)
        elif group in semiring_contract:
            op_fn = semiring_contract[group]
        else:
            raise ValueError(
                f"Morphism '{md.name}': no 'op' clause and semiring '{group}' "
                f"has no 'contract'. Add 'op <fn>' to the morphism or declare a "
                f"semiring contract."
            )
        tf = _resolve(md.transform, namespace) if md.transform else lambda x, y: (x, y)
        if md.compiler is not None:
            eq_compiler = _resolve(md.compiler, namespace)
        elif group in semiring_compiler:
            eq_compiler = semiring_compiler[group]
        else:
            eq_compiler = lambda eq: eq

        spec = MorphismSpec(
            name=md.name, op=op_fn, equation=md.equation,
            src_sort=md.src_sort, tgt_sort=md.tgt_sort,
            equation_compiler=eq_compiler, transform=tf,
            arity=md.arity if md.arity != 'binary' else semiring_arity.get(group, 'binary'),
            accumulate=md.accumulate,
        )
        morphism_specs[md.name] = spec
        equations[md.name] = md.equation
        compiled[md.name]  = compile_morphism(spec)

    return compiled, morphism_specs, equations


def _compile_paths(
    ast: DSLSource,
    compiled: dict,
    morphism_specs: dict,
    morphism_to_semiring: dict,
) -> dict[str, list[str]]:
    """Phase 4: Validate and compile paths.

    Mutates compiled in-place to add path callables.
    Returns path_morphisms dict.
    """
    path_morphisms: dict[str, list[str]] = {}
    for path in ast.paths:
        # Strip bracketed augment tokens for sort/semiring validation
        sort_morphisms = [m for m in path.morphisms
                         if not (m.startswith('[') and m.endswith(']'))]
        # Sort validation
        err = check_sorts(morphism_specs, sort_morphisms)
        if err:
            raise TypeError(f"Path '{path.name}': {err}")
        # Semiring validation
        used = {morphism_to_semiring[m] for m in sort_morphisms}
        if len(used) > 1 and '_bridge' not in used:
            raise ValueError(
                f"Path '{path.name}' spans multiple semirings: {used}. "
                f"Use a bridge morphism to cross semiring boundaries."
            )
        # Build chain, handling augment steps
        has_augments = any(m.startswith('[') and m.endswith(']')
                           for m in path.morphisms)
        if has_augments:
            steps = []
            for m in path.morphisms:
                if m.startswith('[') and m.endswith(']'):
                    fan_name = m[1:-1]
                    if fan_name not in compiled:
                        raise ValueError(
                            f"Path '{path.name}': augment target '{fan_name}' "
                            f"is not a declared morphism, path, or fan"
                        )
                    steps.append(('augment', compiled[fan_name]))
                else:
                    if m not in compiled:
                        raise ValueError(
                            f"Path '{path.name}': morphism '{m}' not found"
                        )
                    steps.append(('step', compiled[m]))
            base = chain_with_augments(steps)
        else:
            base = chain([compiled[m] for m in path.morphisms])
        if path.residual or path.normed:
            if path.normed and path.normed not in compiled:
                raise ValueError(
                    f"Path '{path.name}': normed morphism '{path.normed}' "
                    f"is not a declared morphism or path"
                )
            norm_fn = compiled[path.normed] if path.normed else None
            if path.residual and norm_fn is not None:
                compiled[path.name] = lambda x, y, temp, _b=base, _n=norm_fn: \
                    _n(_b(x, y, temp) + x, y, temp)
            elif path.residual:
                compiled[path.name] = lambda x, y, temp, _b=base: \
                    _b(x, y, temp) + x
            else:
                compiled[path.name] = lambda x, y, temp, _b=base, _n=norm_fn: \
                    _n(_b(x, y, temp), y, temp)
        else:
            compiled[path.name] = base
        path_morphisms[path.name] = sort_morphisms
    return path_morphisms


def _compile_fans(ast: DSLSource, compiled: dict, namespace: dict) -> None:
    """Phase 5: Compile fans.

    Mutates compiled in-place.
    """
    for f in ast.fans:
        branch_callables = {}
        for b in f.branches:
            if b not in compiled:
                raise ValueError(
                    f"Fan '{f.name}': branch '{b}' is not a declared morphism or path"
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


def _resolve_case_cells(
    case_list,
    context_name: str,
    compiled: dict,
    namespace: dict,
) -> dict[str, Callable]:
    """Resolve per-case cells (explicit cell= or morphisms= derived).

    Returns case_cells dict.
    """
    case_cells = {}
    for c in case_list:
        if c.cell == 'identity':
            case_cells[c.name] = lambda payload, child_results, params, temp: \
                payload[0] if payload else (child_results[0] if child_results else None)
        elif c.cell is not None:
            case_cells[c.name] = _resolve(c.cell, namespace)
        elif c.morphisms is not None:
            for mn in c.morphisms:
                if mn not in compiled:
                    raise ValueError(
                        f"{context_name} case '{c.name}': "
                        f"morphism '{mn}' not found in compiled paths"
                    )
            path_fn = chain([compiled[mn] for mn in c.morphisms]) \
                if len(c.morphisms) > 1 else compiled[c.morphisms[0]]
            case_cells[c.name] = _make_morphism_derived_cell(c.name, path_fn)
    return case_cells



def _compile_archs(
    ast: DSLSource,
    compiled: dict,
    morphism_specs: dict,
    namespace: dict,
) -> dict[str, _ArchData]:
    """Phase 7: Compile arch declarations.

    Returns compiled_archs dict.
    """
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
                case_cells = _resolve_case_cells(
                    case_list, f"Arch '{ad.name}' {mode}", compiled, namespace
                )
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
        # 7b. Build accumulate_morphisms mapping
        acc_morphisms = {
            name: spec.accumulate
            for name, spec in morphism_specs.items()
            if spec.accumulate is not None
        }
        if acc_morphisms:
            data.accumulate_legs = acc_morphisms

        # 7c. Resolve observer paths
        if ad.observer_convergence:
            if ad.observer_convergence not in compiled:
                raise ValueError(
                    f"Arch '{ad.name}': observer convergence path "
                    f"'{ad.observer_convergence}' not found"
                )
            data.observer_convergence = compiled[ad.observer_convergence]
        if ad.observer_loss:
            if ad.observer_loss not in compiled:
                raise ValueError(
                    f"Arch '{ad.name}': observer loss path "
                    f"'{ad.observer_loss}' not found"
                )
            data.observer_loss = compiled[ad.observer_loss]
        compiled_archs[ad.name] = data
    return compiled_archs


# ---------------------------------------------------------------------------
# Top-level compiler
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

    sort_defs = {s.name: s for s in ast.sorts}

    # Warn about morphisms referencing undeclared sorts (if sorts were declared)
    if sort_defs:
        import warnings
        for md in ast.morphisms:
            for sort_name, label in [(md.src_sort, 'src'), (md.tgt_sort, 'tgt')]:
                if sort_name not in sort_defs:
                    warnings.warn(
                        f"Morphism '{md.name}': {label} sort '{sort_name}' "
                        f"not in declared sorts",
                        stacklevel=2,
                    )

    # Resolve semirings
    semiring_contract: dict[str, Callable] = {
        sr.name: _resolve(sr.contract, namespace) for sr in ast.semirings
    }
    semiring_compiler: dict[str, Callable] = {
        sr.name: _resolve(sr.compiler, namespace)
        for sr in ast.semirings if sr.compiler is not None
    }
    semiring_arity: dict[str, str] = {sr.name: sr.arity for sr in ast.semirings}

    # Assign semiring groups
    sole = ast.semirings[0].name if len(ast.semirings) == 1 else None
    morphism_to_semiring: dict[str, str] = {}
    for md in ast.morphisms:
        if md.semiring is None:
            morphism_to_semiring[md.name] = '_bridge'
        elif md.semiring == '_default':
            morphism_to_semiring[md.name] = sole if sole else '_default'
        elif md.semiring not in semiring_contract:
            raise ValueError(
                f"Morphism '{md.name}' references undeclared semiring '{md.semiring}'"
            )
        else:
            morphism_to_semiring[md.name] = md.semiring
    compiled, morphism_specs, equations = _compile_morphisms(
        ast, morphism_to_semiring, semiring_contract, semiring_compiler, semiring_arity, namespace
    )
    _compile_fans(ast, compiled, namespace)
    path_morphisms = _compile_paths(ast, compiled, morphism_specs, morphism_to_semiring)
    compiled_archs = _compile_archs(ast, compiled, morphism_specs, namespace)

    return ArchDef(
        paths              = compiled,
        morphism_semiring  = morphism_to_semiring,
        _morphism_specs    = morphism_specs,
        _equations         = equations,
        _path_morphisms    = path_morphisms,
        _archs             = compiled_archs,
        sort_defs          = sort_defs,
    )
