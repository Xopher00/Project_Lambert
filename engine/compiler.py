"""
Compiles DSL source text into executable architecture definitions.

Takes a parsed AST (from parser.py) plus a Python namespace and produces
an ArchDef containing compiled morphisms, paths, fans, and arch interpreters.
The namespace maps dotted names to Python objects (functions, modules).

Key phases:
  1. Resolve semirings — map names to contract/compiler/arity callables
  2. Compile morphisms — MorphismSpec → cached (x, y, temp) → result callables
  3. Compile fans — branch callables + merge strategy
  4. Compile paths — chain/augment/residual composition with sort validation
  5. Compile archs — functor + cell binding for algebra and coalgebra sides

Depends on: parser.py (AST), path_engine.py (morphism compilation, chain, fan),
            functor.py (Case, Functor), arch.py (ArchDef, ArchInterpreter, _ArchData)
"""

from __future__ import annotations

import re as _re
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

    for morph_decl in ast.morphisms:
        group = morphism_to_semiring[morph_decl.name]
        if morph_decl.op is not None:
            op_fn = _resolve(morph_decl.op, namespace)
        elif group in semiring_contract:
            op_fn = semiring_contract[group]
        else:
            raise ValueError(
                f"Morphism '{morph_decl.name}': no 'op' clause and semiring '{group}' "
                f"has no 'contract'. Add 'op <fn>' to the morphism or declare a "
                f"semiring contract."
            )
        transform_fn = _resolve(morph_decl.transform, namespace) if morph_decl.transform else lambda x, y: (x, y)
        if morph_decl.compiler is not None:
            eq_compiler = _resolve(morph_decl.compiler, namespace)
        elif group in semiring_compiler:
            eq_compiler = semiring_compiler[group]
        else:
            eq_compiler = lambda eq: eq

        spec = MorphismSpec(
            name=morph_decl.name, op=op_fn, equation=morph_decl.equation,
            src_sort=morph_decl.src_sort, tgt_sort=morph_decl.tgt_sort,
            equation_compiler=eq_compiler, transform=transform_fn,
            arity=morph_decl.arity if morph_decl.arity != 'binary' else semiring_arity.get(group, 'binary'),
            accumulate=morph_decl.accumulate,
            accumulate_fields=morph_decl.accumulate_fields,
        )
        morphism_specs[morph_decl.name] = spec
        equations[morph_decl.name] = morph_decl.equation
        compiled[morph_decl.name]  = compile_morphism(spec)

    return compiled, morphism_specs, equations


def _build_residual_wrapper(base_fn, norm_fn, has_residual):
    """Wrap a path callable with residual connection and/or normalization."""
    if has_residual and norm_fn is not None:
        return lambda x, y, temp, _b=base_fn, _n=norm_fn: \
            _n(_b(x, y, temp) + x, y, temp)
    elif has_residual:
        return lambda x, y, temp, _b=base_fn: \
            _b(x, y, temp) + x
    elif norm_fn is not None:
        return lambda x, y, temp, _b=base_fn, _n=norm_fn: \
            _n(_b(x, y, temp), y, temp)
    else:
        return base_fn


_TEMPLATE_INST = _re.compile(r'^(\w+)\[(\w+)\]$')


def _resolve_template_instance(
    token: str,
    templates: dict,
    compiled: dict,
    morphism_specs: dict,
    morphism_to_semiring: dict,
    equations: dict,
    namespace: dict,
    semiring_arity: dict,
):
    """Resolve a template instantiation like 'ln[ln1]'.

    Creates a curried morphism callable and registers it in compiled/morphism_specs.
    Returns the instance name (e.g. 'ln[ln1]') or None if token is not a template inst.
    """
    match = _TEMPLATE_INST.match(token)
    if not match:
        return None
    base_name = match.group(1)
    param_value = match.group(2)

    if base_name not in templates:
        return None  # not a template — might be a regular morphism

    inst_name = token  # e.g. 'ln[ln1]'
    if inst_name in compiled:
        return inst_name  # already instantiated

    template_decl = templates[base_name]
    param_name = template_decl.template_param  # e.g. 'prefix'
    base_spec = morphism_specs[base_name]

    # Create curried op that binds the parameter
    base_op = base_spec.op
    def curried_op(eq, *args, _op=base_op, _pn=param_name, _pv=param_value, **kwargs):
        kwargs[_pn] = _pv
        return _op(eq, *args, **kwargs)

    # Create instance spec (same as template but with curried op and instance name)
    inst_spec = MorphismSpec(
        name=inst_name,
        op=curried_op,
        equation=base_spec.equation,
        src_sort=base_spec.src_sort,
        tgt_sort=base_spec.tgt_sort,
        equation_compiler=base_spec.equation_compiler,
        transform=base_spec.transform,
        arity=base_spec.arity,
        accumulate=base_spec.accumulate,
        accumulate_fields=base_spec.accumulate_fields,
    )

    morphism_specs[inst_name] = inst_spec
    equations[inst_name] = base_spec.equation
    morphism_to_semiring[inst_name] = morphism_to_semiring[base_name]
    compiled[inst_name] = compile_morphism(inst_spec)

    return inst_name


def _compile_paths(
    ast: DSLSource,
    compiled: dict,
    morphism_specs: dict,
    morphism_to_semiring: dict,
    templates: dict,
    equations: dict,
    namespace: dict,
    semiring_arity: dict,
) -> dict[str, list[str]]:
    """Phase 4: Validate and compile paths.

    Mutates compiled in-place to add path callables.
    Returns path_morphisms dict.
    """
    path_morphisms: dict[str, list[str]] = {}
    for path in ast.paths:
        # Resolve template instantiations (e.g. 'ln[ln1]' -> curried morphism)
        for m in path.morphisms:
            if not (m.startswith('[') and m.endswith(']')):  # skip augment brackets
                _resolve_template_instance(
                    m, templates, compiled, morphism_specs,
                    morphism_to_semiring, equations, namespace, semiring_arity,
                )
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
            compiled[path.name] = _build_residual_wrapper(base, norm_fn, path.residual)
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
    for case_decl in case_list:
        if case_decl.cell == 'identity':
            case_cells[case_decl.name] = lambda payload, child_results, params, temp: \
                payload[0] if payload else (child_results[0] if child_results else None)
        elif case_decl.cell is not None:
            case_cells[case_decl.name] = _resolve(case_decl.cell, namespace)
        elif case_decl.morphisms is not None:
            for morph_name in case_decl.morphisms:
                if morph_name not in compiled:
                    raise ValueError(
                        f"{context_name} case '{case_decl.name}': "
                        f"morphism '{morph_name}' not found in compiled paths"
                    )
            path_fn = chain([compiled[morph_name] for morph_name in case_decl.morphisms]) \
                if len(case_decl.morphisms) > 1 else compiled[case_decl.morphisms[0]]
            case_cells[case_decl.name] = _make_morphism_derived_cell(case_decl.name, path_fn)
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
    for arch_decl in ast.archs:
        data = _ArchData()
        for mode, case_list, functor_cell in [
            ('algebra',   arch_decl.algebra_cases,   arch_decl.algebra_cell),
            ('coalgebra', arch_decl.coalgebra_cases,  arch_decl.coalgebra_cell),
        ]:
            if case_list is None:
                continue
            cases = [Case(case_decl.name, case_decl.recursive, case_decl.data, case_decl.output) for case_decl in case_list]
            functor = Functor(cases)
            cell = None
            if functor_cell is not None:
                cell = _resolve(functor_cell, namespace)
            else:
                case_cells = _resolve_case_cells(
                    case_list, f"Arch '{arch_decl.name}' {mode}", compiled, namespace
                )
                if case_cells:
                    missing = [case_decl.name for case_decl in case_list if case_decl.name not in case_cells]
                    if missing:
                        raise ValueError(
                            f"Arch '{arch_decl.name}' {mode}: per-case cell binding is "
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
            name: (spec.accumulate, spec.accumulate_fields)
            for name, spec in morphism_specs.items()
            if spec.accumulate is not None
        }
        if acc_morphisms:
            data.accumulate_legs = acc_morphisms

        # 7d. Detect algebra iterate groups
        if arch_decl.algebra_cases:
            iterate_groups = {}
            base_case = None
            epilogue_cases = []
            in_iterate = False
            for case_decl in arch_decl.algebra_cases:
                if case_decl.iterate is not None:
                    in_iterate = True
                    group_name = case_decl.iterate
                    if group_name not in iterate_groups:
                        iterate_groups[group_name] = []
                    iterate_groups[group_name].append(case_decl.name)
                elif case_decl.recursive == 0 and not in_iterate:
                    base_case = case_decl.name
                else:
                    if in_iterate:
                        epilogue_cases.append(case_decl.name)
                    # non-iterate, non-base cases before any iterate block
                    # are just regular cases (no special handling needed)
            if iterate_groups:
                data.iterate_groups = iterate_groups
                data.iterate_base = base_case
                data.iterate_epilogue = epilogue_cases

        # 7c. Resolve observer paths
        if arch_decl.observer_convergence:
            if arch_decl.observer_convergence not in compiled:
                raise ValueError(
                    f"Arch '{arch_decl.name}': observer convergence path "
                    f"'{arch_decl.observer_convergence}' not found"
                )
            data.observer_convergence = compiled[arch_decl.observer_convergence]
        if arch_decl.observer_loss:
            if arch_decl.observer_loss not in compiled:
                raise ValueError(
                    f"Arch '{arch_decl.name}': observer loss path "
                    f"'{arch_decl.observer_loss}' not found"
                )
            data.observer_loss = compiled[arch_decl.observer_loss]
        compiled_archs[arch_decl.name] = data
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
        for morph_decl in ast.morphisms:
            for sort_name, label in [(morph_decl.src_sort, 'src'), (morph_decl.tgt_sort, 'tgt')]:
                if sort_name not in sort_defs:
                    warnings.warn(
                        f"Morphism '{morph_decl.name}': {label} sort '{sort_name}' "
                        f"not in declared sorts",
                        stacklevel=2,
                    )

    # Resolve semirings
    semiring_contract: dict[str, Callable] = {
        semiring_decl.name: _resolve(semiring_decl.contract, namespace) for semiring_decl in ast.semirings
    }
    semiring_compiler: dict[str, Callable] = {
        semiring_decl.name: _resolve(semiring_decl.compiler, namespace)
        for semiring_decl in ast.semirings if semiring_decl.compiler is not None
    }
    semiring_arity: dict[str, str] = {semiring_decl.name: semiring_decl.arity for semiring_decl in ast.semirings}

    # Assign semiring groups (semiring block is optional if all morphisms have explicit ops)
    sole = ast.semirings[0].name if len(ast.semirings) == 1 else None
    morphism_to_semiring: dict[str, str] = {}
    for morph_decl in ast.morphisms:
        if morph_decl.semiring is None:
            morphism_to_semiring[morph_decl.name] = '_bridge'
        elif morph_decl.semiring == '_default':
            morphism_to_semiring[morph_decl.name] = sole if sole else '_default'
        elif morph_decl.semiring not in semiring_contract:
            raise ValueError(
                f"Morphism '{morph_decl.name}' references undeclared semiring '{morph_decl.semiring}'"
            )
        else:
            morphism_to_semiring[morph_decl.name] = morph_decl.semiring
    compiled, morphism_specs, equations = _compile_morphisms(
        ast, morphism_to_semiring, semiring_contract, semiring_compiler, semiring_arity, namespace
    )

    # Build template registry for parameterized morphisms
    templates = {
        morph_decl.name: morph_decl
        for morph_decl in ast.morphisms
        if morph_decl.template_param is not None
    }

    _compile_fans(ast, compiled, namespace)
    path_morphisms = _compile_paths(
        ast, compiled, morphism_specs, morphism_to_semiring,
        templates, equations, namespace, semiring_arity,
    )
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
