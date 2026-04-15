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

Depends on: parser.py (AST), runtime.py (morphism compilation, chain, fan),
            functor.py (Case, Functor), arch.py (ArchDef, ArchInterpreter, _ArchData)

References
----------
Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
theory of all architectures. ICML 2024.  cite{gavranovic2024b}

Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
PODS 2007, pp. 31–40.  cite{green2007}

Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
*Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

Domingos, P. (2025). Tensor logic: The language of AI.
arXiv:2510.12269.  cite{domingos2025}
"""

from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Callable
from functools import reduce

from .runtime import MorphismSpec, compile_morphism, chain, chain_with_augments, fan, check_sorts
from .functor import Case, Functor
from .decl import DSLSource
from .arch import ArchDef, _ArchData
from .parser import parse
import warnings

from .sorts import morphism_to_term, path_to_term, fan_to_term, functor_to_union_type, arch_to_term, sort_to_type, sort_types_from_defs, resolve as _resolve
from .primitives import register_tensor_primitives


# ---------------------------------------------------------------------------
# Cell dispatch generator
# ---------------------------------------------------------------------------

def _make_morphism_derived_cell(case_name: str, path_fn: Callable) -> Callable:
    """Derive a per-case cell from a morphism composition.

    The derived cell applies path_fn(child_results[0], payload[0], temp).
    Signature matches per-case cell: (payload, child_results, params, temp) -> result
    """
    def derived(payload, child_results, _params, temp,
                _fn=path_fn):
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

def _build_step_cell(enter_fn, emit_fn, case_morphism_fn, case_name):
    """Generate a coalgebra cell from step declarations.

    The generated cell reuses the algebra's morphism composition for the
    main computation, wrapping it with enter (input binding) and emit (output).
    """
    from .functor import UnfoldStep

    def cell(state, token, params, temp,
             _enter=enter_fn, _emit=emit_fn, _compute=case_morphism_fn,
             _case=case_name):
        # 1. Enter: bind input token using state as context
        x = _enter(token, state, temp) if _enter is not None else token

        # 2. Main computation: same morphisms as algebra
        if _compute is not None:
            x = _compute(x, state, temp)

        # 3. Build next state (copy + accumulate updates)
        next_state = dict(state) if isinstance(state, dict) else state

        # 4. Emit or silent
        if _emit is not None:
            output = _emit(x, state, temp)
            return UnfoldStep(_case, [x], [next_state], output=output)
        else:
            return UnfoldStep(_case, [x], [next_state])

    return cell


def _compile_morphisms(
    ast: DSLSource,
    morphism_to_semiring: dict,
    semiring_contract: dict,
    semiring_compiler: dict,
    semiring_arity: dict,
    namespace: dict,
):
    """Build MorphismSpecs and compile each morphism.

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

    morphism_terms = {name: morphism_to_term(spec) for name, spec in morphism_specs.items()}
    return compiled, morphism_specs, equations, morphism_terms


def _build_residual_wrapper(base_fn, norm_fn, has_residual):
    """Wrap a path callable with residual connection and/or normalization.

    The residual connection implements the universal map from a coproduct: f(x) + x.

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """
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
    _namespace: dict,
    _semiring_arity: dict,
):
    """Resolve a template instantiation like 'ln[ln1]'.

    Creates a curried morphism callable and registers it in compiled/morphism_specs.
    Returns the instance name (e.g. 'ln[ln1]') or None if token is not a template inst.

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
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
    sort_types: dict | None = None,
) -> dict[str, list[str]]:
    """Validate and compile paths.

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
                         if not (m.startswith('[') and m.endswith(']'))
                         and m in morphism_specs]
        # Sort validation
        err = check_sorts(morphism_specs, sort_morphisms, sort_types)
        if err:
            raise TypeError(f"Path '{path.name}': {err}")
        # Semiring validation
        used = {morphism_to_semiring[m] for m in sort_morphisms if m in morphism_to_semiring}
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
    """Compile fans.

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
            case_cells[case_decl.name] = lambda payload, child_results, _params, _temp: \
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



def _build_functor_and_cells(
    arch_decl,
    compiled: dict,
    namespace: dict,
) -> tuple["Functor", Callable | None]:
    """Build the endofunctor F and resolve the algebra cell for an arch.

    Handles functor construction from cases and resolves per-case cells or
    the default algebra_cell.  Returns (functor, alg_cell).

    Raises ValueError for semantic violations:
    - case must not specify both 'cell' and 'morphisms'
    """
    case_list = arch_decl.cases or arch_decl.algebra_cases or []

    # Semantic validation: cell= and morphisms= are mutually exclusive
    for cd in case_list:
        if cd.cell is not None and cd.morphisms is not None:
            raise ValueError(
                f"Arch '{arch_decl.name}' case '{cd.name}': "
                f"cannot specify both 'cell' and 'morphisms'"
            )

    cases = [Case(cd.name, cd.recursive, cd.data, cd.output) for cd in case_list]
    functor = Functor(cases)

    alg_cell = None
    if arch_decl.algebra_cell is not None:
        alg_cell = _resolve(arch_decl.algebra_cell, namespace)
    else:
        case_cells = _resolve_case_cells(
            case_list, f"Arch '{arch_decl.name}' algebra", compiled, namespace
        )
        if case_cells:
            missing = [cd.name for cd in case_list if cd.name not in case_cells]
            if missing:
                raise ValueError(
                    f"Arch '{arch_decl.name}' algebra: per-case cell binding is "
                    f"incomplete. Missing cells for: {', '.join(missing)}"
                )
            alg_cell = _make_dispatch_cell(case_cells)

    return functor, alg_cell


def _detect_iterate_groups(
    case_list,
) -> tuple[dict | None, str | None, list]:
    """Detect iterate groups from a list of CaseDecls.

    Returns (iterate_groups, iterate_base, iterate_epilogue).
    iterate_groups is None (not an empty dict) when no iterate cases exist.

    References
    ----------
    Dannert, K. M. et al. (2021). Semiring provenance for fixed-point logic.
    CSL 2021, LIPIcs vol. 183.  cite{dannert2021}

    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """
    iterate_groups: dict[str, list[str]] = {}
    base_case: str | None = None
    epilogue_cases: list[str] = []
    in_iterate = False

    for case_decl in case_list:
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
        return iterate_groups, base_case, epilogue_cases
    return None, None, []


def _resolve_observers(
    arch_decl,
    compiled: dict,
) -> tuple[Callable | None, Callable | None]:
    """Resolve convergence and loss observer paths for an arch.

    Returns (convergence_fn, loss_fn).
    """
    convergence_fn = None
    loss_fn = None

    if arch_decl.observer_convergence:
        if arch_decl.observer_convergence not in compiled:
            raise ValueError(
                f"Arch '{arch_decl.name}': observer convergence path "
                f"'{arch_decl.observer_convergence}' not found"
            )
        convergence_fn = compiled[arch_decl.observer_convergence]

    if arch_decl.observer_loss:
        if arch_decl.observer_loss not in compiled:
            raise ValueError(
                f"Arch '{arch_decl.name}': observer loss path "
                f"'{arch_decl.observer_loss}' not found"
            )
        loss_fn = compiled[arch_decl.observer_loss]

    return convergence_fn, loss_fn


def _compile_archs(
    ast: DSLSource,
    compiled: dict,
    namespace: dict,
    accumulate_legs: dict,
) -> dict[str, _ArchData]:
    """Compile arch declarations.

    Returns compiled_archs dict.
    """
    compiled_archs: dict[str, _ArchData] = {}
    for arch_decl in ast.archs:
        data = _ArchData()

        # Centralize the cases fallback once per arch
        case_list = arch_decl.cases or arch_decl.algebra_cases or []

        # Resolve the unified endofunctor F and algebra cell
        if case_list:
            functor, alg_cell = _build_functor_and_cells(arch_decl, compiled, namespace)
            data.functor = functor
            data.algebra_cell = alg_cell

        # Propagate pre-computed accumulate_legs
        if accumulate_legs:
            data.accumulate_legs = accumulate_legs

        # Detect iterate groups from unified cases
        if case_list:
            iterate_groups, iterate_base, iterate_epilogue = _detect_iterate_groups(case_list)
            if iterate_groups is not None:
                data.iterate_groups = iterate_groups
                data.iterate_base = iterate_base
                data.iterate_epilogue = iterate_epilogue

        # Resolve observer paths
        convergence_fn, loss_fn = _resolve_observers(arch_decl, compiled)
        if convergence_fn is not None:
            data.observer_convergence = convergence_fn
        if loss_fn is not None:
            data.observer_loss = loss_fn

        # Compile state_fields into a Hydra TypeRecord
        if arch_decl.state_fields is not None:
            from .decl import SortDecl as _SortDecl
            state_sort = _SortDecl(
                name=f"_state_{arch_decl.name}",
                fields=arch_decl.state_fields,
            )
            state_sort_defs = {state_sort.name: state_sort}
            data.state_type = sort_to_type(state_sort.name, state_sort_defs)

        # Step protocol: generate coalgebra cell from step declarations
        if arch_decl.step_enter is not None or arch_decl.step_emit is not None:
            if arch_decl.step_enter and arch_decl.step_enter not in compiled:
                raise ValueError(
                    f"Arch '{arch_decl.name}': step enter morphism "
                    f"'{arch_decl.step_enter}' not found"
                )
            if arch_decl.step_emit and arch_decl.step_emit not in compiled:
                raise ValueError(
                    f"Arch '{arch_decl.name}': step emit morphism "
                    f"'{arch_decl.step_emit}' not found"
                )
            enter_fn = compiled.get(arch_decl.step_enter) if arch_decl.step_enter else None
            emit_fn = compiled.get(arch_decl.step_emit) if arch_decl.step_emit else None

            # Find the main computation path from case morphisms
            case_morphism_fn = None
            if case_list:
                for cd in case_list:
                    if cd.morphisms:
                        case_morphism_fn = chain([compiled[m] for m in cd.morphisms]) \
                            if len(cd.morphisms) > 1 else compiled[cd.morphisms[0]]
                        break

            # Find the first recursive case name for the coalgebra step
            step_case_name = None
            if case_list:
                for cd in case_list:
                    if cd.recursive > 0:
                        step_case_name = cd.name
                        break
            if step_case_name is None:
                step_case_name = case_list[0].name if case_list else 'step'

            data.coalgebra_cell = _build_step_cell(enter_fn, emit_fn, case_morphism_fn,
                                                    step_case_name)

        compiled_archs[arch_decl.name] = data
    return compiled_archs


# ---------------------------------------------------------------------------
# Top-level compiler
# ---------------------------------------------------------------------------

def compile(source: str | Path | DSLSource, namespace: dict) -> ArchDef:
    """
    Compile DSL source into an ArchDef.

    Parameters
    ----------
    source    : str, Path, or DSLSource — DSL text, .ua file path, or pre-parsed AST
    namespace : dict  — Python namespace for resolving dotted names
                        e.g. {'numpy': numpy, 'ops': ops_module}

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}

    Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
    PODS 2007, pp. 31–40.  cite{green2007}
    """
    if isinstance(source, Path) or (isinstance(source, str) and source.endswith('.ua')):
        source = Path(source).read_text()
    ast = parse(source) if isinstance(source, str) else source

    sort_defs = {s.name: s for s in ast.sorts}

    # Warn about morphisms referencing undeclared sorts (if sorts were declared)
    if sort_defs:
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
    compiled, morphism_specs, equations, morphism_terms = _compile_morphisms(
        ast, morphism_to_semiring, semiring_contract, semiring_compiler, semiring_arity, namespace
    )

    # Build template registry for parameterized morphisms
    templates = {
        morph_decl.name: morph_decl
        for morph_decl in ast.morphisms
        if morph_decl.template_param is not None
    }

    # Build Hydra sort types for type-checked sort validation
    morph_sorts = {s for m in ast.morphisms for s in (m.src_sort, m.tgt_sort)}
    sort_types = sort_types_from_defs(sort_defs, morph_sorts)

    # Build Hydra primitives for tensor ops
    morph_decls = {m.name: m for m in ast.morphisms}
    sr_decls = {s.name: s for s in ast.semirings}
    hydra_primitives = register_tensor_primitives(sr_decls, morph_decls, namespace)

    # Pre-resolve template instances referenced in fan branches before _compile_fans
    for f in ast.fans:
        for b in f.branches:
            _resolve_template_instance(
                b, templates, compiled, morphism_specs,
                morphism_to_semiring, equations, namespace, semiring_arity,
            )

    _compile_fans(ast, compiled, namespace)
    fan_terms = {f.name: fan_to_term(f.name, f.branches, f.merge) for f in ast.fans}
    path_morphisms = _compile_paths(
        ast, compiled, morphism_specs, morphism_to_semiring,
        templates, equations, namespace, semiring_arity,
        sort_types,
    )
    path_terms = {
        p.name: path_to_term(p.name, path_morphisms.get(p.name, p.morphisms), p.residual, p.normed)
        for p in ast.paths
    }
    # Compute accumulate_legs once here (from morphism_specs) and pass into _compile_archs
    accumulate_legs = {
        name: (spec.accumulate, spec.accumulate_fields)
        for name, spec in morphism_specs.items()
        if spec.accumulate is not None
    }
    compiled_archs = _compile_archs(ast, compiled, namespace, accumulate_legs)

    # Build Hydra arch terms and union types
    arch_terms = {}
    arch_types = {}
    for arch_decl in ast.archs:
        unified = arch_decl.cases or arch_decl.algebra_cases
        arch_terms[arch_decl.name] = arch_to_term(
            arch_decl.name,
            algebra_cases=unified,
            observer_convergence=arch_decl.observer_convergence,
            observer_loss=arch_decl.observer_loss,
        )
        if unified:
            arch_types[f"{arch_decl.name}.cases"] = functor_to_union_type(unified)

    return ArchDef(
        paths              = compiled,
        morphism_semiring  = morphism_to_semiring,
        _morphism_specs    = morphism_specs,
        _equations         = equations,
        _path_morphisms    = path_morphisms,
        _archs             = compiled_archs,
        sort_defs          = sort_defs,
        _morphism_terms    = morphism_terms,
        _path_terms        = path_terms,
        _fan_terms         = fan_terms,
        _hydra_primitives  = hydra_primitives,
        _arch_terms        = arch_terms,
        _arch_types        = arch_types,
    )
