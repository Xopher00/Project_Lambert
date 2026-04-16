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
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable
from functools import reduce

from .runtime import MorphismSpec, Backend, NUMPY_BACKEND, compile_morphism, chain, chain_with_augments, fan, check_sorts, _close_coercions, grade_sorts
from .functor import Case, Functor
from .decl import DSLSource
from .arch import ArchDef, _ArchData
from .parser import parse
import warnings

from .sorts import functor_to_union_type, sort_to_type, sort_types_from_defs
from .terms import morphism_to_term, path_to_term, fan_to_term, arch_to_term
from .utils import resolve as _resolve
from .primitives import register_primitives


# ---------------------------------------------------------------------------
# Compilation context
# ---------------------------------------------------------------------------

@dataclass
class CompilationContext:
    ast: Any          # DSLSource
    namespace: dict
    backend: Backend = dc_field(default_factory=lambda: NUMPY_BACKEND)
    # Semiring resolution
    semiring_contract: dict = dc_field(default_factory=dict)
    semiring_compiler: dict = dc_field(default_factory=dict)
    semiring_arity: dict = dc_field(default_factory=dict)
    morphism_to_semiring: dict = dc_field(default_factory=dict)
    # Compiled callables and specs
    compiled: dict = dc_field(default_factory=dict)   # name -> callable
    specs: dict = dc_field(default_factory=dict)      # name -> MorphismSpec
    equations: dict = dc_field(default_factory=dict)  # name -> str
    # Template registry and sort info
    templates: dict = dc_field(default_factory=dict)
    sort_types: dict = dc_field(default_factory=dict)
    coercion_grades: Any = None
    # Hydra terms and types (populated after compilation)
    morphism_terms: dict = dc_field(default_factory=dict)
    path_morphisms: dict = dc_field(default_factory=dict)
    path_terms: dict = dc_field(default_factory=dict)
    fan_terms: dict = dc_field(default_factory=dict)
    arch_terms: dict = dc_field(default_factory=dict)
    arch_types: dict = dc_field(default_factory=dict)
    hydra_primitives: Any = None
    errors: list = dc_field(default_factory=list)
    warnings: list = dc_field(default_factory=list)


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
    ctx: "CompilationContext",
):
    """Build MorphismSpecs and compile each morphism.

    Returns (compiled, morphism_specs, equations).
    """
    compiled:       dict[str, Callable]     = {}
    morphism_specs: dict[str, MorphismSpec] = {}
    equations:      dict[str, str]          = {}

    for morph_decl in ast.morphisms:
        if morph_decl.name not in morphism_to_semiring:
            continue  # semiring resolution failed for this morphism; error already recorded
        group = morphism_to_semiring[morph_decl.name]
        if morph_decl.op is not None:
            try:
                op_fn = _resolve(morph_decl.op, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Morphism '{morph_decl.name}': cannot resolve op '{morph_decl.op}': {e}")
                continue
        elif group in semiring_contract:
            op_fn = semiring_contract[group]
        else:
            ctx.errors.append(
                f"Morphism '{morph_decl.name}': no 'op' clause and semiring '{group}' "
                f"has no 'contract'. Add 'op <fn>' to the morphism or declare a "
                f"semiring contract."
            )
            continue
        if morph_decl.transform:
            try:
                transform_fn = _resolve(morph_decl.transform, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Morphism '{morph_decl.name}': cannot resolve transform '{morph_decl.transform}': {e}")
                continue
        else:
            transform_fn = lambda x, y: (x, y)
        if morph_decl.compiler is not None:
            try:
                eq_compiler = _resolve(morph_decl.compiler, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Morphism '{morph_decl.name}': cannot resolve compiler '{morph_decl.compiler}': {e}")
                continue
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
    ctx: "CompilationContext",
    sort_types: dict | None = None,
    coercion_grades: dict | None = None,
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
        path_has_error = False
        if coercion_grades is not None:
            errs, warns = grade_sorts(
                morphism_specs, sort_morphisms, sort_types or {},
                coercion_grades, ast.sort_threshold,
            )
            for w in warns:
                warnings.warn(f"Path '{path.name}': {w}", stacklevel=4)
            for e in errs:
                ctx.errors.append(f"Path '{path.name}': {e}")
                path_has_error = True
        else:
            err = check_sorts(morphism_specs, sort_morphisms, sort_types)
            if err:
                ctx.errors.append(f"Path '{path.name}': {err}")
                path_has_error = True
        # Semiring validation
        used = {morphism_to_semiring[m] for m in sort_morphisms if m in morphism_to_semiring}
        if len(used) > 1 and '_bridge' not in used:
            ctx.errors.append(
                f"Path '{path.name}' spans multiple semirings: {used}. "
                f"Use a bridge morphism to cross semiring boundaries."
            )
            path_has_error = True
        if path_has_error:
            continue  # skip building chain for this path, collect more errors from other paths
        # Build chain, handling augment steps
        has_augments = any(m.startswith('[') and m.endswith(']')
                           for m in path.morphisms)
        if has_augments:
            steps = []
            augment_ok = True
            for m in path.morphisms:
                if m.startswith('[') and m.endswith(']'):
                    fan_name = m[1:-1]
                    if fan_name not in compiled:
                        ctx.errors.append(
                            f"Path '{path.name}': augment target '{fan_name}' "
                            f"is not a declared morphism, path, or fan"
                        )
                        augment_ok = False
                    else:
                        steps.append(('augment', compiled[fan_name]))
                else:
                    if m not in compiled:
                        ctx.errors.append(
                            f"Path '{path.name}': morphism '{m}' not found"
                        )
                        augment_ok = False
                    else:
                        steps.append(('step', compiled[m]))
            if not augment_ok:
                continue
            base = chain_with_augments(steps)
        else:
            missing = [m for m in path.morphisms if m not in compiled]
            if missing:
                for m in missing:
                    ctx.errors.append(
                        f"Path '{path.name}': morphism '{m}' not found"
                    )
                continue
            base = chain([compiled[m] for m in path.morphisms])
        if path.residual or path.normed:
            if path.normed and path.normed not in compiled:
                ctx.errors.append(
                    f"Path '{path.name}': normed morphism '{path.normed}' "
                    f"is not a declared morphism or path"
                )
                continue
            norm_fn = compiled[path.normed] if path.normed else None
            compiled[path.name] = _build_residual_wrapper(base, norm_fn, path.residual)
        else:
            compiled[path.name] = base
        path_morphisms[path.name] = sort_morphisms

    return path_morphisms


def _compile_fans(ast: DSLSource, compiled: dict, namespace: dict,
                  ctx: "CompilationContext", backend: "Backend" = None) -> None:
    """Compile fans.

    Mutates compiled in-place.
    """
    if backend is None:
        backend = NUMPY_BACKEND
    for f in ast.fans:
        branch_callables = {}
        fan_ok = True
        for b in f.branches:
            if b not in compiled:
                ctx.errors.append(
                    f"Fan '{f.name}': branch '{b}' is not a declared morphism or path"
                )
                fan_ok = False
            else:
                branch_callables[b] = compiled[b]
        if not fan_ok:
            continue
        if f.merge == 'dict':
            merge_fn = lambda results: results
        elif f.merge == 'meet':
            merge_fn = lambda results, _be=backend: reduce(_be.minimum, results.values())
        elif f.merge == 'join':
            merge_fn = lambda results, _be=backend: reduce(_be.maximum, results.values())
        else:
            merge_fn = _resolve(f.merge, namespace)
        compiled[f.name] = fan(branch_callables, merge_fn)


def _resolve_case_cells(
    case_list,
    context_name: str,
    compiled: dict,
    namespace: dict,
    errors: list,
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
            case_ok = True
            for morph_name in case_decl.morphisms:
                if morph_name not in compiled:
                    errors.append(
                        f"{context_name} case '{case_decl.name}': "
                        f"morphism '{morph_name}' not found in compiled paths"
                    )
                    case_ok = False
            if case_ok:
                path_fn = chain([compiled[morph_name] for morph_name in case_decl.morphisms]) \
                    if len(case_decl.morphisms) > 1 else compiled[case_decl.morphisms[0]]
                case_cells[case_decl.name] = _make_morphism_derived_cell(case_decl.name, path_fn)
    return case_cells



def _build_functor_and_cells(
    arch_decl,
    compiled: dict,
    namespace: dict,
    errors: list,
) -> tuple["Functor", Callable | None]:
    """Build the endofunctor F and resolve the algebra cell for an arch.

    Handles functor construction from cases and resolves per-case cells or
    the default algebra_cell.  Returns (functor, alg_cell).

    Raises ValueError for semantic violations:
    - case must not specify both 'cell' and 'morphisms'
    """
    case_list = arch_decl.cases or []

    # Semantic validation: cell= and morphisms= are mutually exclusive
    for cd in case_list:
        if cd.cell is not None and cd.morphisms is not None:
            errors.append(
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
            case_list, f"Arch '{arch_decl.name}' algebra", compiled, namespace, errors
        )
        if case_cells:
            missing = [cd.name for cd in case_list if cd.name not in case_cells]
            if missing:
                errors.append(
                    f"Arch '{arch_decl.name}' algebra: per-case cell binding is "
                    f"incomplete. Missing cells for: {', '.join(missing)}"
                )
            else:
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
    errors: list,
) -> tuple[Callable | None, Callable | None]:
    """Resolve convergence and loss observer paths for an arch.

    Returns (convergence_fn, loss_fn).
    """
    convergence_fn = None
    loss_fn = None

    if arch_decl.observer_convergence:
        if arch_decl.observer_convergence not in compiled:
            errors.append(
                f"Arch '{arch_decl.name}': observer convergence path "
                f"'{arch_decl.observer_convergence}' not found"
            )
        else:
            convergence_fn = compiled[arch_decl.observer_convergence]

    if arch_decl.observer_loss:
        if arch_decl.observer_loss not in compiled:
            errors.append(
                f"Arch '{arch_decl.name}': observer loss path "
                f"'{arch_decl.observer_loss}' not found"
            )
        else:
            loss_fn = compiled[arch_decl.observer_loss]

    return convergence_fn, loss_fn


def _compile_archs(
    ast: DSLSource,
    compiled: dict,
    namespace: dict,
    accumulate_specs: dict,
    ctx: "CompilationContext",
    backend: "Backend" = None,
) -> dict[str, _ArchData]:
    """Compile arch declarations.

    Returns compiled_archs dict.
    """
    if backend is None:
        backend = NUMPY_BACKEND
    compiled_archs: dict[str, _ArchData] = {}
    for arch_decl in ast.archs:
        data = _ArchData()
        data.backend = backend

        # Centralize the cases fallback once per arch
        case_list = arch_decl.cases or []

        # Resolve the unified endofunctor F and algebra cell
        if case_list:
            functor, alg_cell = _build_functor_and_cells(arch_decl, compiled, namespace, ctx.errors)
            data.functor = functor
            data.algebra_cell = alg_cell

        # Propagate pre-computed accumulate_specs
        if accumulate_specs:
            data.accumulate_specs = accumulate_specs

        # Detect iterate groups from unified cases
        if case_list:
            iterate_groups, iterate_base, iterate_epilogue = _detect_iterate_groups(case_list)
            if iterate_groups is not None:
                data.iterate_groups = iterate_groups
                data.iterate_base = iterate_base
                data.iterate_epilogue = iterate_epilogue

        # Resolve observer paths
        convergence_fn, loss_fn = _resolve_observers(arch_decl, compiled, ctx.errors)
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
            step_ok = True
            if arch_decl.step_enter and arch_decl.step_enter not in compiled:
                ctx.errors.append(
                    f"Arch '{arch_decl.name}': step enter morphism "
                    f"'{arch_decl.step_enter}' not found"
                )
                step_ok = False
            if arch_decl.step_emit and arch_decl.step_emit not in compiled:
                ctx.errors.append(
                    f"Arch '{arch_decl.name}': step emit morphism "
                    f"'{arch_decl.step_emit}' not found"
                )
                step_ok = False
            if arch_decl.step_compute and arch_decl.step_compute not in compiled:
                ctx.errors.append(
                    f"Arch '{arch_decl.name}': step compute morphism "
                    f"'{arch_decl.step_compute}' not found"
                )
                step_ok = False
            if step_ok:
                enter_fn = compiled.get(arch_decl.step_enter) if arch_decl.step_enter else None
                emit_fn = compiled.get(arch_decl.step_emit) if arch_decl.step_emit else None

                # Find the main computation path: explicit step_compute overrides first-case heuristic
                if arch_decl.step_compute:
                    case_morphism_fn = compiled[arch_decl.step_compute]
                else:
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
# Name validation pass
# ---------------------------------------------------------------------------

def _validate_names(ctx: CompilationContext) -> None:
    """Check that all name references in arch blocks, fans, and step fields resolve to declared names."""
    ast = ctx.ast
    declared: set[str] = set()
    declared.update(m.name for m in ast.morphisms)
    declared.update(p.name for p in ast.paths)
    declared.update(f.name for f in ast.fans)

    for arch in ast.archs:
        for case in (arch.cases or []):
            if case.morphisms:
                for m in case.morphisms:
                    if m not in declared:
                        ctx.errors.append(
                            f"arch '{arch.name}' case '{case.name}': "
                            f"morphism/path '{m}' not declared"
                        )
        for attr in ('step_enter', 'step_emit', 'step_compute'):
            ref = getattr(arch, attr, None)
            if ref and ref not in declared:
                ctx.errors.append(
                    f"arch '{arch.name}' step '{attr}': '{ref}' not declared"
                )
        if arch.observer_convergence and arch.observer_convergence not in declared:
            ctx.errors.append(
                f"arch '{arch.name}' observer convergence: "
                f"'{arch.observer_convergence}' not declared"
            )
        if arch.observer_loss and arch.observer_loss not in declared:
            ctx.errors.append(
                f"arch '{arch.name}' observer loss: "
                f"'{arch.observer_loss}' not declared"
            )

    for fan in ast.fans:
        for branch in (fan.branches or []):
            if branch not in declared:
                ctx.errors.append(
                    f"fan '{fan.name}' branch: '{branch}' is not a declared morphism or path"
                )


# ---------------------------------------------------------------------------
# Top-level compiler helpers
# ---------------------------------------------------------------------------

def _validate_hydra_terms(ctx: CompilationContext) -> None:
    if not ctx.errors and ctx.arch_terms:
        try:
            import hydra.validate.core as hvc
            from hydra.dsl.python import Just as HydraJust
            from hydra.dsl.python import FrozenDict
            import hydra.graph as hg
            import hydra.core as hc
            _empty_graph = hg.Graph(
                bound_terms=FrozenDict({}),
                bound_types=FrozenDict({}),
                class_constraints=FrozenDict({}),
                lambda_variables=frozenset(),
                metadata=FrozenDict({}),
                primitives=FrozenDict({}),
                schema_types=FrozenDict({}),
                type_variables=frozenset(),
            )
            for name, arch_term in ctx.arch_terms.items():
                result = hvc.term(False, _empty_graph, arch_term)
                if isinstance(result, HydraJust):
                    ctx.errors.append(
                        f"Structural validation error in {name!r}: {result.value}"
                    )
        except ImportError:
            pass


def _audit_unreferenced_declarations(ctx: CompilationContext, ast) -> None:
    if not ctx.errors:
        try:
            import re as _re2
            from hydra.dependencies import term_dependency_names
            # Collect Hydra-level term dependencies from all emitted terms.
            all_hydra_terms = {
                **ctx.morphism_terms,
                **ctx.path_terms,
                **ctx.fan_terms,
                **ctx.arch_terms,
            }
            hydra_referenced: frozenset = frozenset().union(
                *(term_dependency_names(True, True, False, t) for t in all_hydra_terms.values())
            ) if all_hydra_terms else frozenset()
            # Collect AST-level cross-references (paths reference morphisms,
            # fans reference paths/morphisms, arch cases reference morphisms/paths).
            ast_referenced: set[str] = set()
            for p in ast.paths:
                for tok in p.morphisms:
                    # Strip augment brackets and template params: "[kv]" -> "kv", "ln[ln1]" -> "ln1"
                    bracket = _re2.fullmatch(r'\[(\w+)\]', tok)
                    template = _re2.fullmatch(r'\w+\[(\w+)\]', tok)
                    if bracket:
                        ast_referenced.add(bracket.group(1))
                    elif template:
                        ast_referenced.add(template.group(1))
                    else:
                        ast_referenced.add(tok)
                if p.normed:
                    ast_referenced.add(p.normed)
            for f in ast.fans:
                ast_referenced.update(f.branches)
            for arch_decl in ast.archs:
                for field_name in (arch_decl.step_enter, arch_decl.step_emit,
                                   arch_decl.step_compute, arch_decl.observer_convergence,
                                   arch_decl.observer_loss):
                    if field_name:
                        ast_referenced.add(field_name)
                if arch_decl.cases:
                    for c in arch_decl.cases:
                        if c.morphisms:
                            ast_referenced.update(c.morphisms)
            referenced = hydra_referenced | ast_referenced
            declared = {m.name for m in ast.morphisms}
            for name in sorted(declared - referenced):
                warnings.warn(f"Declared but unreferenced: {name!r}", UserWarning, stacklevel=2)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Top-level compiler
# ---------------------------------------------------------------------------

def compile(source: str | Path | DSLSource, namespace: dict,
            backend: Backend = None) -> ArchDef:
    """
    Compile DSL source into an ArchDef.

    Parameters
    ----------
    source    : str, Path, or DSLSource — DSL text, .ua file path, or pre-parsed AST
    namespace : dict  — Python namespace for resolving dotted names
                        e.g. {'numpy': numpy, 'ops': ops_module}
    backend   : Backend or None — array operation backend; defaults to NUMPY_BACKEND

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

    ctx = CompilationContext(ast=ast, namespace=namespace,
                             backend=backend if backend is not None else NUMPY_BACKEND)

    # Phase 1: Validate names
    _validate_names(ctx)
    if ctx.errors:
        raise ValueError("DSL name validation failed:\n" + "\n".join(f"  - {e}" for e in ctx.errors))
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

    # Phase 2: Resolve semirings
    ctx.semiring_contract = {
        sd.name: _resolve(sd.contract, namespace) for sd in ast.semirings
    }
    ctx.semiring_compiler = {
        sd.name: _resolve(sd.compiler, namespace)
        for sd in ast.semirings if sd.compiler is not None
    }
    ctx.semiring_arity = {sd.name: sd.arity for sd in ast.semirings}

    sole = ast.semirings[0].name if len(ast.semirings) == 1 else None
    for morph_decl in ast.morphisms:
        if morph_decl.semiring is None:
            ctx.morphism_to_semiring[morph_decl.name] = '_bridge'
        elif morph_decl.semiring == '_default':
            ctx.morphism_to_semiring[morph_decl.name] = sole if sole else '_default'
        elif morph_decl.semiring not in ctx.semiring_contract:
            ctx.errors.append(
                f"Morphism '{morph_decl.name}' references undeclared semiring '{morph_decl.semiring}'"
            )
        else:
            ctx.morphism_to_semiring[morph_decl.name] = morph_decl.semiring

    # Phase 3: Compile morphisms
    ctx.compiled, ctx.specs, ctx.equations, ctx.morphism_terms = _compile_morphisms(
        ast, ctx.morphism_to_semiring, ctx.semiring_contract,
        ctx.semiring_compiler, ctx.semiring_arity, namespace, ctx,
    )

    # Build template registry for parameterized morphisms
    ctx.templates = {
        morph_decl.name: morph_decl
        for morph_decl in ast.morphisms
        if morph_decl.template_param is not None
    }

    # Build Hydra sort types for type-checked sort validation
    morph_sorts = {s for m in ast.morphisms for s in (m.src_sort, m.tgt_sort)}
    ctx.sort_types = sort_types_from_defs(sort_defs, morph_sorts)

    # Build Hydra primitives for tensor ops (skip if errors already accumulated —
    # register_primitives will re-raise on unresolvable ops)
    morph_decls = {m.name: m for m in ast.morphisms}
    if not ctx.errors:
        ctx.hydra_primitives = register_primitives(
            ctx.semiring_contract,
            ctx.morphism_to_semiring,
            ctx.equations,
            morph_decls,
            namespace,
        )
    else:
        ctx.hydra_primitives = {}

    # Phase 4: Compile fans (pre-resolve template instances in branches first)
    for f in ast.fans:
        for b in f.branches:
            _resolve_template_instance(
                b, ctx.templates, ctx.compiled, ctx.specs,
                ctx.morphism_to_semiring, ctx.equations, namespace, ctx.semiring_arity,
            )
    _compile_fans(ast, ctx.compiled, namespace, ctx, ctx.backend)
    ctx.fan_terms = {f.name: fan_to_term(f.name, f.branches, f.merge) for f in ast.fans}

    # Phase 5: Compile paths
    ctx.coercion_grades = _close_coercions(ast.coercions) if ast.coercions else None
    ctx.path_morphisms = _compile_paths(
        ast, ctx.compiled, ctx.specs, ctx.morphism_to_semiring,
        ctx.templates, ctx.equations, namespace, ctx.semiring_arity,
        ctx, ctx.sort_types, ctx.coercion_grades,
    )
    ctx.path_terms = {
        p.name: path_to_term(p.name, ctx.path_morphisms.get(p.name, p.morphisms), p.residual, p.normed)
        for p in ast.paths
    }

    # Phase 6: Compile archs
    accumulate_specs = {
        name: (spec.accumulate, spec.accumulate_fields)
        for name, spec in ctx.specs.items()
        if spec.accumulate is not None
    }
    compiled_archs = _compile_archs(ast, ctx.compiled, namespace, accumulate_specs, ctx, ctx.backend)

    # Phase 7: Build Hydra arch terms and union types
    for arch_decl in ast.archs:
        unified = arch_decl.cases
        ctx.arch_terms[arch_decl.name] = arch_to_term(
            arch_decl.name,
            cases=unified,
            observer_convergence=arch_decl.observer_convergence,
            observer_loss=arch_decl.observer_loss,
        )
        if unified:
            ctx.arch_types[f"{arch_decl.name}.cases"] = functor_to_union_type(unified)

    # Post-Phase-7 — structural validation via hydra.validate.core
    _validate_hydra_terms(ctx)

    # Phase 7.5 — unused declaration audit
    _audit_unreferenced_declarations(ctx, ast)

    if ctx.errors:
        raise ValueError("\n".join(ctx.errors))

    return ArchDef(
        paths              = ctx.compiled,
        morphism_semiring  = ctx.morphism_to_semiring,
        _morphism_specs    = ctx.specs,
        _equations         = ctx.equations,
        _path_morphisms    = ctx.path_morphisms,
        _archs             = compiled_archs,
        sort_defs          = sort_defs,
        _morphism_terms    = ctx.morphism_terms,
        _path_terms        = ctx.path_terms,
        _fan_terms         = ctx.fan_terms,
        _hydra_primitives  = ctx.hydra_primitives,
        _arch_terms        = ctx.arch_terms,
        _arch_types        = ctx.arch_types,
    )
