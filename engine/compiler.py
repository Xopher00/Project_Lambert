"""
Compiles DSL source text into executable architecture definitions.

Key phases:
  1. Resolve semirings — map names to contract/compiler/arity callables
  2. Compile morphisms — MorphismSpec → cached (x, y, temp) → result callables
  3. Compile fans — branch callables + merge strategy
  4. Compile paths — chain/augment/residual composition with sort validation
  5. Compile archs — functor + cell binding for algebra and coalgebra sides
"""

from __future__ import annotations

import re as _re
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable
from functools import reduce

from .runtime import MorphismSpec, Backend, NUMPY_BACKEND, compile_morphism, chain, chain_with_augments, fan
from .functor import Case, Functor
from .parser import DSLSource
from .arch import ArchDef, _ArchData
from .parser import parse
from .sorts import sort_to_type, sort_types_from_defs, _close_coercions, grade_sorts
import hydra.dsl.types as _types
from hydra.core import Name as _Name
from .terms import path_to_term, fan_to_term, arch_to_term
from .primitives import register_primitives


def _resolve(dotted: str, namespace: dict) -> object:
    """Walk a dotted name through a namespace dict."""
    parts = dotted.split('.')
    obj = namespace.get(parts[0])
    if obj is None:
        raise NameError(f"Name {parts[0]!r} not found in provided namespace")
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj


def _resolve_semiring_op(name: str, backend, namespace: dict):
    """Resolve a semiring op: backend field first, then namespace."""
    if hasattr(backend, name):
        return getattr(backend, name)
    return _resolve(name, namespace)


def _make_semiring_contract(plus_fn, times_fn):
    """Synthesize contract(compiled_eq, x, y, temp) from + and * operations.

    For equation 'sh,hh->sh': contracts over h.
    result[s,h'] = plus_h( times(x[s,h], y[h,h']) )

    plus_fn must accept an axis= kwarg (reduction).
    times_fn must support elementwise broadcasting.
    """
    def contract(compiled_eq, x, y, temp=0.0):
        lhs, rhs = compiled_eq.split('->')
        x_idx, y_idx = lhs.split(',')
        out_idx = rhs.strip()

        seen, all_idx = set(), []
        for c in x_idx + y_idx:
            if c not in seen:
                seen.add(c); all_idx.append(c)

        contracted = [c for c in all_idx
                      if c in x_idx and c in y_idx and c not in out_idx]

        x_shape = [x.shape[x_idx.index(c)] if c in x_idx else 1 for c in all_idx]
        y_shape = [y.shape[y_idx.index(c)] if c in y_idx else 1 for c in all_idx]

        result = times_fn(x.reshape(x_shape), y.reshape(y_shape))

        for c in reversed(contracted):
            ax = all_idx.index(c)
            all_idx.pop(ax)
            result = plus_fn(result, axis=ax)

        non_contr = list(all_idx)
        if list(out_idx) != non_contr:
            perm = [non_contr.index(c) for c in out_idx]
            result = result.transpose(perm)

        return result
    return contract


def _check_semiring_axioms(sr_name, plus_fn, times_fn, zero_val, one_val, warn_list):
    """Warn if declared semiring operations appear to violate semiring axioms.

    Tests commutativity/associativity of +, distributivity of * over +,
    annihilation by zero, and identity of one against representative scalars.
    Emits warnings (not errors) — the user asserts correctness.
    """
    import itertools, math

    def approx(a, b):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError, OverflowError):
            return a == b

    scalars = [zero_val, one_val, 2.0, -1.0]
    checks = [
        ("+ commutative",    lambda a, b, c: approx(plus_fn(a, b), plus_fn(b, a))),
        ("+ associative",    lambda a, b, c: approx(plus_fn(a, plus_fn(b, c)), plus_fn(plus_fn(a, b), c))),
        ("* distributes +",  lambda a, b, c: approx(times_fn(a, plus_fn(b, c)),
                                                     plus_fn(times_fn(a, b), times_fn(a, c)))),
        ("zero annihilates", lambda a, b, c: approx(times_fn(zero_val, a), zero_val)),
        ("one is identity",  lambda a, b, c: approx(times_fn(one_val, a), a)),
    ]
    for label, check in checks:
        for a, b, c in itertools.islice(itertools.product(scalars, repeat=3), 16):
            try:
                if not check(a, b, c):
                    warn_list.append(f"semiring '{sr_name}': {label} may not hold")
                    break
            except Exception:
                pass  # skip on numeric error (e.g. -inf arithmetic)


# ---------------------------------------------------------------------------
# Compilation context
# ---------------------------------------------------------------------------

@dataclass
class CompilationContext:
    ast: Any          # DSLSource
    namespace: dict
    backend: Backend = dc_field(default_factory=lambda: NUMPY_BACKEND)
    semiring_contract: dict = dc_field(default_factory=dict)
    semiring_compiler: dict = dc_field(default_factory=dict)
    semiring_arity: dict = dc_field(default_factory=dict)
    semiring_plus:    dict = dc_field(default_factory=dict)
    semiring_times:   dict = dc_field(default_factory=dict)
    morphism_to_semiring: dict = dc_field(default_factory=dict)
    compiled: dict = dc_field(default_factory=dict)
    specs: dict = dc_field(default_factory=dict)
    equations: dict = dc_field(default_factory=dict)
    templates: dict = dc_field(default_factory=dict)
    sort_types: dict = dc_field(default_factory=dict)
    coercion_grades: Any = None
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
    morphism_decls: list,
    morphism_to_semiring: dict,
    semiring_contract: dict,
    semiring_compiler: dict,
    semiring_arity: dict,
    namespace: dict,
    ctx: "CompilationContext",
):
    """Build MorphismSpecs and compile each morphism.

    Returns (compiled, morphism_specs, equations, morphism_terms).
    """
    compiled:       dict[str, Callable]     = {}
    morphism_specs: dict[str, MorphismSpec] = {}
    equations:      dict[str, str]          = {}
    decl_map:       dict                    = {m.name: m for m in morphism_decls}

    for m in morphism_decls:
        name = m.name
        if name not in morphism_to_semiring:
            continue  # semiring resolution failed for this morphism; error already recorded
        group = morphism_to_semiring[name]
        if m.op is not None:
            try:
                op_fn = _resolve(m.op, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Morphism '{name}': cannot resolve op '{m.op}': {e}")
                continue
        elif group in semiring_contract:
            op_fn = semiring_contract[group]
        else:
            ctx.errors.append(
                f"Morphism '{name}': no 'op' clause and semiring '{group}' "
                f"has no 'contract'. Add 'op <fn>' to the morphism or declare a "
                f"semiring contract."
            )
            continue
        if m.transform:
            try:
                transform_fn = _resolve(m.transform, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Morphism '{name}': cannot resolve transform '{m.transform}': {e}")
                continue
        else:
            transform_fn = lambda x, y: (x, y)
        if m.compiler is not None:
            try:
                eq_compiler = _resolve(m.compiler, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Morphism '{name}': cannot resolve compiler '{m.compiler}': {e}")
                continue
        elif group in semiring_compiler:
            eq_compiler = semiring_compiler[group]
        else:
            eq_compiler = lambda eq: eq

        spec = MorphismSpec(
            name=name, op=op_fn, equation=m.equation,
            src_sort=m.src_sort, tgt_sort=m.tgt_sort,
            equation_compiler=eq_compiler, transform=transform_fn,
            arity=m.arity if m.arity != 'binary' else semiring_arity.get(group, 'binary'),
            accumulate=m.accumulate,
            accumulate_fields=m.accumulate_fields,
        )
        morphism_specs[name] = spec
        equations[name] = m.equation
        compiled[name]  = compile_morphism(spec)

    from .terms import morphism as _morphism_term
    morphism_terms = {
        name: _morphism_term(
            spec.name, spec.src_sort, spec.tgt_sort, spec.arity,
            [decl_map[name].template_param] if decl_map.get(name) and decl_map[name].template_param else [],
        ).value
        for name, spec in morphism_specs.items()
    }
    return compiled, morphism_specs, equations, morphism_terms


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

    template_conf = templates[base_name]
    param_name = template_conf.template_param  # e.g. 'prefix'
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
    path_decls: list,
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
    sort_threshold: float | None = None,
) -> dict[str, list[str]]:
    """Validate and compile paths.

    Mutates compiled in-place to add path callables.
    Returns path_morphisms dict.
    """
    path_morphisms: dict[str, list[str]] = {}
    for p in path_decls:
        name      = p.name
        morphisms = p.morphisms
        residual  = p.residual
        normed    = p.normed
        # Resolve template instantiations (e.g. 'ln[ln1]' -> curried morphism)
        for m in morphisms:
            if not (m.startswith('[') and m.endswith(']')):  # skip augment brackets
                _resolve_template_instance(
                    m, templates, compiled, morphism_specs,
                    morphism_to_semiring, equations, namespace, semiring_arity,
                )
        # Strip bracketed augment tokens for sort/semiring validation
        sort_morphisms = [m for m in morphisms
                         if not (m.startswith('[') and m.endswith(']'))
                         and m in morphism_specs]
        # Sort validation
        path_has_error = False
        if coercion_grades is not None:
            errs, warns = grade_sorts(
                morphism_specs, sort_morphisms, sort_types or {},
                coercion_grades, sort_threshold,
            )
            for w in warns:
                warnings.warn(f"Path '{name}': {w}", stacklevel=4)
            for e in errs:
                ctx.errors.append(f"Path '{name}': {e}")
                path_has_error = True
        else:
            for i in range(1, len(sort_morphisms)):
                prev_tgt = morphism_specs[sort_morphisms[i - 1]].tgt_sort
                curr_src = morphism_specs[sort_morphisms[i]].src_sort
                prev_type = sort_types.get(prev_tgt)
                curr_type = sort_types.get(curr_src)
                if prev_type is not None and curr_type is not None:
                    mismatch = (prev_type != curr_type)
                else:
                    mismatch = (prev_tgt != curr_src)
                if mismatch:
                    ctx.errors.append(
                        f"Path '{name}': Type mismatch at step {i}: "
                        f"{sort_morphisms[i - 1]!r} outputs {prev_tgt!r} but "
                        f"{sort_morphisms[i]!r} expects {curr_src!r}"
                    )
                    path_has_error = True
                    break
        # Semiring validation
        used = {morphism_to_semiring[m] for m in sort_morphisms if m in morphism_to_semiring}
        if len(used) > 1 and '_bridge' not in used:
            ctx.errors.append(
                f"Path '{name}' spans multiple semirings: {used}. "
                f"Use a bridge morphism to cross semiring boundaries."
            )
            path_has_error = True
        if path_has_error:
            continue  # skip building chain for this path, collect more errors from other paths
        # Build chain, handling augment steps
        has_augments = any(m.startswith('[') and m.endswith(']') for m in morphisms)
        if has_augments:
            steps = []
            augment_ok = True
            for m in morphisms:
                if m.startswith('[') and m.endswith(']'):
                    fan_name = m[1:-1]
                    if fan_name not in compiled:
                        ctx.errors.append(
                            f"Path '{name}': augment target '{fan_name}' "
                            f"is not a declared morphism, path, or fan"
                        )
                        augment_ok = False
                    else:
                        steps.append(('augment', compiled[fan_name]))
                else:
                    if m not in compiled:
                        ctx.errors.append(
                            f"Path '{name}': morphism '{m}' not found"
                        )
                        augment_ok = False
                    else:
                        steps.append(('step', compiled[m]))
            if not augment_ok:
                continue
            base = chain_with_augments(steps)
        else:
            missing = [m for m in morphisms if m not in compiled]
            if missing:
                for m in missing:
                    ctx.errors.append(f"Path '{name}': morphism '{m}' not found")
                continue
            base = chain([compiled[m] for m in morphisms])
        if residual or normed:
            if normed and normed not in compiled:
                ctx.errors.append(
                    f"Path '{name}': normed morphism '{normed}' "
                    f"is not a declared morphism or path"
                )
                continue
            norm_fn = compiled[normed] if normed else None
            if residual and norm_fn is not None:
                compiled[name] = lambda x, y, temp, _b=base, _n=norm_fn: \
                    _n(_b(x, y, temp) + x, y, temp)
            elif residual:
                compiled[name] = lambda x, y, temp, _b=base: \
                    _b(x, y, temp) + x
            elif norm_fn is not None:
                compiled[name] = lambda x, y, temp, _b=base, _n=norm_fn: \
                    _n(_b(x, y, temp), y, temp)
            else:
                compiled[name] = base
        else:
            compiled[name] = base
        path_morphisms[name] = sort_morphisms

    return path_morphisms


def _compile_fans(fan_decls: list, compiled: dict, namespace: dict,
                  ctx: "CompilationContext", backend: "Backend" = None) -> None:
    """Compile fans.

    Mutates compiled in-place.
    """
    if backend is None:
        backend = NUMPY_BACKEND
    for f in fan_decls:
        name = f.name
        branch_callables = {}
        fan_ok = True
        for b in f.branches:
            if b not in compiled:
                ctx.errors.append(
                    f"Fan '{name}': branch '{b}' is not a declared morphism or path"
                )
                fan_ok = False
            else:
                branch_callables[b] = compiled[b]
        if not fan_ok:
            continue
        merge = f.merge
        if merge == 'dict':
            merge_fn = lambda results: results
        elif merge == 'meet':
            merge_fn = lambda results, _be=backend: reduce(_be.minimum, results.values())
        elif merge == 'join':
            merge_fn = lambda results, _be=backend: reduce(_be.maximum, results.values())
        else:
            merge_fn = _resolve(merge, namespace)
        compiled[name] = fan(branch_callables, merge_fn)


def _resolve_case_cells(
    case_list,
    context_name: str,
    compiled: dict,
    namespace: dict,
    errors: list,
) -> dict[str, Callable]:
    """Resolve per-case cells (explicit cell= or morphisms= derived).

    case_list is a list of CaseDecl objects.
    Returns case_cells dict.
    """
    case_cells = {}
    for c in case_list:
        cname = c.name
        cell = c.cell
        morphisms = c.morphisms
        if cell == 'identity':
            case_cells[cname] = lambda payload, child_results, _params, _temp: \
                payload[0] if payload else (child_results[0] if child_results else None)
        elif cell is not None:
            case_cells[cname] = _resolve(cell, namespace)
        elif morphisms is not None:
            case_ok = True
            for morph_name in morphisms:
                if morph_name not in compiled:
                    errors.append(
                        f"{context_name} case '{cname}': "
                        f"morphism '{morph_name}' not found in compiled paths"
                    )
                    case_ok = False
            if case_ok:
                path_fn = chain([compiled[m] for m in morphisms]) \
                    if len(morphisms) > 1 else compiled[morphisms[0]]
                case_cells[cname] = _make_morphism_derived_cell(cname, path_fn)
    return case_cells


def _build_functor_and_cells(
    arch_name: str,
    case_list: list,
    algebra_cell,
    compiled: dict,
    namespace: dict,
    errors: list,
) -> tuple["Functor", Callable | None]:
    """Build the endofunctor F and resolve the algebra cell for an arch.

    case_list is a list of CaseDecl objects.
    Returns (functor, alg_cell).
    """
    # Semantic validation: cell= and morphisms= are mutually exclusive
    for c in case_list:
        if c.cell is not None and c.morphisms is not None:
            errors.append(
                f"Arch '{arch_name}' case '{c.name}': "
                f"cannot specify both 'cell' and 'morphisms'"
            )

    cases = [Case(c.name, c.recursive, c.data, c.output) for c in case_list]
    functor = Functor(cases)

    alg_cell = None
    if algebra_cell is not None:
        alg_cell = _resolve(algebra_cell, namespace)
    else:
        case_cells = _resolve_case_cells(
            case_list, f"Arch '{arch_name}' algebra", compiled, namespace, errors
        )
        if case_cells:
            missing = [c.name for c in case_list if c.name not in case_cells]
            if missing:
                errors.append(
                    f"Arch '{arch_name}' algebra: per-case cell binding is "
                    f"incomplete. Missing cells for: {', '.join(missing)}"
                )
            else:
                alg_cell = _make_dispatch_cell(case_cells)

    return functor, alg_cell


def _compile_archs(
    arch_decls: list,
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
    for a in arch_decls:
        name = a.name
        data = _ArchData()
        data.backend = backend

        case_list = a.cases or []

        # Resolve the unified endofunctor F and algebra cell
        if case_list:
            functor, alg_cell = _build_functor_and_cells(
                name, case_list, a.algebra_cell, compiled, namespace, ctx.errors
            )
            data.functor = functor
            data.algebra_cell = alg_cell

        # Propagate pre-computed accumulate_specs
        if accumulate_specs:
            data.accumulate_specs = accumulate_specs

        # Detect iterate groups from unified cases
        if case_list:
            _iterate_groups: dict[str, list[str]] = {}
            _base_case: str | None = None
            _epilogue_cases: list[str] = []
            _in_iterate = False
            for c in case_list:
                if c.iterate is not None:
                    _in_iterate = True
                    _gname = c.iterate
                    if _gname not in _iterate_groups:
                        _iterate_groups[_gname] = []
                    _iterate_groups[_gname].append(c.name)
                elif c.recursive == 0 and not _in_iterate:
                    _base_case = c.name
                else:
                    if _in_iterate:
                        _epilogue_cases.append(c.name)
            if _iterate_groups:
                data.iterate_groups = _iterate_groups
                data.iterate_base = _base_case
                data.iterate_epilogue = _epilogue_cases

        # Resolve observer paths
        convergence_fn = None
        loss_fn = None
        if a.observer_convergence:
            if a.observer_convergence not in compiled:
                ctx.errors.append(
                    f"Arch '{name}': observer convergence path "
                    f"'{a.observer_convergence}' not found"
                )
            else:
                convergence_fn = compiled[a.observer_convergence]
        if a.observer_loss:
            if a.observer_loss not in compiled:
                ctx.errors.append(
                    f"Arch '{name}': observer loss path "
                    f"'{a.observer_loss}' not found"
                )
            else:
                loss_fn = compiled[a.observer_loss]
        if convergence_fn is not None:
            data.observer_convergence = convergence_fn
        if loss_fn is not None:
            data.observer_loss = loss_fn

        # Compile state_fields into a Hydra TypeRecord
        if a.state_fields is not None:
            from .parser import SortDecl as _SD
            state_name = f"_state_{name}"
            state_sort_defs = {state_name: _SD(state_name, a.state_fields)}
            data.state_type = sort_to_type(state_name, state_sort_defs)

        # Step protocol: generate coalgebra cell from step declarations
        step_enter = a.step_enter
        step_emit = a.step_emit
        step_compute = a.step_compute
        if step_enter is not None or step_emit is not None:
            step_ok = True
            if step_enter and step_enter not in compiled:
                ctx.errors.append(
                    f"Arch '{name}': step enter morphism '{step_enter}' not found"
                )
                step_ok = False
            if step_emit and step_emit not in compiled:
                ctx.errors.append(
                    f"Arch '{name}': step emit morphism '{step_emit}' not found"
                )
                step_ok = False
            if step_compute and step_compute not in compiled:
                ctx.errors.append(
                    f"Arch '{name}': step compute morphism '{step_compute}' not found"
                )
                step_ok = False
            if step_ok:
                enter_fn = compiled.get(step_enter) if step_enter else None
                emit_fn = compiled.get(step_emit) if step_emit else None

                if step_compute:
                    case_morphism_fn = compiled[step_compute]
                else:
                    case_morphism_fn = None
                    for c in case_list:
                        if c.morphisms:
                            case_morphism_fn = chain([compiled[m] for m in c.morphisms]) \
                                if len(c.morphisms) > 1 else compiled[c.morphisms[0]]
                            break

                step_case_name = None
                for c in case_list:
                    if c.recursive > 0:
                        step_case_name = c.name
                        break
                if step_case_name is None:
                    step_case_name = case_list[0].name if case_list else 'step'

                data.coalgebra_cell = _build_step_cell(enter_fn, emit_fn, case_morphism_fn,
                                                        step_case_name)

        compiled_archs[name] = data

    return compiled_archs


# ---------------------------------------------------------------------------
# Top-level compiler helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Top-level compiler
# ---------------------------------------------------------------------------

def compile(source: str | Path | DSLSource, namespace: dict,
            backend: Backend = None) -> ArchDef:
    """Compile DSL source into an ArchDef."""
    if isinstance(source, Path) or (isinstance(source, str) and source.endswith('.ua')):
        source = Path(source).read_text()
    ast = parse(source) if isinstance(source, str) else source

    ctx = CompilationContext(ast=ast, namespace=namespace,
                             backend=backend if backend is not None else NUMPY_BACKEND)

    sort_defs = {s.name: s for s in ast.sorts}

    # Phase 2: Resolve semirings
    ctx.semiring_contract = {}
    ctx.semiring_compiler = {}
    ctx.semiring_arity    = {}
    ctx.semiring_plus     = {}
    ctx.semiring_times    = {}

    for sd in ast.semirings:
        ctx.semiring_arity[sd.name] = sd.arity

        if sd.compiler is not None:
            try:
                ctx.semiring_compiler[sd.name] = _resolve(sd.compiler, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Semiring '{sd.name}': cannot resolve compiler '{sd.compiler}': {e}")

        if sd.plus is not None and sd.times is not None:
            # Algebraic declaration — resolve ops through backend first, then namespace
            try:
                plus_fn  = _resolve_semiring_op(sd.plus,  ctx.backend, namespace)
                times_fn = _resolve_semiring_op(sd.times, ctx.backend, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Semiring '{sd.name}': cannot resolve plus/times: {e}")
                continue

            ctx.semiring_plus[sd.name]    = plus_fn
            ctx.semiring_times[sd.name]   = times_fn
            ctx.semiring_contract[sd.name] = _make_semiring_contract(plus_fn, times_fn)

            if sd.zero is not None and sd.one is not None:
                try:
                    zero_val = _resolve_semiring_op(sd.zero, ctx.backend, namespace)
                    one_val  = _resolve_semiring_op(sd.one,  ctx.backend, namespace)
                    if isinstance(zero_val, str):
                        zero_val = float(zero_val)
                    if isinstance(one_val, str):
                        one_val = float(one_val)
                except Exception:
                    try:
                        zero_val = float(sd.zero)
                        one_val  = float(sd.one)
                    except ValueError:
                        zero_val = one_val = None

                if zero_val is not None:
                    _check_semiring_axioms(
                        sd.name, plus_fn, times_fn, zero_val, one_val, ctx.warnings
                    )

        elif sd.contract is not None:
            # Legacy: user-provided fused contract function
            try:
                ctx.semiring_contract[sd.name] = _resolve(sd.contract, namespace)
            except (NameError, AttributeError) as e:
                ctx.errors.append(f"Semiring '{sd.name}': cannot resolve contract '{sd.contract}': {e}")
        else:
            ctx.errors.append(
                f"Semiring '{sd.name}': declare 'contract' OR both 'plus' and 'times'"
            )

    sole = ast.semirings[0].name if len(ast.semirings) == 1 else None
    for m in ast.morphisms:
        if m.semiring is None:
            ctx.morphism_to_semiring[m.name] = '_bridge'
        elif m.semiring == '_default':
            ctx.morphism_to_semiring[m.name] = sole if sole else '_default'
        elif m.semiring not in ctx.semiring_contract:
            ctx.errors.append(
                f"Morphism '{m.name}' references undeclared semiring '{m.semiring}'"
            )
        else:
            ctx.morphism_to_semiring[m.name] = m.semiring

    # Phase 3: Compile morphisms
    ctx.compiled, ctx.specs, ctx.equations, ctx.morphism_terms = _compile_morphisms(
        ast.morphisms, ctx.morphism_to_semiring, ctx.semiring_contract,
        ctx.semiring_compiler, ctx.semiring_arity, namespace, ctx,
    )

    # Build template registry for parameterized morphisms
    ctx.templates = {
        m.name: m for m in ast.morphisms if m.template_param is not None
    }

    # Build Hydra sort types for type-checked sort validation
    morph_sorts = {s for m in ast.morphisms for s in (m.src_sort, m.tgt_sort)}
    ctx.sort_types = sort_types_from_defs(sort_defs, morph_sorts)

    # Build Hydra primitives for tensor ops (skip if errors already accumulated —
    # register_primitives will re-raise on unresolvable ops)
    if not ctx.errors:
        ctx.hydra_primitives = register_primitives(
            ctx.semiring_contract,
            ctx.morphism_to_semiring,
            ctx.equations,
            {m.name: {'op': m.op, 'arity': m.arity} for m in ast.morphisms},
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
    _compile_fans(ast.fans, ctx.compiled, namespace, ctx, ctx.backend)
    ctx.fan_terms = {
        f.name: fan_to_term(f.name, f.branches) for f in ast.fans
    }

    # Phase 5: Compile paths
    ctx.coercion_grades = _close_coercions(ast.coercions) if ast.coercions else None
    ctx.path_morphisms = _compile_paths(
        ast.paths, ctx.compiled, ctx.specs, ctx.morphism_to_semiring,
        ctx.templates, ctx.equations, namespace, ctx.semiring_arity,
        ctx, ctx.sort_types, ctx.coercion_grades, ast.sort_threshold,
    )
    ctx.path_terms = {
        p.name: path_to_term(p.name, ctx.path_morphisms.get(p.name, p.morphisms), p.residual)
        for p in ast.paths
    }

    # Phase 6: Compile archs
    accumulate_specs = {
        name: (spec.accumulate, spec.accumulate_fields)
        for name, spec in ctx.specs.items()
        if spec.accumulate is not None
    }
    compiled_archs = _compile_archs(ast.archs, ctx.compiled, namespace, accumulate_specs, ctx, ctx.backend)

    # Phase 7: Build Hydra arch terms and union types
    for a in ast.archs:
        cases = a.cases or []
        ctx.arch_terms[a.name] = arch_to_term(
            a.name,
            cases=cases,
            step_enter=a.step_enter,
            step_emit=a.step_emit,
        )
        if cases:
            _ndarray = _types.variable("ua.tensor.NDArray")
            _case_fields = []
            for c in cases:
                fs = [_types.field("data", _types.list_(_ndarray))]
                if c.recursive > 0:
                    fs.append(_types.field("children", _types.list_(_types.variable("ua.engine.TreeNode"))))
                if c.output > 0:
                    fs.append(_types.field("output", _ndarray))
                _case_fields.append(_types.field(c.name, _types.record_with_name(_Name("ua.engine.case." + c.name), fs)))
            ctx.arch_types[f"{a.name}.cases"] = _types.union(_case_fields)

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
