"""
Turns DSL source text into a collection of TTerms.

Parses a flat, indentation-aware text format. Each top-level keyword
(semiring, sort, morphism, path, fan, arch) produces a typed TTerm.
SemiringDecl and SortDecl are the only remaining dataclasses; all other
constructs are TTerms collected in a DSLSource object. The parser
is line-oriented; morphism declarations support multi-line continuation
via indented sub-clauses.

Key functions:
  parse           — entry point: source string → DSLSource AST
  _parse_semiring — block parser for 'semiring <n>:' declarations
  _parse_morphism — single-line parser for morphism declarations → TTerm
  _parse_path     — single-line parser for path compositions → TTerm
  _parse_fan      — single-line parser for fan-out declarations → TTerm
  _parse_arch     — block parser for 'arch <n>:' declarations → TTerm

Syntax
------

    semiring <n>:
        contract = <dotted.name>
        compiler = <dotted.name>
        arity = binary|ternary

    sort <n>, <n>, ...
    sort <n>(<field>: <type>, ...)

    morphism <n> : <src> -> <tgt>  "<eq>"  <dotted.name>         (compact)
    morphism <n> : <src> -> <tgt>  via "<eq>"  op <dotted.name>  (explicit)

    Optional clauses (either form, separated by 2+ spaces or on continuation lines):
        [using <semiring>]  [transform <dotted.name>]  [compiler <dotted.name>]
        [arity unary|binary|pointwise|ternary]  [accumulate cat [on <f1>, <f2>]]
        [bridge]

    Multi-line morphism (continuation lines indented):
        morphism <n> : <src> -> <tgt>
            "<eq>"
            <dotted.name>
            arity unary

    Semiring block is optional when all morphisms have explicit op bindings.

    path <n> = <morphism> [<fan>] <morphism> ...  [residual]  [normed <morphism>]

    fan <n> = <morphism> & <morphism> & ...  [merge dict|meet|join|<dotted.name>]

    arch <n>:
        cases:                                                      (or 'algebra:' — alias)
            <n>: leaf  data=<int>  [cell=<dotted.name>]            (compact)
            <n>: node  data=<int>  [morphisms = <m1> <m2> ...]     (compact)
            case <n>: recursive=<int>  data=<int>  ...             (explicit)
        state:
            <field>: <type>
        step:
            enter = <morphism_name>
            emit  = <morphism_name>
        observer:
            convergence = <path_name>
            loss = <path_name>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# AST node declarations (moved here from decl.py)
# ---------------------------------------------------------------------------

@dataclass
class SemiringDecl:
    name:     str
    contract: str | None = None  # legacy fused contract (optional if plus/times present)
    compiler: str | None = None
    arity:    str = 'binary'
    plus:     str | None = None  # ⊕ operation name
    times:    str | None = None  # ⊗ operation name
    zero:     str | None = None  # additive identity (value or dotted name)
    one:      str | None = None  # multiplicative identity (value or dotted name)



@dataclass
class SortDecl:
    name:   str
    fields: dict[str, str] | None = None  # None = opaque

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, SortDecl) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        if isinstance(other, str):
            return self.name < other
        if isinstance(other, SortDecl):
            return self.name < other.name
        return NotImplemented


@dataclass
class SortCoercion:
    src:   str
    tgt:   str
    grade: float


@dataclass
class DSLSource:
    semirings:      list[SemiringDecl]
    sorts:          list[SortDecl]
    morphisms:      list  # list[TTerm] — ua.engine.Morphism records
    paths:          list  # list[TTerm] — ua.engine.Path records
    fans:           list  # list[TTerm] — ua.engine.Fan records
    archs:          list = field(default_factory=list)  # list[TTerm] — ua.engine.Arch records
    coercions:      list = field(default_factory=list)  # list[SortCoercion]
    sort_threshold: float = 1.0


# Deferred import: avoids circular dependency with engine.sorts
# (sorts.py imports SortDecl/SortCoercion from here; terms.py imports from sorts)
# SortDecl and SortCoercion are defined above, so this import is safe here.
from . import terms as _terms  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOP_LEVEL_DECL = re.compile(
    r'^(semiring|sort|morphism|leg|path|fan|arch|coerce|sort_threshold)\b'
)

_ARCH_SUB_HEADER = re.compile(
    r'^(cases|algebra|coalgebra|state|step|observer)\s*:'
)


def _scan_sub_block(
    lines: list[str], i: int, boundary_keywords: re.Pattern | None = None
) -> tuple[list[str], int]:
    """Collect non-empty stripped lines for a sub-block inside an arch.

    Starts at index *i* and stops when:
    - a top-level declaration keyword is seen, or
    - an arch-level sub-header keyword is seen, or
    - ``boundary_keywords`` matches the line (if provided), or
    - end of ``lines`` is reached.

    Returns ``(collected_lines, new_i)`` where ``new_i`` points to the first
    line *not* consumed (the boundary line itself, or end-of-list).
    """
    collected: list[str] = []
    while i < len(lines):
        sub = lines[i].strip()
        if not sub:
            i += 1
            continue
        if _TOP_LEVEL_DECL.match(sub):
            break
        if _ARCH_SUB_HEADER.match(sub):
            break
        if boundary_keywords and boundary_keywords.match(sub):
            break
        collected.append(sub)
        i += 1
    return collected, i


def _strip_comments(source: str) -> str:
    return '\n'.join(line.split('#')[0].rstrip() for line in source.splitlines())


def _parse_case_line(inner: str):
    """Parse a case declaration, or return None if not a match.

    Accepts two forms:
      case <name>: <attrs>     (original, with 'case' keyword)
      <name>: <attrs>          (compact, bare name)

    Returns a TTerm (ua.engine.Case) or None.
    """
    match_case = re.match(r'^case\s+(\w+)\s*:\s*(.+)$', inner)
    if not match_case:
        # Compact form: bare name without 'case' keyword
        match_case = re.match(r'^(\w+)\s*:\s*(.+)$', inner)
    if not match_case:
        return None
    case_name = match_case.group(1)
    attrs_str = match_case.group(2)

    # Handle leaf/node aliases before the attrs regex loop.
    # Negative lookbehind for '.' prevents matching dotted names like ops.leaf.
    # Negative lookahead for '=' prevents matching keys in key=value pairs.
    if re.search(r'(?<!\.)\bleaf\b(?!\s*=)', attrs_str):
        if 'recursive=' in attrs_str:
            raise SyntaxError(
                f"case '{case_name}': cannot use 'leaf' together with 'recursive='"
            )
        attrs_str = re.sub(r'(?<!\.)\bleaf\b(?!\s*=)', 'recursive=0', attrs_str, count=1)
    elif re.search(r'(?<!\.)\bnode\b(?!\s*=)', attrs_str):
        if 'recursive=' in attrs_str:
            raise SyntaxError(
                f"case '{case_name}': cannot use 'node' together with 'recursive='"
            )
        attrs_str = re.sub(r'(?<!\.)\bnode\b(?!\s*=)', 'recursive=1', attrs_str, count=1)

    # Extract 'morphisms = name1 name2 ...' or legacy 'legs = name1 name2 ...'
    case_morphisms: list[str] | None = None
    match_morphisms = re.search(
        r'\b(?:morphisms|legs)\s*=\s*((?:\w+\s*)+?)(?=\s+\w+\s*=|\s*$)',
        attrs_str,
    )
    if match_morphisms:
        case_morphisms = match_morphisms.group(1).split()
        attrs_str = attrs_str[:match_morphisms.start()] + attrs_str[match_morphisms.end():]

    attrs: dict[str, int] = {}
    case_cell: str = ""
    case_iterate: str = ""
    for kv in re.findall(r'(\w+)\s*=\s*([\w.]+)', attrs_str):
        if kv[0] == 'cell':
            case_cell = kv[1]
        elif kv[0] == 'iterate':
            case_iterate = kv[1]
        else:
            attrs[kv[0]] = int(kv[1])
    if 'recursive' not in attrs or 'data' not in attrs:
        raise SyntaxError(
            f"case '{case_name}': must declare 'recursive' and 'data'"
        )
    return _terms.case(
        name=case_name,
        recursive=attrs['recursive'],
        data=attrs['data'],
        output=attrs.get('output', 0),
        cell=case_cell,
        morphisms=case_morphisms,
        iterate=case_iterate,
    )


def _parse_sort_items(rhs: str) -> list[SortDecl]:
    """Parse the right-hand side of a sort declaration.

    Supports both opaque sorts (``sort i, j``) and structured sorts
    (``sort model(Wq: matrix, bq: vector), i``).
    """
    results = []
    rest = rhs
    while rest:
        rest = rest.lstrip(' ,')
        if not rest:
            break
        # Structured sort: name(field: type, ...)
        match_struct = re.match(r'^(\w+)\(([^)]*)\)', rest)
        if match_struct:
            name = match_struct.group(1)
            fields_str = match_struct.group(2).strip()
            fields = {}
            if fields_str:
                for pair in fields_str.split(','):
                    pair = pair.strip()
                    match_field = re.match(r'^(\w+)\s*:\s*(\w+)$', pair)
                    if not match_field:
                        raise SyntaxError(
                            f"sort '{name}': invalid field declaration {pair!r}"
                        )
                    fields[match_field.group(1)] = match_field.group(2)
            results.append(SortDecl(name, fields if fields else None))
            rest = rest[match_struct.end():]
            continue
        # Opaque sort: plain name
        match_name = re.match(r'^(\w+)', rest)
        if match_name:
            results.append(SortDecl(match_name.group(1), None))
            rest = rest[match_name.end():]
            continue
        break
    return results


# ---------------------------------------------------------------------------
# Block sub-handlers
# ---------------------------------------------------------------------------

def _parse_semiring(lines: list[str], i: int) -> tuple[SemiringDecl, int]:
    match_header = re.match(r'^semiring\s+(\w+)\s*:', lines[i].strip())
    sr_name = match_header.group(1)
    fields = {}
    i += 1
    while i < len(lines):
        inner = lines[i].strip()
        if not inner:
            i += 1; continue
        if _TOP_LEVEL_DECL.match(inner):
            break
        m = re.match(r'^(\w+)\s*=\s*(.+)$', inner)
        if m:
            fields[m.group(1)] = m.group(2).strip()
        i += 1

    contract_val = fields.get('contract')
    compiler_val = fields.get('compiler')
    sr_arity     = fields.get('arity', 'binary')
    plus_val     = fields.get('plus')
    times_val    = fields.get('times')
    zero_val     = fields.get('zero')
    one_val      = fields.get('one')

    if sr_arity not in ('binary', 'ternary'):
        raise SyntaxError(f"semiring '{sr_name}': arity must be 'binary' or 'ternary', got '{sr_arity}'")
    if contract_val is None and (plus_val is None or times_val is None):
        raise SyntaxError(
            f"semiring '{sr_name}': declare 'contract' OR both 'plus' and 'times'"
        )

    return SemiringDecl(
        sr_name, contract_val, compiler_val, sr_arity,
        plus_val, times_val, zero_val, one_val
    ), i


def _parse_morphism(line: str):
    """Parse a morphism declaration line, returning a TTerm.

    Accepts two forms:
      morphism <n> : <src> -> <tgt>  via "<eq>"  [clauses...]   (original)
      morphism <n> : <src> -> <tgt>  "<eq>"  [clauses...]       (compact — no 'via')

    Clauses are separated by two or more spaces. In the clause list, bare
    dotted names (e.g. ``ops.q_proj``) are treated as ``op`` bindings — the
    ``op`` keyword is optional.

    Multi-line morphisms should be joined by the caller (parse()) before
    calling this function.
    """
    # Try original form first: via "eq"
    match_header = re.match(
        r'^(?:morphism|leg)\s+(\w+)(?:\[(\w+)\])?\s*:\s*(\w+)\s*->\s*(\w+)\s+via\s+"([^"]+)"(.*)$',
        line,
    )
    if not match_header:
        # Compact form: bare "eq" without via keyword
        match_header = re.match(
            r'^(?:morphism|leg)\s+(\w+)(?:\[(\w+)\])?\s*:\s*(\w+)\s*->\s*(\w+)\s+"([^"]+)"(.*)$',
            line,
        )
    if not match_header:
        raise SyntaxError(f"Invalid morphism declaration: {line!r}")
    template_param = match_header.group(2) or ""  # "" if no [param]
    name     = match_header.group(1)
    src_sort = match_header.group(3)
    tgt_sort = match_header.group(4)
    equation = match_header.group(5)
    rest     = match_header.group(6).strip()

    semiring:       str | None = '_default'
    op_name:        str = ""
    transform_name: str = ""
    compiler_name:  str = ""
    arity_val:      str = 'binary'
    accumulate_val:        str = ""
    accumulate_fields_val: list[str] | None = None

    for clause in re.split(r'\s{2,}', rest):
        clause = clause.strip()
        if not clause:
            continue
        match_using = re.match(r'^using\s+(\w+)$', clause)
        if match_using:
            semiring = match_using.group(1)
            continue
        match_op = re.match(r'^op\s+([\w.]+)$', clause)
        if match_op:
            op_name = match_op.group(1)
            continue
        match_transform = re.match(r'^transform\s+([\w.]+)$', clause)
        if match_transform:
            transform_name = match_transform.group(1)
            continue
        match_compiler = re.match(r'^compiler\s+([\w.]+)$', clause)
        if match_compiler:
            compiler_name = match_compiler.group(1)
            continue
        match_arity = re.match(r'^arity\s+(unary|binary|pointwise|ternary)$', clause)
        if match_arity:
            arity_val = match_arity.group(1)
            continue
        match_accumulate = re.match(r'^accumulate\s+(\w+)(?:\s+on\s+(.+))?$', clause)
        if match_accumulate:
            accumulate_val = match_accumulate.group(1)
            fields_str = match_accumulate.group(2)
            if fields_str:
                accumulate_fields_val = [f.strip() for f in fields_str.split(',')]
            continue
        if clause == 'bridge':
            semiring = None
            continue
        # Bare dotted name → implicit op binding (e.g. "ops.q_proj" without "op" prefix)
        if re.match(r'^[\w.]+$', clause) and '.' in clause:
            op_name = clause
            continue
        raise SyntaxError(f"morphism '{name}': unrecognised clause {clause!r}")

    return _terms.morphism(
        name=name,
        src=src_sort,
        tgt=tgt_sort,
        arity=arity_val,
        template_params=[template_param] if template_param else [],
        semiring=semiring if semiring else "",
        equation=equation,
        op=op_name,
        transform=transform_name,
        compiler_name=compiler_name,
        accumulate=accumulate_val,
        accumulate_fields=accumulate_fields_val,
        template_param=template_param,
    )


def _parse_path(line: str):
    """Parse a single 'path <n> = <morphisms...>  [residual]  [normed <m>]' line.

    Returns a TTerm (ua.engine.Path).
    """
    match_header = re.match(r'^path\s+(\w+)\s*=\s*(.+)$', line)
    path_name = match_header.group(1)
    rhs = match_header.group(2).strip()
    residual_flag = False
    normed_name = ""
    match_normed = re.search(r'\s+normed\s+(\w+)\s*$', rhs)
    if match_normed:
        normed_name = match_normed.group(1)
        rhs = rhs[:match_normed.start()]
    match_residual = re.search(r'\s+residual\s*$', rhs)
    if match_residual:
        residual_flag = True
        rhs = rhs[:match_residual.start()]
    return _terms.path(
        name=path_name,
        morphisms=rhs.split(),
        residual=residual_flag,
        normed=normed_name,
    )


def _parse_fan(line: str):
    """Parse a single 'fan <n> = <m> & <m> & ...  [merge <mode>]' line.

    Returns a TTerm (ua.engine.Fan).
    """
    match_header = re.match(r'^fan\s+(\w+)\s*=\s*(.+)$', line)
    if not match_header:
        # Extract name for the error message even when RHS is missing
        match_name = re.match(r'^fan\s+(\w+)', line)
        fan_name = match_name.group(1) if match_name else '?'
        raise SyntaxError(f"fan '{fan_name}': no branches declared")
    fan_name = match_header.group(1)
    rhs = match_header.group(2).strip()
    merge_match = re.search(r'\s+merge\s+(\S+)\s*$', rhs)
    if merge_match:
        merge_mode = merge_match.group(1)
        rhs = rhs[:merge_match.start()]
    else:
        merge_mode = 'dict'
    branches = [b.strip() for b in rhs.split('&') if b.strip()]
    if not branches:
        raise SyntaxError(f"fan '{fan_name}': no branches declared")
    return _terms.fan(name=fan_name, branches=branches, merge=merge_mode)


def _parse_arch(lines: list[str], i: int):
    """Parse an 'arch <n>:' block starting at line index i.

    Returns (TTerm, new_i) where TTerm is a ua.engine.Arch record.
    """
    match_header = re.match(r'^arch\s+(\w+)\s*:', lines[i].strip())
    arch_name = match_header.group(1)
    unified_cases: list | None = None  # list of case TTerms
    alg_cell:        str = ""
    obs_convergence: str = ""
    obs_loss:        str = ""
    state_fields: dict[str, str] | None = None
    step_enter:   str = ""
    step_emit:    str = ""
    step_compute: str = ""
    i += 1
    while i < len(lines):
        inner = lines[i].strip()
        if not inner:
            i += 1
            continue
        if _TOP_LEVEL_DECL.match(inner):
            break
        match_observer = re.match(r'^observer\s*:', inner)
        if match_observer:
            i += 1
            collected, i = _scan_sub_block(lines, i)
            for sub in collected:
                match_convergence = re.match(r'^convergence\s*=\s*(\w+)$', sub)
                if match_convergence:
                    obs_convergence = match_convergence.group(1)
                    continue
                match_loss = re.match(r'^loss\s*=\s*(\w+)$', sub)
                if match_loss:
                    obs_loss = match_loss.group(1)
            continue
        match_state = re.match(r'^state\s*:', inner)
        if match_state:
            state_fields = {}
            i += 1
            collected, i = _scan_sub_block(lines, i)
            for sub in collected:
                match_field = re.match(r'^(\w+)\s*:\s*(\w+(?:\[[\w,\s]+\])?)$', sub)
                if match_field:
                    state_fields[match_field.group(1)] = match_field.group(2)
            continue
        match_step = re.match(r'^step\s*:', inner)
        if match_step:
            i += 1
            collected, i = _scan_sub_block(lines, i)
            for sub in collected:
                match_enter = re.match(r'^enter\s*=\s*(\w+)$', sub)
                if match_enter:
                    step_enter = match_enter.group(1)
                    continue
                match_emit = re.match(r'^emit\s*=\s*(\w+)$', sub)
                if match_emit:
                    step_emit = match_emit.group(1)
                    continue
                match_compute = re.match(r'^compute\s*=\s*(\w+)$', sub)
                if match_compute:
                    step_compute = match_compute.group(1)
            continue
        if re.match(r'^coalgebra\s*:', inner):
            raise SyntaxError("'coalgebra:' is removed; use 'cases:' and 'step:'")
        match_mode = re.match(r'^(cases|algebra)\s*:', inner)
        if match_mode:
            mode = match_mode.group(1)
            cases: list = []
            cell_name: str = ""
            i += 1
            collected, i = _scan_sub_block(lines, i)
            for sub in collected:
                match_cell = re.match(r'^cell\s*=\s*(.+)$', sub)
                if match_cell:
                    cell_name = match_cell.group(1).strip()
                    continue
                cd = _parse_case_line(sub)
                if cd is not None:
                    cases.append(cd)
            if not cases:
                raise SyntaxError(
                    f"arch '{arch_name}' {mode}: must declare at least one case"
                )
            # Both cases: and algebra: are aliases for the unified endofunctor
            unified_cases = cases
            alg_cell = cell_name
            continue
        i += 1
    if unified_cases is None:
        raise SyntaxError(
            f"arch '{arch_name}' must declare cases or algebra"
        )
    # Derive state field name/type lists for the TTerm
    state_field_names = list(state_fields.keys()) if state_fields else None
    state_field_types = list(state_fields.values()) if state_fields else None
    term = _terms.arch(
        name=arch_name,
        cases=unified_cases,
        step_enter=step_enter if step_enter else None,
        step_emit=step_emit if step_emit else None,
        algebra_cell=alg_cell,
        observer_convergence=obs_convergence,
        observer_loss=obs_loss,
        step_compute=step_compute,
        state_field_names=state_field_names,
        state_field_types=state_field_types,
    )
    return term, i


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse(source: str) -> DSLSource:
    source = _strip_comments(source)
    semirings:      list[SemiringDecl] = []
    sorts:          list[SortDecl]    = []
    morphisms:      list              = []  # list of TTerm
    paths:          list              = []  # list of TTerm
    fans:           list              = []  # list of TTerm
    archs:          list              = []  # list of TTerm
    coercions:      list[SortCoercion] = []
    sort_threshold: float              = 1.0

    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if re.match(r'^semiring\s+\w+\s*:', line):
            decl, i = _parse_semiring(lines, i)
            semirings.append(decl)
            continue

        if re.match(r'^sort\s+', line):
            match_sort = re.match(r'^sort\s+(.+)$', line)
            sorts.extend(_parse_sort_items(match_sort.group(1)))
            i += 1
            continue

        if re.match(r'^(?:morphism|leg)\s+\w+(?:\[\w+\])?\s*:', line):
            # Join any indented continuation lines into the same logical line
            i += 1
            while i < len(lines):
                next_raw = lines[i]
                if next_raw and next_raw[0] in (' ', '\t') and not _TOP_LEVEL_DECL.match(next_raw.strip()):
                    line = line + '  ' + next_raw.strip()
                    i += 1
                else:
                    break
            morphisms.append(_parse_morphism(line))
            continue

        if re.match(r'^path\s+\w+\s*=', line):
            paths.append(_parse_path(line))
            i += 1
            continue

        if re.match(r'^fan\s+\w+\s*=', line):
            fans.append(_parse_fan(line))
            i += 1
            continue

        if re.match(r'^arch\s+\w+\s*:', line):
            decl, i = _parse_arch(lines, i)
            archs.append(decl)
            continue

        match_coerce = re.match(r'^coerce\s+(\w+)\s*->\s*(\w+)\s*:\s*([0-9]*\.?[0-9]+)$', line)
        if match_coerce:
            coercions.append(SortCoercion(
                src=match_coerce.group(1),
                tgt=match_coerce.group(2),
                grade=float(match_coerce.group(3)),
            ))
            i += 1
            continue

        match_threshold = re.match(r'^sort_threshold\s+([0-9]*\.?[0-9]+)$', line)
        if match_threshold:
            sort_threshold = float(match_threshold.group(1))
            i += 1
            continue

        raise SyntaxError(f"Unrecognised DSL line: {line!r}")

    return DSLSource(semirings, sorts, morphisms, paths, fans, archs, coercions, sort_threshold)
