"""
Turns DSL source text into an abstract syntax tree.

Parses a flat, indentation-aware text format into a list of typed
declarations (SemiringDecl, SortDecl, MorphismDecl, etc.) collected
in a DSLSource object. The parser is line-oriented: each top-level
keyword (semiring, sort, morphism, path, fan, arch) starts a new
declaration. Morphism declarations support multi-line continuation
via indented sub-clauses.

Key functions:
  parse           — entry point: source string → DSLSource AST
  _parse_semiring — block parser for 'semiring <n>:' declarations
  _parse_morphism — single-line parser for morphism declarations
  _parse_path     — single-line parser for path compositions
  _parse_fan      — single-line parser for fan-out declarations
  _parse_arch     — block parser for 'arch <n>:' declarations

Depends on: decl.py (AST dataclasses)

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
        algebra:
            <n>: leaf  data=<int>  [cell=<dotted.name>]            (compact)
            <n>: node  data=<int>  [morphisms = <m1> <m2> ...]     (compact)
            case <n>: recursive=<int>  data=<int>  ...             (explicit)
        coalgebra:
            [cell = <dotted.name>]
            <n>: node  data=<int>  [output=<0|1>]
        observer:
            convergence = <path_name>
            loss = <path_name>
"""

from __future__ import annotations

import re

from .decl import (
    SemiringDecl, MorphismDecl, PathDecl, FanDecl,
    CaseDecl, ArchDecl, DSLSource, SortDecl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_BOUNDARY = re.compile(
    r'^(semiring|sort|morphism|leg|path|fan|functor|arch)\b'
)


def _strip_comments(source: str) -> str:
    return '\n'.join(line.split('#')[0].rstrip() for line in source.splitlines())


def _parse_case_line(inner: str) -> CaseDecl | None:
    """Parse a case declaration, or return None if not a match.

    Accepts two forms:
      case <name>: <attrs>     (original, with 'case' keyword)
      <name>: <attrs>          (compact, bare name)
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
    case_cell: str | None = None
    case_iterate: str | None = None
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
    if case_cell is not None and case_morphisms is not None:
        raise SyntaxError(
            f"case '{case_name}': cannot specify both 'cell' and 'morphisms'"
        )
    return CaseDecl(
        case_name, attrs['recursive'], attrs['data'],
        attrs.get('output', 0), case_cell, case_morphisms,
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
    """Parse a 'semiring <n>:' block starting at line index i."""
    match_header = re.match(r'^semiring\s+(\w+)\s*:', lines[i].strip())
    sr_name = match_header.group(1)
    contract_val = None
    compiler_val = None
    sr_arity = 'binary'
    i += 1
    while i < len(lines):
        inner = lines[i].strip()
        if not inner:
            i += 1
            continue
        if _BLOCK_BOUNDARY.match(inner):
            break
        match_contract = re.match(r'^contract\s*=\s*(.+)$', inner)
        if match_contract:
            contract_val = match_contract.group(1).strip()
            i += 1
            continue
        match_compiler = re.match(r'^compiler\s*=\s*(.+)$', inner)
        if match_compiler:
            compiler_val = match_compiler.group(1).strip()
            i += 1
            continue
        match_arity = re.match(r'^arity\s*=\s*(binary|ternary)$', inner)
        if match_arity:
            sr_arity = match_arity.group(1)
            i += 1
            continue
        i += 1
    if contract_val is None:
        raise SyntaxError(f"semiring '{sr_name}' must declare 'contract'")
    return SemiringDecl(sr_name, contract_val, compiler_val, sr_arity), i


def _parse_morphism(line: str) -> MorphismDecl:
    """Parse a morphism declaration line.

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
    template_param = match_header.group(2)  # None if no [param]
    name     = match_header.group(1)
    src_sort = match_header.group(3)
    tgt_sort = match_header.group(4)
    equation = match_header.group(5)
    rest     = match_header.group(6).strip()

    semiring:       str | None = '_default'
    op_name:        str | None = None
    transform_name: str | None = None
    compiler_name:  str | None = None
    arity_val:      str = 'binary'
    accumulate_val:        str | None       = None
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
        raise SyntaxError(
            f"morphism '{name}': unrecognised clause {clause!r}; "
            f"expected 'using <semiring>', 'op <dotted.name>', "
            f"'transform <dotted.name>', 'compiler <dotted.name>', "
            f"'arity unary|binary|pointwise|ternary', or 'bridge'"
        )

    return MorphismDecl(
        name, src_sort, tgt_sort, equation, semiring,
        op_name, transform_name, compiler_name,
        arity_val, accumulate_val, accumulate_fields_val,
        template_param=template_param,
    )


def _parse_path(line: str) -> PathDecl:
    """Parse a single 'path <n> = <morphisms...>  [residual]  [normed <m>]' line."""
    match_header = re.match(r'^path\s+(\w+)\s*=\s*(.+)$', line)
    path_name = match_header.group(1)
    rhs = match_header.group(2).strip()
    residual_flag = False
    normed_name = None
    match_normed = re.search(r'\s+normed\s+(\w+)\s*$', rhs)
    if match_normed:
        normed_name = match_normed.group(1)
        rhs = rhs[:match_normed.start()]
    match_residual = re.search(r'\s+residual\s*$', rhs)
    if match_residual:
        residual_flag = True
        rhs = rhs[:match_residual.start()]
    return PathDecl(path_name, rhs.split(), residual_flag, normed_name)


def _parse_fan(line: str) -> FanDecl:
    """Parse a single 'fan <n> = <m> & <m> & ...  [merge <mode>]' line."""
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
    return FanDecl(fan_name, branches, merge_mode)


def _parse_arch(lines: list[str], i: int) -> tuple[ArchDecl, int]:
    """Parse an 'arch <n>:' block starting at line index i."""
    match_header = re.match(r'^arch\s+(\w+)\s*:', lines[i].strip())
    arch_name = match_header.group(1)
    alg_cases:  list[CaseDecl] | None = None
    coalg_cases: list[CaseDecl] | None = None
    alg_cell:   str | None = None
    coalg_cell: str | None = None
    obs_convergence: str | None = None
    obs_loss:        str | None = None
    i += 1
    while i < len(lines):
        inner = lines[i].strip()
        if not inner:
            i += 1
            continue
        if _BLOCK_BOUNDARY.match(inner):
            break
        match_observer = re.match(r'^observer\s*:', inner)
        if match_observer:
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if not sub:
                    i += 1
                    continue
                if _BLOCK_BOUNDARY.match(sub):
                    break
                if re.match(r'^(algebra|coalgebra|observer)\s*:', sub):
                    break
                match_convergence = re.match(r'^convergence\s*=\s*(\w+)$', sub)
                if match_convergence:
                    obs_convergence = match_convergence.group(1)
                    i += 1
                    continue
                match_loss = re.match(r'^loss\s*=\s*(\w+)$', sub)
                if match_loss:
                    obs_loss = match_loss.group(1)
                    i += 1
                    continue
                i += 1
            continue
        match_mode = re.match(r'^(algebra|coalgebra)\s*:', inner)
        if match_mode:
            mode = match_mode.group(1)
            cases: list[CaseDecl] = []
            cell_name: str | None = None
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if not sub:
                    i += 1
                    continue
                if _BLOCK_BOUNDARY.match(sub):
                    break
                if re.match(r'^(algebra|coalgebra|observer)\s*:', sub):
                    break
                match_cell = re.match(r'^cell\s*=\s*(.+)$', sub)
                if match_cell:
                    cell_name = match_cell.group(1).strip()
                    i += 1
                    continue
                cd = _parse_case_line(sub)
                if cd:
                    cases.append(cd)
                    i += 1
                    continue
                i += 1
            if not cases:
                raise SyntaxError(
                    f"arch '{arch_name}' {mode}: must declare at least one case"
                )
            if mode == 'algebra':
                alg_cases = cases
                alg_cell = cell_name
            else:
                coalg_cases = cases
                coalg_cell = cell_name
            continue
        i += 1
    if alg_cases is None and coalg_cases is None:
        raise SyntaxError(
            f"arch '{arch_name}' must declare algebra and/or coalgebra"
        )
    return ArchDecl(
        arch_name, alg_cases, coalg_cases, alg_cell, coalg_cell,
        obs_convergence, obs_loss,
    ), i


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse(source: str) -> DSLSource:
    source = _strip_comments(source)
    semirings: list[SemiringDecl]   = []
    sorts:     list[SortDecl]       = []
    morphisms: list[MorphismDecl]   = []
    paths:     list[PathDecl]       = []
    fans:      list[FanDecl]        = []
    archs:     list[ArchDecl]       = []

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
                if next_raw and next_raw[0] in (' ', '\t') and not _BLOCK_BOUNDARY.match(next_raw.strip()):
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

        if re.match(r'^functor\s+\w+\s*:', line):
            match_functor = re.match(r'^functor\s+(\w+)\s*:', line)
            fname = match_functor.group(1) if match_functor else '?'
            raise SyntaxError(
                f"'functor' keyword has been removed. "
                f"Replace 'functor {fname}:' with 'arch {fname}: algebra:' — "
                f"see CLAUDE.md for migration notes."
            )

        if re.match(r'^arch\s+\w+\s*:', line):
            decl, i = _parse_arch(lines, i)
            archs.append(decl)
            continue

        raise SyntaxError(f"Unrecognised DSL line: {line!r}")

    return DSLSource(semirings, sorts, morphisms, paths, fans, archs)
