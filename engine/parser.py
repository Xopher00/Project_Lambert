"""
parser.py — DSL parser.

Turns DSL source text into a DSLSource AST.

Syntax
------

    semiring <n>:
        contract = <dotted.name>
        compiler = <dotted.name>
        arity = binary|ternary

    sort <n>, <n>, ...
    sort <n>(<field>: <type>, ...)

    morphism <n> : <src> -> <tgt>  via "<eq>"  [using <semiring>]  [op <dotted.name>]
                   [transform <dotted.name>]  [compiler <dotted.name>]  [bridge]
                   [arity unary|binary|pointwise|ternary]  [accumulate cat]

    path <n> = <morphism> [<fan>] <morphism> ...  [residual]  [normed <morphism>]

    fan <n> = <morphism> & <morphism> & ...  [merge dict|meet|join|<dotted.name>]

    arch <n>:
        algebra:
            case <n>: recursive=<int>  data=<int>  [cell=<dotted.name>]
                      [morphisms = <m1> <m2> ...]
        coalgebra:
            [cell = <dotted.name>]
            case <n>: recursive=<int>  data=<int>  [output=<0|1>]  [cell=<dotted.name>]
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
    """Parse a 'case <name>: <attrs>' line, or return None if not a match."""
    cm = re.match(r'^case\s+(\w+)\s*:\s*(.+)$', inner)
    if not cm:
        return None
    case_name = cm.group(1)
    attrs_str = cm.group(2)

    # Extract 'morphisms = name1 name2 ...' or legacy 'legs = name1 name2 ...'
    case_morphisms: list[str] | None = None
    lm = re.search(
        r'\b(?:morphisms|legs)\s*=\s*((?:\w+\s*)+?)(?=\s+\w+\s*=|\s*$)',
        attrs_str,
    )
    if lm:
        case_morphisms = lm.group(1).split()
        attrs_str = attrs_str[:lm.start()] + attrs_str[lm.end():]

    attrs: dict[str, int] = {}
    case_cell: str | None = None
    for kv in re.findall(r'(\w+)\s*=\s*([\w.]+)', attrs_str):
        if kv[0] == 'cell':
            case_cell = kv[1]
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
        sm = re.match(r'^(\w+)\(([^)]*)\)', rest)
        if sm:
            name = sm.group(1)
            fields_str = sm.group(2).strip()
            fields = {}
            if fields_str:
                for pair in fields_str.split(','):
                    pair = pair.strip()
                    fm = re.match(r'^(\w+)\s*:\s*(\w+)$', pair)
                    if not fm:
                        raise SyntaxError(
                            f"sort '{name}': invalid field declaration {pair!r}"
                        )
                    fields[fm.group(1)] = fm.group(2)
            results.append(SortDecl(name, fields if fields else None))
            rest = rest[sm.end():]
            continue
        # Opaque sort: plain name
        nm = re.match(r'^(\w+)', rest)
        if nm:
            results.append(SortDecl(nm.group(1), None))
            rest = rest[nm.end():]
            continue
        break
    return results


# ---------------------------------------------------------------------------
# Block sub-handlers
# ---------------------------------------------------------------------------

def _parse_semiring(lines: list[str], i: int) -> tuple[SemiringDecl, int]:
    """Parse a 'semiring <n>:' block starting at line index i."""
    m = re.match(r'^semiring\s+(\w+)\s*:', lines[i].strip())
    sr_name = m.group(1)
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
        cm = re.match(r'^contract\s*=\s*(.+)$', inner)
        if cm:
            contract_val = cm.group(1).strip()
            i += 1
            continue
        cp = re.match(r'^compiler\s*=\s*(.+)$', inner)
        if cp:
            compiler_val = cp.group(1).strip()
            i += 1
            continue
        ap = re.match(r'^arity\s*=\s*(binary|ternary)$', inner)
        if ap:
            sr_arity = ap.group(1)
            i += 1
            continue
        i += 1
    if contract_val is None:
        raise SyntaxError(f"semiring '{sr_name}' must declare 'contract'")
    return SemiringDecl(sr_name, contract_val, compiler_val, sr_arity), i


def _parse_morphism(line: str) -> MorphismDecl:
    """Parse a single 'morphism <n> : <src> -> <tgt>  via "<eq>"  [clauses...]' line."""
    m = re.match(
        r'^(?:morphism|leg)\s+(\w+)\s*:\s*(\w+)\s*->\s*(\w+)\s+via\s+"([^"]+)"(.*)$',
        line,
    )
    name     = m.group(1)
    src_sort = m.group(2)
    tgt_sort = m.group(3)
    equation = m.group(4)
    rest     = m.group(5).strip()

    semiring:       str | None = '_default'
    op_name:        str | None = None
    transform_name: str | None = None
    compiler_name:  str | None = None
    arity_val:      str = 'binary'
    accumulate_val: str | None = None

    for clause in re.split(r'\s{2,}', rest):
        clause = clause.strip()
        if not clause:
            continue
        um = re.match(r'^using\s+(\w+)$', clause)
        if um:
            semiring = um.group(1)
            continue
        om = re.match(r'^op\s+([\w.]+)$', clause)
        if om:
            op_name = om.group(1)
            continue
        tm = re.match(r'^transform\s+([\w.]+)$', clause)
        if tm:
            transform_name = tm.group(1)
            continue
        cpm = re.match(r'^compiler\s+([\w.]+)$', clause)
        if cpm:
            compiler_name = cpm.group(1)
            continue
        am = re.match(r'^arity\s+(unary|binary|pointwise|ternary)$', clause)
        if am:
            arity_val = am.group(1)
            continue
        acm = re.match(r'^accumulate\s+(\w+)$', clause)
        if acm:
            accumulate_val = acm.group(1)
            continue
        if clause == 'bridge':
            semiring = None
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
        arity_val, accumulate_val,
    )


def _parse_path(line: str) -> PathDecl:
    """Parse a single 'path <n> = <morphisms...>  [residual]  [normed <m>]' line."""
    m = re.match(r'^path\s+(\w+)\s*=\s*(.+)$', line)
    path_name = m.group(1)
    rhs = m.group(2).strip()
    residual_flag = False
    normed_name = None
    nm = re.search(r'\s+normed\s+(\w+)\s*$', rhs)
    if nm:
        normed_name = nm.group(1)
        rhs = rhs[:nm.start()]
    rm = re.search(r'\s+residual\s*$', rhs)
    if rm:
        residual_flag = True
        rhs = rhs[:rm.start()]
    return PathDecl(path_name, rhs.split(), residual_flag, normed_name)


def _parse_fan(line: str) -> FanDecl:
    """Parse a single 'fan <n> = <m> & <m> & ...  [merge <mode>]' line."""
    m = re.match(r'^fan\s+(\w+)\s*=\s*(.+)$', line)
    if not m:
        # Extract name for the error message even when RHS is missing
        nm = re.match(r'^fan\s+(\w+)', line)
        fan_name = nm.group(1) if nm else '?'
        raise SyntaxError(f"fan '{fan_name}': no branches declared")
    fan_name = m.group(1)
    rhs = m.group(2).strip()
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
    m = re.match(r'^arch\s+(\w+)\s*:', lines[i].strip())
    arch_name = m.group(1)
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
        obs_m = re.match(r'^observer\s*:', inner)
        if obs_m:
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
                conv_m = re.match(r'^convergence\s*=\s*(\w+)$', sub)
                if conv_m:
                    obs_convergence = conv_m.group(1)
                    i += 1
                    continue
                loss_m = re.match(r'^loss\s*=\s*(\w+)$', sub)
                if loss_m:
                    obs_loss = loss_m.group(1)
                    i += 1
                    continue
                i += 1
            continue
        mode_m = re.match(r'^(algebra|coalgebra)\s*:', inner)
        if mode_m:
            mode = mode_m.group(1)
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
                cell_m2 = re.match(r'^cell\s*=\s*(.+)$', sub)
                if cell_m2:
                    cell_name = cell_m2.group(1).strip()
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
            m = re.match(r'^sort\s+(.+)$', line)
            sorts.extend(_parse_sort_items(m.group(1)))
            i += 1
            continue

        if re.match(r'^(?:morphism|leg)\s+\w+\s*:', line):
            morphisms.append(_parse_morphism(line))
            i += 1
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
            m = re.match(r'^functor\s+(\w+)\s*:', line)
            fname = m.group(1) if m else '?'
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
