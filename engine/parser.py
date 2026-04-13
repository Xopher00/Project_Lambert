"""
parser.py — DSL parser.

Turns DSL source text into a DSLSource AST.

Syntax
------

    semiring <n>:
        contract = <dotted.name>
        compiler = <dotted.name>

    sort <n>, <n>, ...

    leg <n> : <src> -> <tgt>  via "<eq>"  [using <semiring>]  [op <dotted.name>]
              [transform <dotted.name>]  [compiler <dotted.name>]  [bridge]

    path <n> = <leg> <leg> ...

    fan <n> = <leg> & <leg> & ...  [merge dict|meet|join|<dotted.name>]

    functor <n>:
        [cell = <dotted.name>]
        case <n>: recursive=<int>  data=<int>  [output=<0|1>]  [cell=<dotted.name>]

    arch <n>:
        algebra:
            case <n>: recursive=<int>  data=<int>  [cell=<dotted.name>]
        coalgebra:
            [cell = <dotted.name>]
            case <n>: recursive=<int>  data=<int>  [output=<0|1>]  [cell=<dotted.name>]
"""

from __future__ import annotations

import re

from .decl import (
    SemiringDecl, LegDecl, PathDecl, FanDecl,
    CaseDecl, FunctorDecl, ArchDecl, DSLSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_BOUNDARY = re.compile(r'^(semiring|sort|leg|path|fan|functor|arch)\b')


def _strip_comments(source: str) -> str:
    return '\n'.join(line.split('#')[0].rstrip() for line in source.splitlines())


def _parse_case_line(inner: str) -> CaseDecl | None:
    """Parse a 'case <name>: <attrs>' line, or return None if not a match."""
    cm = re.match(r'^case\s+(\w+)\s*:\s*(.+)$', inner)
    if not cm:
        return None
    case_name = cm.group(1)
    attrs_str = cm.group(2)
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
    return CaseDecl(
        case_name, attrs['recursive'], attrs['data'],
        attrs.get('output', 0), case_cell,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse(source: str) -> DSLSource:
    source = _strip_comments(source)
    semirings: list[SemiringDecl] = []
    sorts:     list[str]          = []
    legs:      list[LegDecl]      = []
    paths:     list[PathDecl]     = []
    fans:      list[FanDecl]      = []
    functors:  list[FunctorDecl]  = []
    archs:     list[ArchDecl]     = []

    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # semiring <n>:
        m = re.match(r'^semiring\s+(\w+)\s*:', line)
        if m:
            sr_name = m.group(1)
            contract_val = None
            compiler_val = None
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
                i += 1
            if contract_val is None:
                raise SyntaxError(f"semiring '{sr_name}' must declare 'contract'")
            semirings.append(SemiringDecl(sr_name, contract_val, compiler_val))
            continue

        # sort <n>, <n>, ...
        m = re.match(r'^sort\s+(.+)$', line)
        if m:
            sorts.extend(s.strip() for s in m.group(1).split(',') if s.strip())
            i += 1
            continue

        # leg <n> : <src> -> <tgt>  via "<eq>"  [clauses...]
        m = re.match(
            r'^leg\s+(\w+)\s*:\s*(\w+)\s*->\s*(\w+)\s+via\s+"([^"]+)"(.*)$',
            line,
        )
        if m:
            name     = m.group(1)
            src_sort = m.group(2)
            tgt_sort = m.group(3)
            equation = m.group(4)
            rest     = m.group(5).strip()

            semiring:       str | None = '_default'
            op_name:        str | None = None
            transform_name: str | None = None
            compiler_name:  str | None = None

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
                if clause == 'bridge':
                    semiring = None
                    continue
                raise SyntaxError(
                    f"leg '{name}': unrecognised clause {clause!r}; "
                    f"expected 'using <semiring>', 'op <dotted.name>', "
                    f"'transform <dotted.name>', 'compiler <dotted.name>', or 'bridge'"
                )

            legs.append(LegDecl(name, src_sort, tgt_sort, equation, semiring,
                                op_name, transform_name, compiler_name))
            i += 1
            continue

        # path <n> = <leg> <leg> ...
        m = re.match(r'^path\s+(\w+)\s*=\s*(.+)$', line)
        if m:
            paths.append(PathDecl(m.group(1), m.group(2).split()))
            i += 1
            continue

        # fan <n> = <leg> & <leg> & ...  [merge <mode>]
        m = re.match(r'^fan\s+(\w+)\s*=\s*(.+)$', line)
        if m:
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
            fans.append(FanDecl(fan_name, branches, merge_mode))
            i += 1
            continue

        # functor <n>: (legacy — still accepted)
        m = re.match(r'^functor\s+(\w+)\s*:', line)
        if m:
            func_name = m.group(1)
            cases: list[CaseDecl] = []
            cell_name: str | None = None
            i += 1
            while i < len(lines):
                inner = lines[i].strip()
                if not inner:
                    i += 1
                    continue
                if _BLOCK_BOUNDARY.match(inner):
                    break
                cell_m = re.match(r'^cell\s*=\s*(.+)$', inner)
                if cell_m:
                    cell_name = cell_m.group(1).strip()
                    i += 1
                    continue
                cd = _parse_case_line(inner)
                if cd:
                    cases.append(cd)
                    i += 1
                    continue
                i += 1
            if not cases:
                raise SyntaxError(f"functor '{func_name}' must declare at least one case")
            functors.append(FunctorDecl(func_name, cases, cell_name))
            continue

        # arch <n>:
        m = re.match(r'^arch\s+(\w+)\s*:', line)
        if m:
            arch_name = m.group(1)
            alg_cases:  list[CaseDecl] | None = None
            coalg_cases: list[CaseDecl] | None = None
            alg_cell:   str | None = None
            coalg_cell: str | None = None
            i += 1
            while i < len(lines):
                inner = lines[i].strip()
                if not inner:
                    i += 1
                    continue
                if _BLOCK_BOUNDARY.match(inner):
                    break
                mode_m = re.match(r'^(algebra|coalgebra)\s*:', inner)
                if mode_m:
                    mode = mode_m.group(1)
                    cases = []
                    cell_name = None
                    i += 1
                    while i < len(lines):
                        sub = lines[i].strip()
                        if not sub:
                            i += 1
                            continue
                        if _BLOCK_BOUNDARY.match(sub):
                            break
                        if re.match(r'^(algebra|coalgebra)\s*:', sub):
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
            archs.append(ArchDecl(
                arch_name, alg_cases, coalg_cases, alg_cell, coalg_cell,
            ))
            continue

        raise SyntaxError(f"Unrecognised DSL line: {line!r}")

    return DSLSource(semirings, sorts, legs, paths, fans, functors, archs)
