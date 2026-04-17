#!/usr/bin/env python3
"""
Citation consistency checker.

Reads citation keys from research/bibliography.md (lines containing <!-- [key] -->)
and scans Python source files for cite{KEY} markers in docstrings and comments.
Reports any keys used in code that are not registered in the bibliography.

Usage:
    python tools/check_citations.py            # check all .py files
    python tools/check_citations.py core/      # check a specific directory

Exit code 0 = clean. Exit code 1 = unknown keys found (blocks commit).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BIBLIOGRAPHY = ROOT / "research" / "bibliography.md"

KEY_IN_BIB   = re.compile(r'<!--\s*\[(\w+)\]\s*-->')
KEY_IN_CODE  = re.compile(r'cite\{(\w+)\}')


def load_bib_keys(path: Path) -> set[str]:
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        keys.update(KEY_IN_BIB.findall(line))
    return keys


def scan_files(paths: list[Path]) -> dict[str, list[tuple[Path, int]]]:
    """Return mapping key -> [(file, lineno), ...] for every cite{key} found."""
    found: dict[str, list[tuple[Path, int]]] = {}
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for key in KEY_IN_CODE.findall(line):
                found.setdefault(key, []).append((path, lineno))
    return found


def collect_py_files(roots: list[str]) -> list[Path]:
    paths = []
    for r in roots:
        p = Path(r)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(p.rglob("*.py"))
    # exclude venv, __pycache__, and this script itself
    self_path = Path(__file__).resolve()
    return [p for p in paths
            if "venv" not in p.parts
            and ".venv" not in p.parts
            and "__pycache__" not in p.parts
            and p.resolve() != self_path]


def main() -> int:
    if not BIBLIOGRAPHY.exists():
        print(f"error: bibliography not found at {BIBLIOGRAPHY}", file=sys.stderr)
        return 1

    bib_keys = load_bib_keys(BIBLIOGRAPHY)

    search_roots = sys.argv[1:] or [str(ROOT)]
    py_files = collect_py_files(search_roots)

    used_keys = scan_files(py_files)
    unknown = {k: locs for k, locs in used_keys.items() if k not in bib_keys}

    if not unknown:
        print(f"ok: {len(used_keys)} citation(s) checked against "
              f"{len(bib_keys)} bibliography entries.")
        return 0

    print(f"error: {len(unknown)} unknown citation key(s):\n")
    for key, locs in sorted(unknown.items()):
        for path, lineno in locs:
            rel = path.relative_to(ROOT)
            print(f"  cite{{{key}}}  {rel}:{lineno}")
    print(f"\nAdd missing keys to research/bibliography.md with the format:")
    print(f"  - Author, A. (year). Title. Journal. <!-- [key] -->")
    return 1


if __name__ == "__main__":
    sys.exit(main())
