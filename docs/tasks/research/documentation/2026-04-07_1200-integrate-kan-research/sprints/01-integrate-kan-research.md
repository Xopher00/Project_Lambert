# Sprint 1: Integrate Kan/CQL Research Notes

## Meta

- **PRD:** `../spec.md`
- **Sprint:** 1 of 3
- **Depends on:** None
- **Batch:** 1 (parallel with Sprint 2)
- **Model:** sonnet
- **Estimated effort:** S

## Objective

Merge the staged Kan extension / CQL framing additions from `research/plan/` into the live research notes and clear the staging area.

## File Boundaries

### Creates (new files)

- none

### Modifies (can touch)

- `research/core/tensor.md` — insert Kan extension paragraph + CQL table after the Shen & Tang paragraph in the Residuate section; update witness tracking note
- `research/theory.md` — extend query semantics section with Kan framing subsection
- `research/lattice/explorer.md` — extend generation gap section with Kan diagnosis
- `research/bibliography.md` — add Kan 1958, Schultz et al. 2017, Fong & Spivak 2019 entries after Shen & Tang in Category theory subsection
- `research/plan/index.md` — remove both pending entries

### Read-Only (reference but do NOT modify)

- `research/plan/kan.md` — source of all three research note insertions; use verbatim
- `research/plan/bibliography_additions.md` — source of bibliography entries and annotation text

### Shared Contracts (consume from PRD)

- Cite key format: `<!-- [keyname] -->` in bibliography.md; `cite{keyname}` inline in research .md files
- New keys: `kan1958`, `schultz2017`, `fong2019`

### Consumed Invariants

- Cite key consistency — after this sprint, `python tools/check_citations.py` must exit 0

## Tasks

- [ ] Read `research/plan/kan.md` in full to confirm exact insertion targets and text
- [ ] Read `research/plan/bibliography_additions.md` in full
- [ ] Insert the Kan extension paragraph into `research/core/tensor.md` immediately after the line "**Reference:** Shen, L., & Tang, X. (2021)..." in the Residuate section — the full block from `## Addition to core/tensor.md` in kan.md, including the CQL adjoint triple table and the two Reference lines at the end of that section
- [ ] In `research/theory.md`, insert the "### The Kan framing" subsection (and everything through the second `**Reference:**` line in the `## Addition to theory.md` section of kan.md) into the Query semantics section, after the existing "### What this unlocks" block
- [ ] In `research/lattice/explorer.md`, insert the "### The generation gap as a Kan extension problem" subsection (from `## Addition to lattice/explorer.md` in kan.md) into the "What the explorer does not do: the generation gap" section, after the existing two numbered structural absences
- [ ] In `research/bibliography.md`, insert the three new entries (Kan 1958, Schultz et al. 2017, Fong & Spivak 2019) into the Category theory subsection, after the existing Shen & Tang (2021) entry; use the annotation text from `bibliography_additions.md` adapted to the bold-name narrative style
- [ ] Update `research/plan/index.md` to remove the two pending entries (kan.md and bibliography_additions.md), leaving only the header and a note that both were merged
- [ ] Update the "Witness tracking" section in `research/core/tensor.md` to replace "is currently under review" with a note that provenance has been retired to `legacy/provenance/` and that lattice traversal is the forward path

## Acceptance Criteria

- [ ] `python tools/check_citations.py` exits 0
- [ ] `research/bibliography.md` contains all three new entries: `<!-- [kan1958] -->`, `<!-- [schultz2017] -->`, `<!-- [fong2019] -->`
- [ ] `research/core/tensor.md` contains the text "Kan extension" and the CQL adjoint triple table
- [ ] `research/theory.md` contains "right Kan" and "left Kan" in the query semantics section
- [ ] `research/lattice/explorer.md` generation gap section contains "right Kan direction" and "left Kan direction"
- [ ] `research/plan/index.md` contains no `- [kan.md]` or `- [bibliography_additions.md]` pending entries

## Verification

- [ ] `python tools/check_citations.py` (cite key consistency check)
- [ ] `grep -c "kan1958\|schultz2017\|fong2019" research/bibliography.md` returns 3 or more (keys present)
- [ ] `grep -c "Kan extension" research/core/tensor.md` returns at least 1

## Context

All insertion text is pre-written in `research/plan/kan.md`. The file clearly labels each addition with `## Addition to <file> — <section>`. Use those labels to locate the correct insertion point in each target file. The text should be inserted verbatim except:

1. The cite key style in kan.md uses `cite{key}` (no backslash, no curly-only) — this is correct for the project.
2. The bibliography entries should follow the bold-name annotation style visible in existing entries: open with a bold `**Author (year)**` sentence explaining the paper's relevance.

Insertion order within `research/core/tensor.md`: the Kan paragraph goes at the end of the Residuate section (after the Shen & Tang reference paragraph), before the `## Closure: transitive reachability` heading.

## Agent Notes (filled during execution)

- Assigned to: —
- Started: —
- Completed: —
- Decisions made: —
- Assumptions: —
- Issues found: —
