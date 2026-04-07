# PRD: Fix Schultz et al. (2017) citation errors and truncated sentence

**Area:** research / documentation  
**Category:** bugfix  
**Mode:** Standard  
**Created:** 2026-04-07  

---

## Context Loaded

Sources consulted:
- `https://arxiv.org/abs/1602.03501` — confirmed 4 authors: Patrick Schultz, David I. Spivak, Christina Vasilakopoulou, Ryan Wisnesky
- `https://kerodon.net/bibliography/MR131451` — confirmed Kan (1958) entry is correct as-is
- `https://arxiv.org/abs/1803.05316` — confirmed Fong & Spivak (2019) entry is correct as-is
- Full read of all 10 research docs: `bibliography.md`, `theory.md`, `core/algebra.md`, `core/activations.md`, `core/tensor.md`, `core/fixpoint.md`, `lattice/embed.md`, `lattice/attention.md`, `lattice/explorer.md`, `research/plan/index.md`
- Code cross-check of `lattice/embed.py` and `lattice/explorer.py` against their research note claims

---

## Problem Statement

When the Kan extension research was integrated into the research docs (2026-04-07), two errors were introduced:

**Error 1 — Missing author.** The Schultz et al. (2017) paper (arXiv:1602.03501) has four authors: Patrick Schultz, David I. Spivak, Christina Vasilakopoulou, and Ryan Wisnesky. Every citation in the research docs omits Vasilakopoulou, citing only three authors. This error appears in four files.

**Error 2 — Truncated sentence.** In `research/theory.md`, the sentence about what Schultz et al. prove ("all six standard relational algebra operations...") ends mid-sentence with "factor through the adjoint" and is cut off before its predicate.

**Non-issues confirmed by investigation:**
- Kan (1958) citation: correct in all locations (MR131451 confirmed).
- Fong & Spivak (2019) citation: correct — 2019 is the Cambridge UP book date; arxiv link 1803.05316 is accurate.
- `cite{key}` inline markers in research `.md` files: intentional project convention, validated by `tools/check_citations.py`. Not to be removed.
- Research notes on unimplemented features (`EmbR` not wired in, learning rule absent, `_concept_fixpoint` override redundancy, dead `learn` parameter): all verified accurate against current code. No phantom feature notes found.

---

## Audience

Researchers and contributors reading the research notes. The notes are the authoritative mathematical reference for the project; incorrect citations undermine their credibility.

## Verification

The fix is correct when:
1. Every occurrence of the Schultz et al. (2017) author string matches the 4-author list from arXiv:1602.03501
2. The truncated sentence in `theory.md` is grammatically complete and accurately describes the paper's content
3. `python tools/check_citations.py` exits 0 (cite keys still resolve)
4. No other factual errors are introduced

---

## Scope

### Files to modify

| File | Change |
|---|---|
| `research/bibliography.md` | (a) Narrative heading line 30: add Vasilakopoulou to bold-name lead. (b) Citation line 39: add Vasilakopoulou to author list. |
| `research/theory.md` | (a) Inline reference line 101: add Vasilakopoulou. (b) Lines 98–100: complete the truncated sentence. |
| `research/core/tensor.md` | Inline reference line 109: add Vasilakopoulou. |
| `research/lattice/explorer.md` | Inline reference line 180: add Vasilakopoulou. |

### Files to verify (read-only)

| File | What to verify |
|---|---|
| `tools/check_citations.py` | Still exits 0 after changes (no cite key renames) |
| `research/lattice/embed.md` | Schultz not cited here directly — confirmed no change needed |
| `research/core/algebra.md`, `core/activations.md`, `core/fixpoint.md`, `lattice/attention.md` | Schultz not cited — confirmed no change needed |

---

## Detailed Changes

### 1. `research/bibliography.md`

**Line 30 — narrative heading** (current):
```
**Schultz, Wisnesky & Spivak (2017)**
```
Change to:
```
**Schultz, Wisnesky, Vasilakopoulou & Spivak (2017)**
```

**Line 39 — citation entry** (current):
```
- Schultz, P., Wisnesky, R., & Spivak, D. I. (2017). Algebraic databases. ...
```
Change to:
```
- Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases. ...
```

### 2. `research/theory.md`

**Lines 98–102 — truncated sentence and inline reference** (current):
```
Schultz et al. (2017) show that all six standard relational algebra operations
(select, project, join, union, intersection, difference) factor through the adjoint

**Reference:** Schultz, P., Wisnesky, R., & Spivak, D. I. (2017). Algebraic databases.
```
Change to:
```
Schultz, Wisnesky, Vasilakopoulou & Spivak (2017) show that all standard relational
algebra operations factor through the adjoint triple (Σ ⊣ Δ ⊣ Π) — every query is a
composition of left Kan, restriction, and right Kan extensions.

**Reference:** Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases.
```

### 3. `research/core/tensor.md`

**Line 109 — inline reference** (current):
```
**Reference:** Schultz, P., Wisnesky, R., & Spivak, D. I. (2017). Algebraic databases.
```
Change to:
```
**Reference:** Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases.
```

### 4. `research/lattice/explorer.md`

**Line 180 — inline reference** (current):
```
**Reference:** Schultz, P., Wisnesky, R., & Spivak, D. I. (2017). Algebraic databases.
```
Change to:
```
**Reference:** Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases.
```

---

## Acceptance Criteria

- [x] `research/bibliography.md`: citation line includes Vasilakopoulou
- [x] `research/theory.md`: truncated sentence is complete and grammatically correct; inline reference includes Vasilakopoulou
- [x] `research/core/tensor.md`: inline reference includes Vasilakopoulou
- [x] `research/lattice/explorer.md`: inline reference includes Vasilakopoulou
- [x] `python tools/check_citations.py` exits 0 — verified: "ok: 5 citation(s) checked against 58 bibliography entries."
- [x] No other content changed

---

## Non-Goals

- Do not modify cite key names (`schultz2017` remains valid — key is by first author and year, not full author list)
- Do not alter the `<!-- [schultz2017] -->` key in bibliography.md
- Do not modify any Python source files
- Do not change any other citations (Kan, Fong & Spivak, Shen & Tang, etc.)
- Do not modify the existing spec for the integrate-kan-research task (already complete)
