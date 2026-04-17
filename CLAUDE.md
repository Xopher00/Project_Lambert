## What this project is

ua_tensors is one claim: **every AI architecture is a V-enriched categorical
composition. Change V, get a different computation. Change the evaluation mode,
get batch or streaming.** The engine DSL in `engine/` is how a researcher writes
down any architecture in declarative form. Hydra is the DSL authoring framework
we build on.

Full framing and reference: `research/dsl-roadmap.md`. Read it when starting a
sprint. The v1.5 revision (2026-04-17) is the latest.

## The two axes (read before planning any sprint)

The project is complete along one axis and partial along another. Confusing
them is the root cause of the bloat-then-regression cycle prior sprints hit.

**Axis A — DSL as Python compiler.** `.ua` text compiles to Python callables;
runs any AI architecture; supports algebra + coalgebra. ~97% done. T1-T4 /
E1-E5 complete. Remaining work: polish (P2 parser, P5 coalgebra branching) or
deferred features. Sprints that target axis A succeed by deleting code while
tests stay green.

**Axis B — DSL as Hydra DSL.** Engine AST = Hydra AST; engine terms evaluable
by Hydra reduction; types checkable by Hydra; archs rewritable. Read roadmap
section 3.4 for current step status — do not hardcode it here.

Every sprint is one of the two. Name which before planning.

## Axis-B sprint rules (the unification thesis)

Axis B is where real unfinished work lives. It does not fit the LOC-reduction
pattern. The staged plan is in `research/dsl-roadmap.md` section 3.4 — read
it for current step status. Do not hardcode step status here; it goes stale.

Some steps are additive (LOC grows). Some are deletions. An axis-B sprint
that adds code on an additive step is correct. The LOC-negative rule is
axis-A only.

**Anti-patterns** (named in roadmap 3.4; repeated here because prior sprints
fell into them):

- **Mechanical swap trap.** Replacing a hand-rolled helper with a Hydra module
  is only a win if the DSL's AST is already Hydra's AST. Otherwise it wraps
  inert metadata.
- **Sidecar trap.** Adding more Hydra imports to validate / audit metadata
  terms. Makes imports go up; makes nothing load-bearing.
- **LOC-only thinking.** The spine is parser + compiler's dataclass walk.
  Trimming leaf helpers hits diminishing returns fast.
- **Wrapping without deleting.** Rule from v1.3 stands, but **only applies to
  step 6**. Steps 1, 3, 4 are additive; step 5 is where wrapping-without-
  deleting is disallowed.

## Axis-A sprint rules

Traditional simplification. Delete code; keep tests green. LOC-negative target.

**Rule.** Every axis-A change is net-negative or net-zero LOC. If you're about
to add code, first identify what you're removing to make room.

**LOC budgets** (measure live with `wc -l engine/*.py`; never trust numbers
in this file or the roadmap — they go stale):

| Component | Target |
|-----------|--------|
| engine/ source | ≤2,500 (post-unification) |
| tests/engine/ | ≤1,500 |

The ≤2,500 engine target assumes axis-B unification has landed. Pre-
unification, the engine will be larger because both AST representations live.
This is expected and is not a failure.

## Stop conditions

Simplification ends when:
- Axis A: engine/ and tests/engine/ at or below budget AND tests green.
- Axis B: roadmap 3.4 steps 1-6 landed; compiler consumes Hydra `Term.Record`
  instead of Python dataclasses; `terms.py` collapsed; `check_sorts` replaced
  with `hydra.checking`.

At that point, say so. Don't invent new work.

## Behavioral rules

These override default LLM tendencies. Every agent, every session.

- **No LLM-speak.** No "there it is", "smoking gun", "let's dive in", "as we
  can see", trailing summaries, or narrating your own process. State facts.
- **Stop on reversal.** When you catch yourself saying "actually", "wait",
  "but the whole point was" — you are lost. Stop, summarize findings, ask.
- **Verify before asserting.** File exists? Read it. Function signature?
  Grep it. LOC count? `wc -l`. Never trust instructions or memory over code.
- **One change at a time.** Edit, verify, then next. No whole-file rewrites.
- **No decorative theory.** Labels that nothing consumes are noise, not
  integration. If deleting the annotation breaks nothing, it was decorative.

Full rules with examples: `.claude/claudemd-topics/behavioral-rules.md`.

## Session protocol

At session start:
1. Read this file.
2. `git status`, `git log --oneline -5` for branch state.
3. Measure live: `wc -l engine/*.py tests/engine/*.py`.
4. Identify which axis the requested sprint targets. If unclear, ask.

At session end / "wrap up":
1. Update relevant memory files with current state.
2. Update this file ONLY if a new lesson emerged that a future session needs.
   Do not update with stale numbers. Do not re-duplicate roadmap content.
3. If code structure changed, check `.claude/agents/*.md` and
   `.claude/claudemd-topics/*.md` for stale references. Structural
   descriptions go stale silently — "measure live" applies to them too.
4. Do NOT commit unless asked.

## Hydra reference

The authoritative account of what Hydra can do for this DSL is `research/dsl-roadmap.md`
section 3.4 (the unification thesis and six-step plan) and section 8.7
(capability map). Do not duplicate that content here; pointer only.

Hydra exploration path:
- `hydra/CLAUDE.md` — Hydra project overview
- `hydra/docs/dsl-guide-python.md` — Python DSL guide (types, terms, phantoms)
- `hydra/docs/hydra-lexicon.txt` — ~180 primitive signatures
- `hydra/dist/python/hydra-kernel/src/main/python/hydra/` — generated Python kernel

## Python version

Python 3.12+. Tests: `uv run --python 3.12 pytest`. The `.venv` uses 3.12.
Do not use system `python3` (3.11).

## graphify

Knowledge graph at `graphify-out/`. MCP server connected. Use before reading
raw files for high-connectivity code.

- `mcp__graphify__god_nodes` — most-connected nodes
- `mcp__graphify__get_neighbors "Node"` — blast radius before deleting
- `mcp__graphify__query_graph "term"` — structural search
- `mcp__graphify__shortest_path "A" "B"` — trace coupling

Skip the graph for: test consolidation, small-scope changes, naming decisions.

## Topic Files

Read on demand — do not load preemptively.

- `.claude/claudemd-topics/behavioral-rules.md` — before starting implementation or when behavioral drift is noticed
- `.claude/claudemd-topics/agent-dispatch.md` — before delegating to sub-agents or dispatching a sprint
