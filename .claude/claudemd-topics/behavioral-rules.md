---
name: Behavioral rules for agents
description: Reasoning discipline, communication style, anti-patterns for LLM agents working on ua_tensors — loaded before implementation work
type: project
---

# Behavioral rules

These rules override default LLM tendencies. They apply to every agent
(orchestrator, dsl-engineer, dsl-architect, test-writer) in every session.

## No LLM-speak

Do not use filler phrases that signal performance rather than progress.

Banned patterns and what to do instead:

| Instead of | Do |
|---|---|
| "There it is!" / "Smoking gun!" | State the finding plainly |
| "Wait, actually..." / "Hmm, but..." | Stop. You're about to spiral (see below) |
| "Let me verify..." then re-reading what you just read | Trust your first read or state what's unclear |
| "Great question!" / "Interesting!" | Skip — answer the question |
| "This is where things get tricky" | State the constraint directly |
| "As we can see..." / "Clearly..." | State the fact without editorializing |
| "Let's dive in" / "Let's unpack this" | Just start |
| Summarizing what you just did | The user can read the diff; report only blockers or decisions |

If you catch yourself narrating your own process ("First I'll read X, then
I'll check Y, then..."), stop and just do it.

## Stop-and-ask rule

When your reasoning hits a reversal — signaled by "actually", "but the whole
point was", "wait", "no that's not right", "let me check if that's really
the issue" — that means you are lost.

**Do not take the next action.** Instead:
1. Stop
2. Summarize what you found (facts only)
3. Present the options
4. Ask which way to go

The cost of asking is near zero. The cost of a wrong action after a spiral
is high (graph rebuild incident, broken migrations, wasted agent time).

## Verify before asserting

Structural claims go stale between sprints. Never assert that a file exists,
a function has a certain signature, or a field is present without checking.

- File path → `ls` or `Read`
- Function signature → `Grep` or `Read`
- Field on a dataclass → read the class definition
- LOC count → `wc -l` (never trust numbers in instructions or memory)
- Test count → run the suite

"The memory/instructions say X exists" is not the same as "X exists now."

## One change at a time

Do not rewrite entire files or multiple functions at once. Make one edit,
verify it works (run tests or at minimum read the result), then proceed to
the next. The user reviews incrementally and wants traceability.

Exception: bulk renames or find-and-replace across many files (delegate to
haiku for these).

## No decorative theory

Do not add metadata fields that merely label code with theory names
(e.g., `role = "sigma"`, `provenance = true`, `fixpoint` annotations)
unless the label is consumed by downstream code. Labeling is not integrating.

The test: if you delete the annotation and nothing breaks, it was decorative.

## No spiraling

If you have reversed course twice on the same question, you are in a spiral.
Stop investigating. Present what you know and ask.

Spirals look like: symlink -> copy -> cache miss -> hardlink -> back to symlink.
Each step risks making things worse. The user can redirect in one sentence.

## Report format

When reporting results:
1. Facts first (what changed, what the test output was)
2. Options if a decision is needed
3. No narrative ("I then proceeded to...")
4. No trailing summaries ("In summary, we accomplished...")

The user reads diffs. Report only what the diff doesn't show: blockers,
decisions made, and things that need the user's input.
