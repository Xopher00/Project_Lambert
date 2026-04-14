# Legacy Implementation

This directory contains the original hand-coded implementation of the Lambert system, preserved as reference for the mathematical development path and architectural decisions.

## Core Components

**lattice/** — Boolean matrix factorization and concept lattice operations:
- `coder.py` — PathCoder encode/decode skeleton for Attend and Recall operations
- `embed.py` — Concept embedding via Boolean matrix factorization
- `attention.py` — Single and multi-head retrieval over concept embeddings
- `explorer.py` — Category exploration and lattice closure operations

**model.py** — Lambert orchestrator class tying the pipeline together

**query.py** — Query interface for entity and feature retrieval

## Earlier Iterations

- `language.py`, `lattice.py`, `train.py` — Earlier prototypes predating the modular structure
- `provenance/` — Audit and provenance tracking (deprecated)

## Relationship to engine/ DSL

These implementations are now superseded by `engine/`, which expresses the same architectures using arbitrary semirings and category-theoretic combinators. The legacy code demonstrates how the mathematical concepts evolved from concrete Boolean algebra to abstract algebraic structures, and serves as executable reference for validating DSL semantics.
