# Architecture Definitions

Compiled architecture definitions and their runtime interpreters. The top-level
compiled artifact of the DSL pipeline.

## ArchDef — Compiled DSL output

`ArchDef` in `engine/arch.py` is the compiled result of a full DSL source block.
It holds every compiled path, morphism, fan, and arch declaration. Provides:

- `explain(name)` — human-readable path description
- `trace(name, x, y, temp)` — step-by-step execution with shapes
- `loss(name, x, y, temp)` — observer loss computation
- `interpreter(name, params, temp)` — create an `ArchInterpreter`

The `module` property assembles a Hydra Module; the `graph` property builds a Hydra
Graph with engine primitives. These connect the DSL to the Hydra type/term system
for formal verification.

**References:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}
- Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
  PODS 2007, pp. 31–40.  cite{green2007}

## ArchInterpreter — Algebra/coalgebra runtime

`ArchInterpreter` in `engine/arch.py` wraps a single arch declaration's algebra
and coalgebra interpreters.

### run_algebra — Forward pass (catamorphism)

`run_algebra(tree, decompose)` folds a pre-built tree using the algebra `Interpreter`
from `engine/functor.py`. Each node is dispatched by case name to its bound cell.

### run_algebra_layers — Iterate combinator

`run_algebra_layers(x0, layers, extras)` builds a tree from iterate groups via
`_build_tree()`, then folds it. This is the primary API for architectures with
repeated structure (e.g. transformer layers). The tree nesting reverses declaration
order: declared `attn, ffn` nests as `ffn(attn(prev))`.

### run_coalgebra — Streaming inference (anamorphism)

`run_coalgebra(state, token_iter, stop)` drives the coalgebra unfold. Convergence
is checked via `_make_convergence_stop()`, which measures the max-abs residual
between consecutive states — the energy descent criterion.

The convergence mechanism is grounded in Ramsauer et al. (2021): attention as
Hopfield energy minimization, where streaming convergence corresponds to energy
descent to a fixed point.

**Reference:**
- Ramsauer, H. et al. (2021). Hopfield networks is all you need.
  ICLR 2021.  cite{ramsauer2021}

## Observer paths

An `ArchDecl` can declare two observer paths:

- `observer_convergence` — tests whether the system has reached a fixed point
  (residual < threshold). This is the closure test from fixpoint theory.
- `observer_loss` — provides the training signal (degree of non-closure) for
  gradient descent.

These are compiled from `PathDecl` entries by `engine/compiler.py` and resolved
into callables by `_resolve_observers()`.

## Algebra/coalgebra duality

The central insight from Gavranović et al. (2024) is that every architecture is a
pair (initial algebra, final coalgebra) over a shared endofunctor F. The `ArchDecl`
makes this duality declarative: the same `cases` list serves both the algebra
(tree fold) and coalgebra (stream unfold) sides.

The Σ⊣Δ⊣Π adjoint triple from Schultz & Wisnesky (2025) maps directly onto this
structure: the algebra side is Σ (left Kan, forward/generative), the coalgebra side
is Π (right Kan, universal/restrictive), and the shared endofunctor is Δ.

**References:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}
- Schultz, P., & Wisnesky, R. (2025). Algebraic data integration.
  *Journal of Functional Programming*, 27.  cite{schultz2025}
