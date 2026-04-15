# Runtime Closure Factories

Core runtime functions that compile, compose, and validate morphisms at execution
time.

## MorphismSpec — Compiled morphism declaration

`MorphismSpec` in `engine/runtime.py` is the compiled form of a `MorphismDecl`. It
carries the resolved op callable, compiled equation, sort annotations, transform
function, and arity. This is the bridge between the declaration layer and runtime
execution.

In Lawvere's (1973) framework, a `MorphismSpec` is a V-functor: a morphism between
V-enriched categories that preserves the enrichment structure. The `src_sort` and
`tgt_sort` fields encode the domain and codomain; the `op` is the action on hom-objects.

**Reference:**
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

## chain — V-functor composition

`chain()` in `engine/runtime.py` composes callables sequentially: each output feeds
the next as input. This is V-functor composition gf: X → Z — the fundamental
operation of enriched category theory.

**Reference:**
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

## chain_with_augments — Right Kan extension

`chain_with_augments()` in `engine/runtime.py` chains steps where 'augment' steps
merge into the relational context y instead of transforming the primary signal x.
Augment steps act as right Kan extensions: they enrich the context without modifying
the query.

This is the concrete mechanism that eliminated the last algebra-side Python escape
hatch in the attention case: the `[fan_name]` syntax in a path runs a fan and merges
its dict output into y.

**Reference:**
- Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via
  quantale-enriched two-variable adjunctions. *Applied Categorical Structures*,
  29, 823–858.  cite{shen2021}

## fan — V-category product

`fan()` in `engine/runtime.py` runs all branches on the same input and merges
results. This implements the V-category product X × Y from Lawvere (1973).

**Reference:**
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

## check_sorts — Composition axiom validation

`check_sorts()` in `engine/runtime.py` validates that consecutive morphisms in a
path have matching sorts: `tgt_sort(f) == src_sort(g)`. Uses Hydra Type equality
for sorts with declared structure, falls back to string equality otherwise.

This enforces the composition law of V-categories: a composition gf is well-defined
only when the codomain of f equals the domain of g.

**Reference:**
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
