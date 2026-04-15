# Declaration Layer

AST node declarations for the engine DSL. Pure data structures representing the
syntactic constructs of the source language, consumed by the compiler.

## SemiringDecl — Quantale V = (V, ⊗, k)

`SemiringDecl` in `engine/decl.py` declares the algebraic structure over which
morphisms compose. It carries a `contract` (the tensor operation implementing ⊗),
an optional `compiler` (equation string → compiled form), and an `arity`.

Each `SemiringDecl` selects a row from Lawvere's (1973) table: metric spaces,
posets, categories, and V-categories are all instances of quantale-enriched
structure. The DSL is parametric over this choice — the same architecture
declarations compile against any declared semiring.

Green et al. (2007) show that relational queries are parameterized by semiring:
the same query evaluated over different semirings yields tuple identity (boolean),
count (natural numbers), confidence (tropical), or probability. `SemiringDecl`
makes this semiring-parametricity a first-class feature.

**References:**
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}
- Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
  PODS 2007, pp. 31–40.  cite{green2007}

## MorphismDecl — V-functor f: src_sort → tgt_sort

`MorphismDecl` in `engine/decl.py` declares a typed morphism: a V-functor
between sorts, computed via an einsum equation string. Each morphism carries
`src_sort` and `tgt_sort` annotations that the compiler checks for composition
validity — the V-functor composition axiom X(x,y) ≤ Y(fx,fy).

The `template_param` field supports parametric morphisms — natural transformations
indexed by a parameter (e.g. `ln[prefix]`). This is the DSL's concrete representation
of the parametric morphisms in §3 of Gavranović et al. (2024).

Domingos (2025) shows that neural network layers and Datalog rules are both einsum
operations. The `equation` field on `MorphismDecl` is this insight made declarative:
every morphism is an einsum, and the semiring determines whether it performs symbolic
(T=0) or subsymbolic (T>0) inference.

**References:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}
- Domingos, P. (2025). Tensor logic: The language of AI.
  arXiv:2510.12269.  cite{domingos2025}
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

## PathDecl — Sequential V-functor composition

`PathDecl` in `engine/decl.py` declares a named chain of morphisms. The compiler
translates it to `chain()` in `engine/runtime.py`, validating sort adjacency at
each step — the composition axiom tgt_sort(f) == src_sort(g).

Paths with `residual = True` implement the universal map from a coproduct: f(x) + x.
The `normed` field applies a normalization morphism after the residual, following the
pre-norm transformer pattern.

In the Σ⊣Δ⊣Π framework of Schultz et al. (2017), a path is a composition of data
migration functors. Join-based paths are Σ (left Kan, existential); Residuate-based
paths are Π (right Kan, universal).

**References:**
- Schultz, P. et al. (2017). Algebraic databases. *Theory and Applications of
  Categories*, 32(16), 547–619.  cite{schultz2017}
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

## FanDecl — V-category product

`FanDecl` in `engine/decl.py` declares parallel fan-out with a merge strategy.
Branches run simultaneously on the same input; results are collected via the merge
(dict, meet, join, or a custom callable). This implements the V-category product
X × Y from Lawvere (1973).

**Reference:**
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories.
  *Rendiconti del Seminario Matematico e Fisico di Milano*, XLIII, 135–166.  cite{lawvere1973}

## CaseDecl — Endofunctor variant

`CaseDecl` in `engine/decl.py` declares one case of a recursive endofunctor F.
Each case specifies `recursive` (number of recursive children), `data` (payload
slots), and `output` (whether the case emits output).

The `iterate` field designates a payload sequence for catamorphism iteration —
`_detect_iterate_groups()` in `engine/compiler.py` uses this to build the algebra
tree. This is the initial algebra evaluation of Gavranović et al. (2024) §5.

The `morphisms` field allows the algebra cell to be derived automatically from a
path composition, eliminating Python escape hatches.

**Reference:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}

## ArchDecl — Algebra/coalgebra duality

`ArchDecl` in `engine/decl.py` declares an architecture as one endofunctor F
with dual evaluation: an algebra side (catamorphism, forward pass) and a coalgebra
side (anamorphism, streaming/KV-cache).

The `cases` list declares the shared endofunctor. The `state:` and `step:` blocks
provide coalgebra-specific configuration: `step_enter` binds input tokens,
`step_emit` produces output. The algebra and coalgebra share the same morphism
compositions — the duality is structural, not duplicated.

This is the central construction of Gavranović et al. (2024): every neural
architecture is a pair (initial algebra, final coalgebra) over a shared endofunctor.

The adjoint data migration functors Σ_F, Δ_F, Π_F from Schultz & Wisnesky (2025)
map directly: the algebra side is the Σ direction (forward/generative), the coalgebra
side is the Π direction (universal/restrictive), and the shared endofunctor is Δ.

**References:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}
- Schultz, P., & Wisnesky, R. (2025). Algebraic data integration. *Journal of
  Functional Programming*, 27.  cite{schultz2025}
