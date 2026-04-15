# Recursive Data Types and Interpreters

Declares the structure of recursive algebraic data and provides the generic
interpreter that drives algebra folds (catamorphisms) and coalgebra unfolds
(anamorphisms).

## Case — Endofunctor variant

`Case` in `engine/functor.py` is one variant of a recursive sum type. Each case
specifies `recursive` (children count), `data` (payload slots), and `output`
(0 or 1). The sum of all cases defines the endofunctor F.

In categorical terms, an endofunctor F: C → C maps objects and morphisms within a
single category. The `Case` sum type is the polynomial endofunctor decomposition:
F(X) = Σᵢ Aᵢ × X^(rᵢ), where Aᵢ is the data payload and rᵢ is the recursive arity.

**Reference:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}

## Functor — Collection of named Cases

`Functor` in `engine/functor.py` collects named `Case` objects into an endofunctor F.
Enforces unique case names and provides lookup by name.

## UnfoldStep — Coalgebra step output

`UnfoldStep` in `engine/functor.py` is the output of a single coalgebra step: case
name, payload, next states, and optional output. The `silent()` class method creates
steps that transition state without emitting — the internal moves of the coalgebra.

## Interpreter — Generic (co)algebra driver

`Interpreter` in `engine/functor.py` drives algebra folds (`run_algebra`) and
coalgebra unfolds (`run_coalgebra`) over declared Functors.

### run_algebra — Catamorphism (initial algebra)

Given a tree and a decompose function, `run_algebra` recursively folds the tree
bottom-up. Each node is decomposed into (case_name, payload, children), children
are folded first, then the cell function produces the result. This is the
catamorphism — evaluation of the initial F-algebra.

The Tarski (1955) fixpoint theorem guarantees that the fold terminates: the tree
is finite (no cycles), so the recursion bottoms out at leaf cases (recursive=0).

### run_coalgebra — Anamorphism (final coalgebra)

Given an initial state and optional token iterator, `run_coalgebra` unfolds a
stream by repeatedly applying the coalgebra cell. The cell returns an `UnfoldStep`
specifying the case, payload, next state, and optional output. The unfold continues
until the token iterator is exhausted or a stop predicate fires.

The coalgebra runner supports the linear single-successor subset: cases with
`recursive=1`. This covers autoregressive streaming (GPT-style generation) where
each step produces one output and one next state.

Accumulation (`accumulate_legs`) supports KV-cache-style state growth: payload
items are concatenated across steps, building the cache incrementally.

**References:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}
- Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its applications.
  *Pacific Journal of Mathematics*, 5(2), 285–309.  cite{tarski1955}
