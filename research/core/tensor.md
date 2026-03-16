# tensor

The relational operations layer. Implements three core operations on matrices
under the max-min semiring: Join, Residuate, and Closure.

## The max-min semiring

Lambert reasons over relations using the max-min semiring — the algebraic
structure ([0,1], max, min). In this semiring:

- **max (∨)** is the combining operation — analogous to addition in standard
  linear algebra
- **min (∧)** is the composing operation — analogous to multiplication

This choice makes every operation semantically transparent: values are
membership grades, max selects the strongest connection, and min finds the
weakest link in a chain.

The underlying structure is a distributive lattice with over 50 years of
theoretical grounding in fuzzy set theory and relational algebra.

**Reference:** Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3),
338–353.

## Join: relational composition

    Join(A, B)[x,z] = max over y of min(A[x,y], B[y,z])

For each output cell (x, z), Join finds the best intermediate node y by taking
the min of A[x,y] and B[y,z] — the weakest link in the chain x→y→z — then
takes the max over all y — the strongest such chain. This is Zadeh's max-min
relational composition.

In logical terms: x relates to z if there exists a y such that x relates to y
and y relates to z. The strength of the connection is determined by the weakest
link in the best available chain.

Only non-zero entries are visited for efficiency. At T=0 the operation is exact.
At T>0 SmoothMax and SmoothMin are used, introducing a controlled approximation
error bounded by the Nesterov gap.

**Reference:** Sanchez, E. (1976). Resolution of composite fuzzy relation
equations. *Information and Control*, 30, 38–48.

## Residuate: the adjoint of Join

    Residuate(A, C)[y,z] = min over i of Implies(A[i,y], C[i,z])

Given a left relation A and a target C, Residuate finds the greatest B such
that Join(A, B) ≤ C. This is the right residual of the max-min semiring — the
exact algebraic inverse of Join within the semiring structure.

In logical terms: Residuate asks "given what we know about A and the target C,
how large can B be without exceeding C?" It gives the tightest upper bound on B
consistent with A and C.

The operation is the α (alpha) operation from Sanchez (1976), applied row by
row. For each pair (y, z), it takes the min over all rows i of Implies(A[i,y],
C[i,z]) — the tightest constraint that every row of A places on B[y,z].

This is not a heuristic approximation — it is the exact Galois adjoint of Join
in the max-min semiring.

**Reference:** Sanchez, E. (1976). Theorem 5.

## Closure: transitive reachability

Closure iterates Join to fixpoint, computing all transitive connections implied
by a base relation E.

**Convergence** is guaranteed by the Knaster-Tarski fixpoint theorem: Join
iteration (R ← Join(R, E)) is monotone increasing in R on the complete lattice
([0,1]^{n×n}, ≤), so a fixpoint must exist and the iteration will reach it.
The FixpointIterator handles the bookkeeping — detecting when successive
iterates stop changing — but convergence is guaranteed by the algebraic
structure, not by the iterator.

**The Residuate correction** serves a separate purpose: at each step, the grown
relation is clipped back using Residuate to enforce the invariant R ∘ E ≤ E.
This determines *which* fixpoint is reached — constraining the closure to remain
grounded in E — rather than whether a fixpoint is reached. Without the
correction, Join converges to the least fixpoint above the initial R,
unconstrained. With it, the result is a fixpoint that is both closed under
E-composition and consistent with E as an upper bound.

The combined corrected iteration (Join followed by Residuate at each step) does
not have a clean Tarski guarantee — Residuate is monotone decreasing in its
first argument, so the composed operator is not obviously monotone. Convergence
of the corrected iteration is empirically observed but not covered by the
theorem as stated.

**References:**
- Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its
  applications. *Pacific Journal of Mathematics*, 5(2), 285–309.
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
  *Information and Control*, 30, 38–48.

## Witness tracking

During Join, every intermediate node y that connects x to z above a threshold
is recorded in a witness store. This is not a separate interpretability
mechanism — it is a direct readout of the computation itself.

The witnesses are used by the provenance layer to reconstruct proof trees: given
a conclusion (x relates to z), the witnesses identify exactly which intermediate
entities justified it and how strongly. This makes the system's reasoning
auditable by construction.

Witness tracking is currently under review. As the concept lattice layer
matures, lattice traversal paths may supersede explicit witness-based proof
trees.
