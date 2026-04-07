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
that Join(A, B) ≤ C. This is the adjoint operation to a fuzzy relational join. 
Though not the same, this is similar to how division is related to multiplication.

In logical terms: Residuate asks "given what we know about A and the target C,
how large can B be without exceeding C?" It gives the tightest upper bound on B
consistent with A and C.

The operation is the α (alpha) operation from Sanchez (1976), applied row by
row. For each pair (y, z), it takes the min over all rows i of Implies(A[i,y],
C[i,z]) — the tightest constraint that every row of A places on B[y,z].

**Reference:** Sanchez, E. (1976). Theorem 5.

The adjoint relationship between Join and Residuate is a special case of the
quantale-enriched two-variable adjunction framework of Shen & Tang (2021).
Lambert's quantale V = ([0,1], min, 1) with Gödel implication satisfies their
Definition 3.3, so Proposition 5.3 applies: Join and Residuate arise as the
Kan adjunctions induced by the relation matrix φ: A^op ⊗ B → V. The concept
lattice Mφ = Fix(Residuate ∘ Join) is the complete V-category characterised by
their Theorem 6.2.

**Reference:** Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan
adjunctions via quantale-enriched two-variable adjunctions. *Applied
Categorical Structures*, 29, 823–858.

The Join ⊣ Residuate adjoint pair is a specific instance of a **Kan extension** — one
of the fundamental constructions in category theory. Given a functor (here: the relation
matrix φ: A^op ⊗ B → V), the right Kan extension along φ produces the greatest solution
consistent with φ, and the left Kan extension produces the least co-solution. In the
quantale-enriched setting, these are exactly Residuate (right Kan) and Join (left Kan)
— Shen & Tang (2021) use the name "Kan adjunctions" explicitly (Proposition 5.3) to
identify them as such.

The same construction appears in the **Categorical Query Language (CQL)** of Schultz,
Wisnesky et al. (2017). CQL models a database schema as a category and an instance as a
functor from schema to Set. Data migration along a functor F: S → T between schemas
decomposes into an adjoint triple:

```
Σ_F  ⊣  Δ_F  ⊣  Π_F
```

where Δ_F is restriction (pull data back along F), Σ_F is the left Kan extension
(push forward — existential: "there exists a row that maps to..."), and Π_F is the
right Kan extension (push forward — universal: "for all rows that map to..."). Every
well-typed query in CQL is one of these three operations, or a composition of them.

Lambert's operations are the same triple, restricted to the quantale V = ([0,1], min, 1):

| CQL operation | Lambert operation | Semantics |
|---|---|---|
| Δ_F (restriction) | slice R by active rows/columns | fix context, read known values |
| Σ_F (left Kan / existential) | Join(q, R) | forward inference: q reaches z if ∃y |
| Π_F (right Kan / universal) | Residuate(R, q) | backward inference: greatest B s.t. A∘B ≤ q |

The concept fixpoint — alternating Residuate(R, ·) and Residuate(R.T, ·) — is
restriction followed by the right Kan extension in each direction, settling to the
fixed point of their composition: the formal concept containing q.

**Reference:** Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases.
*Theory and Applications of Categories*, 32(16), 547–619.  cite{schultz2017}

**Reference:** Kan, D. M. (1958). Adjoint functors. *Transactions of the American
Mathematical Society*, 87(2), 294–329.  cite{kan1958}

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

## Tensor Logic

Domingos (2025) shows that Datalog rules and neural network operations are the same
mathematical object. A rule like `Ancestor(x,z) ← Parent(x,y), Parent(y,z)` is, in
the standard Boolean encoding, a tensor einsum with a Heaviside step function. In the
max-min semiring the step function disappears and the rule becomes exactly
`Join(Parent, Parent)`. Lambert's Join is therefore not an approximation of symbolic
reasoning — it *is* symbolic reasoning, expressed natively in the semiring. At T=0
it performs exact Datalog inference; at T>0 the smooth approximations introduce a
controlled softening that interpolates toward analogical reasoning. The system is
neurosymbolic by construction, not by composition.

**Reference:** Domingos, P. (2025). *Tensor Logic: The Language of AI.*

## Witness tracking

During Join, every intermediate node y that connects x to z above a threshold
is recorded in a witness store. This is not a separate interpretability
mechanism — it is a direct readout of the computation itself.

The witnesses are used by the provenance layer to reconstruct proof trees: given
a conclusion (x relates to z), the witnesses identify exactly which intermediate
entities justified it and how strongly. This makes the system's reasoning
auditable by construction.

The provenance package has been retired to `legacy/provenance/`. Lattice
traversal paths through the concept hierarchy are the forward path for
interpretability — the concept lattice provides a richer and more structured
account of inference than explicit witness-based proof trees.
