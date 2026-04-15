# fixpoint

The fixpoint iteration infrastructure. Wraps any operator and iterates it until
the state stops changing, with temperature derived from energy at each step.

This module is the theoretical heart of the system. The pattern it implements —
measure free energy, derive temperature from it, iterate until convergence — is
what connects Lambert's computation to the free energy principle in predictive
coding. Every layer above this one delegates to it: Closure, Attention,
ConceptEmbed, and Lambert itself all run inside a FixpointIterator. The number
of reasoning steps the system takes, and the sharpness of every smooth operator
during those steps, is determined entirely by the energy dynamics described here.

## Fixpoint iteration and the Tarski guarantee

A fixpoint of an operator f is a state x such that f(x) = x — the operator
applied to the state returns the state unchanged. For Lambert's operators (Join,
Attend, _concept_fixpoint), fixpoints correspond to stable beliefs: states where
further reasoning produces no new information.

Convergence of fixpoint iteration is guaranteed by the Knaster-Tarski theorem
for monotone operators on complete lattices. The FixpointIterator handles the
bookkeeping — measuring energy (how much the state changed) and stopping when it
falls below a threshold — but it is the algebraic structure of the operator that
guarantees a fixpoint exists and will be reached.

**Reference:** Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its
applications. *Pacific Journal of Mathematics*, 5(2), 285–309.

## The energy function

At each step, FixpointIterator measures how much the state changed using an
energy function with two terms:

    dynamic error  =  Σ |new - old|²
    sensory error  =  Σ |raw - corrected|²
    energy         =  dynamic error + sensory error

**Dynamic error** measures how far the state moved from the previous step. When
this is zero, the system has reached a fixpoint.

**Sensory error** measures how much the raw prediction (the operator's output
before any correction) differed from the corrected belief (the state after
applying a Residuate or similar constraint). It is only present when the operator
returns an auxiliary value alongside the new state.

The energy function is, mathematically, a loss function — the same squared-error
sums used in standard supervised learning. What differs is the interpretation:
rather than measuring distance from a labelled target, it measures distance from
a fixpoint. The system is not trained toward an answer; it iterates toward
internal consistency.

This two-term structure instantiates the same theoretical pattern as the
prediction error decomposition in active inference (Parr, Pezzulo & Friston
2022, equation 4.19): free energy as a sum of precision-weighted prediction
errors across sensory and dynamic levels. Lambert's sensory error mirrors the
sensory prediction error ε_y; the dynamic error mirrors the dynamic prediction
error ε_x. The correspondence is structural — both frameworks use a multi-term
residual to drive inference toward a consistent belief state.

A third term — parametric error, measuring divergence between current and prior
parameters — appears at the third level of equation 4.19 but is not included
here. It may become relevant when parameter learning is added.

**Reference:** Parr, T., Pezzulo, G. & Friston, K.J. (2022). *Active Inference:
The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
DOI: 10.7551/mitpress/12441.001.0001.

## Temperature annealing

Temperature is derived from energy at each step using a formula inspired by the
Boltzmann distribution in statistical mechanics:

    T = |−E / (N × mean(log(state)))|

where E is the current energy and N is the state size.

This is the computational interpretation of precision in predictive coding: far
from a consistent belief state, the system operates with high temperature — soft,
exploratory, analogical. As beliefs stabilise, energy falls, and the system
sharpens toward crisp outputs. No external schedule is needed.

The specific formula is a heuristic borrowed by analogy from thermodynamics
(F = U − TS). It produces qualitatively correct behaviour but uses a non-standard
entropy definition and has known boundary cases when state values are zero or all
one. A formula with a cleaner theoretical derivation may exist.

Domingos (2025) discusses temperature in a related context: T=0 is purely
deductive; increasing T makes reasoning increasingly analogical, with less similar
examples borrowing inferences from each other. The optimal T differs per
application — mathematical rules warrant T=0, rules accumulating weak evidence
over sparse data warrant higher T. Lambert implements this automatically: sparse
or contradictory inputs produce high energy and therefore high temperature;
well-supported inputs converge quickly at low temperature.

**Reference:** Domingos, P. (2025). *Tensor Logic: The Language of AI.*
arXiv:2510.12269v3.

## Fixpoints as algebraic closure

There is a deeper algebraic reading of the fixpoint iteration. The pattern —
a system that can formulate questions it cannot answer, and must be extended
until it can — is the same structure that appears in algebraic closure of fields.

In ℝ, the polynomial x² + 1 = 0 is *expressible* within ℝ but has *no solution*
in ℝ. The algebraic closure ℂ is the minimal extension where every expressible
polynomial has a root. In Lambert, a relational query composed from the available
morphisms (Join, Residuate, etc.) is *expressible* but may point to a concept
that doesn't exist in the current lattice. The nonzero fixpoint residual is the
witness — exactly analogous to evaluating x² + 1 over all of ℝ and finding it
never vanishes.

The general structure is:

1. **A domain** — ℝ, a concept lattice, a set under operations
2. **A language of expressions** — polynomials, relational path compositions, operation sequences
3. **Closure** = every expression in the language that should have a solution, does have one in the domain

A system is closed when every question expressible in its language has an answer
in its domain. Algebraic closure for fields, operational closure for sets,
fixpoint closure for Lambert — all instances of the same condition.

### Idempotency is the key

The connection to semiring choice is precise. In the smooth max-min semiring
(temp > 0), the operations are *not* idempotent: applying softmax or softmin
twice gives a different result than applying it once. But at temp = 0, the
operations collapse to exact max and min, which *are* idempotent: max(x, x) = x,
min(x, x) = x. Idempotency means that applying the operation to its own output
is a no-op — the output is already a fixpoint.

This is the mechanism by which temperature controls closure:

- **temp = 0**: idempotent operations, fixpoints exist trivially (every output
  is already a fixpoint of the operation that produced it), reasoning is exact
  but discrete — the lattice either contains the answer or it doesn't
- **temp > 0**: non-idempotent operations, fixpoints must be *found* by
  iteration, the residual is a continuous signal pointing toward the missing
  concept — gradient-based learning is possible

The smoothing doesn't change *what* closure means; it changes *how* the system
searches for it. At zero temperature, closure is a yes/no structural property.
At positive temperature, the degree of non-closure becomes a differentiable loss,
and learning is the process of extending the lattice until closure is achieved.

### Learning as algebraic closure

This reframes Lambert's learning algorithm: **learning is computing the algebraic
closure of the concept lattice with respect to the observed data.** Each training
step measures the residual (the degree of non-closure under the current queries),
and grows or reshapes the lattice in the direction the residual points, until
every query the data can express has a concept that satisfies it.

The observer block in the engine DSL implements exactly this: the convergence
path tests whether the lattice is closed (residual < threshold), and the loss
path provides the training signal (degree of non-closure) for gradient descent.
Incremental concept lattice construction — growing the lattice one concept at a
time in the direction of maximal residual reduction — is the concrete algorithm
that performs this closure.

## Engine DSL representation

Convergence is managed through `observer_convergence` on `ArchDecl` in
`engine/decl.py`. The compiler resolves this to a path callable, and
`_make_convergence_stop()` in `engine/arch.py` wraps it into a halt predicate for
the coalgebra runner. The `ArchInterpreter.run_coalgebra()` method in
`engine/arch.py` uses energy-driven convergence to decide when streaming inference
has stabilized.

The iterate combinator (`iterate = layers` on `CaseDecl`) provides the catamorphism
side. Dannert et al. (2021) prove that provenance is preserved through fixpoint
iterations over absorptive semirings — the engine's `SemiringDecl` abstraction
ensures this guarantee holds.

See [engine/arch](../engine/arch.md) and [engine/compiler](../engine/compiler.md)
for full DSL documentation.

