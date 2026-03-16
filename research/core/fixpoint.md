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

