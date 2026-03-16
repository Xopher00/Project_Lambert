# fixpoint

The fixpoint iteration infrastructure. Wraps any operator and iterates it until
the state stops changing, with temperature derived from energy at each step.

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

This two-term structure is inspired by the prediction error decomposition in
active inference (Parr, Pezzulo & Friston 2022, equation 4.19), where free
energy is a sum of precision-weighted prediction errors across sensory and
dynamic levels. The structural correspondence is real: Lambert's sensory error
mirrors the sensory prediction error ε_y, and the dynamic error mirrors the
dynamic prediction error ε_x.

The limit of the analogy: the full variational free energy framework requires a
joint generative model, an approximate posterior, and a KL divergence applied to
a probability measure. Lambert has none of these — the max-min semiring is not a
probability measure, and Join is not marginalisation. Lambert's energy is best
understood as a fixpoint residual in the max-min semiring, not as an evidence
lower bound.

A third term — parametric error, measuring KL divergence between current and
prior parameters — appears at the third level of equation 4.19 but is not
included here. It may become relevant when parameter learning is added.

**Reference:** Parr, T., Pezzulo, G. & Friston, K.J. (2022). *Active Inference:
The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
DOI: 10.7551/mitpress/12441.001.0001.

## Temperature annealing

Temperature is derived from energy at each step using a formula inspired by the
Boltzmann distribution in statistical mechanics:

    T = |−E / (N × mean(log(state)))|

where E is the current energy and N is the state size. When energy is high,
temperature stays high — keeping the smooth activations (SmoothMax, SmoothMin)
exploratory and fuzzy. As energy falls toward zero, temperature falls too and the
system converges toward crisp boolean outputs.

This couples the reasoning mode to convergence progress automatically: the system
is soft and exploratory when far from a fixpoint, and hard and decisive once it
has nearly converged. No external schedule is needed.

The formula is borrowed by analogy from thermodynamics (F = U − TS), where
temperature governs the trade-off between energy minimisation and entropy. It
produces qualitatively correct behaviour but uses a non-standard entropy
definition and has boundary cases when any state value is zero or all values
are one.

## The perturb interface

FixpointIterator exposes a `perturb` method for incremental updates. Rather than
constructing a new iterator from scratch when new data arrives, the existing
iterator is reset to a new starting state — clearing accumulated iteration state
and temperature — and run to convergence again, preserving the operator and
convergence parameters from construction.
