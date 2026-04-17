# activations

The smooth activation layer. Every function in this module is built on LogSumExp.

## From UA operators to smooth functions

The two fundamental operators in Unified Algebra are:

    x ∧ y  =  min(x, y)
    x ∨ y  =  max(x, y)

Both have sharp corners — non-differentiable kinks — that make them unsuitable for
gradient-based learning. The activation functions in this module smooth them using
a single primitive: LogSumExp.

Every function here is a special case or extension of LogSumExp:

- **SmoothMax** — LogSumExp directly, smooth ∨
- **SmoothMin** — derived from SmoothMax by De Morgan duality, smooth ∧
- **Softplus** — LogSumExp over (0, x), smooth ∨ with zero as lower bound
- **Relu** — Softplus at T=0, exact max(x, 0)
- **SoftMax** — each element's share of LogSumExp, normalised to sum to 1
- **SoftMin** — De Morgan dual of SoftMax

Every function reduces to its exact UA counterpart as T → 0.

## Temperature: the single dial

Temperature T parameterises every function, creating a continuous spectrum between
two reasoning modes:

- **T = 0**: each function collapses to its exact hard counterpart — max, min, or
  step. Operations are crisp and boolean, recovering exact UA semantics.
- **T > 0**: each function becomes smooth and differentiable. Operations are fuzzy
  and graded, enabling analogical reasoning and gradient-based learning.

Lambert does not have two separate modes. Soft and hard reasoning are the same
framework at different temperatures.

## LogSumExp: the fundamental building block

    LSE(x, T) = T × ln(∑ exp(x / T))

This approximates max(x) from above, with a known bounded gap:

    max(x)  ≤  LSE(x, T)  ≤  max(x) + T × ln(n)

where n is the number of elements being reduced. The gap shrinks to zero as T → 0
and grows with temperature and the number of elements. Choosing a small enough
temperature guarantees the smooth operations are within any desired tolerance of
the exact max-min semiring.

**Reference:** Nesterov, Y. (2005). Smooth minimization of non-smooth functions.
*Mathematical Programming, Series A*, 103, 127–152.

## SmoothMax and SmoothMin

SmoothMax is LogSumExp directly — it approximates max from above.

SmoothMin is derived by De Morgan duality:

    SmoothMin(x) = -SmoothMax(-x)

This follows from the law -(x ∨ y) = -x ∧ -y applied in the smooth setting.
Because SmoothMax overshoots max from above, SmoothMin undershoots min from below.
The asymmetry is structural — it follows directly from the single decision to use
LogSumExp as the primitive.

In practice: Join uses SmoothMax (inflating extents slightly upward), while
Residuate uses SmoothMin (tightening constraints slightly downward). These biases
push in opposite directions relative to the fixpoint. At the temperatures Lambert
operates at, the relative ordering of values is preserved and the bias is
operationally acceptable.

## SoftMax and SoftMin

SoftMax distributes the input as shares summing to 1:

    SoftMax(x, T)_i = exp(x_i / T) / ∑ exp(x_j / T)

At T = 0 this collapses to the hard maximum — one element dominates, all others
vanish. At high T every element receives an equal share 1/n. At low T the largest
element dominates proportionally.

When axis is None, SoftMax operates in binary mode — each element's share relative
to 0 — which is equivalent to the sigmoid function. This is used in the attention
layer to normalise a query vector before the Residuate correction step.

SoftMin is the De Morgan dual: SoftMin(x) = -SoftMax(-x).

SoftMax unifies four conventionally distinct functions under a single parameterisation.
Temperature controls sharpness; the axis argument controls whether normalisation is
relative to the full input vector or to zero:

| T | axis | Reduces to |
|---|---|---|
| → 0 | over vector | argmax — one element dominates, all others vanish |
| > 0 | over vector | softmax — graded distribution summing to 1 |
| → 0 | None (binary) | Heaviside step — ⊤ if x > 0, ⊥ otherwise |
| > 0 | None (binary) | sigmoid — smooth 0→1 transition |

The sigmoid case follows directly from softmax: sigmoid(x) is softmax over the
two-element list (0, x), returning x's share of the total — which is exp(x) / (1 + exp(x)).

## Softplus and Relu

Softplus is LogSumExp over the two-element list (0, x) — a smooth approximation
of max(x, 0):

    max(x, 0)  ≤  T × ln(1 + exp(x/T))  ≤  max(x, 0) + T × ln 2

The gap is largest at x = 0, the location of the sharp corner, and vanishes as
T → 0. Relu is the T = 0 special case — the exact max(x, 0) with no smoothing.

## Engine DSL representation

The smooth activation functions are the computational substrate for every semiring
in the engine DSL. When a `SemiringDecl` in `engine/decl.py` declares a contract
pointing to a max-min operation, that contract internally uses `LogSumExp`. This
means Nesterov's bounded-gap approximation propagates through the engine compiler
to every morphism compiled under that semiring.

See [engine/decl](../engine/decl.md) for semiring declaration documentation.
