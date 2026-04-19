# Idempotency and Smoothing

## Idempotency

An operation is **idempotent** if applying it to the same value twice gives the same result as applying it once:

$$a \oplus a = a$$

In plain terms: repeating an input does not change the output. The operation absorbs duplicates.

### In standard arithmetic

Under arithmetic, $\oplus = +$:

$$a + a = 2a$$

This is not $a$ unless $a = 0$. Addition is not idempotent — combining a value with itself produces something larger.

The same holds for $\otimes = \times$:

$$a \times a = a^2$$

Again not $a$ in general. Neither arithmetic operation is idempotent.

### In tropical geometry

Under the tropical semiring, $\oplus = \min$:

$$\min(a, a) = a$$

The first operation is idempotent — the minimum of a value with itself is just that value.

But $\otimes = +$ is not:

$$a + a = 2a$$

So in tropical geometry, $\oplus$ is idempotent and $\otimes$ is not. This reflects what the operations mean: finding the minimum of two identical costs gives nothing new, but paying a cost twice is still paying it twice.

### In fuzzy logic

Under the fuzzy semiring, both operations are idempotent:

$$\max(a, a) = a \qquad \min(a, a) = a$$

Seeing the same evidence twice is the same as seeing it once. The strength of a connection does not grow by repetition.

**References:**
- Hehner, E. C. R. (2007, revised 2021). Unified algebra. *International Journal of Mathematical Sciences*, 1(1), 20–37. https://www.cs.toronto.edu/~hehner/UA.pdf
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.

### Why it matters

Idempotency is a stability property. When an operation is idempotent, running a computation again on its own output changes nothing. The result has already settled.

In arithmetic, this does not happen. Each pass through the tensor equation accumulates more — values grow, and repeating the operation drives the system away from where it was.

In the fuzzy semiring, both operations are idempotent. Applying the tensor equation to its own output changes nothing — the result has already settled. Running the same step twice is the same as running it once.

---

## Differentiability

A function is **differentiable** at a point if it has a well-defined gradient there — if a small change in input produces a proportional, predictable change in output. Differentiability is what makes gradient-based training possible: the gradient tells an optimiser which direction to nudge each parameter to reduce error.

Neural networks trained by backpropagation depend entirely on this. At every step, gradients are computed and propagated backwards through the network. If any operation in the chain has no well-defined gradient, training stops there.

### Why idempotency creates a kink

Consider $\max(a, b)$. When $a > b$, the output follows $a$ completely — a small increase in $a$ increases the output by the same amount, and $b$ has no effect. When $b > a$, the reverse is true.

At the point where $a = b$ — exactly the idempotency locus, where $\max(a, a) = a$ — the function switches abruptly between these two behaviours. The gradient is $1$ with respect to one input and $0$ with respect to the other, but which is which flips discontinuously at $a = b$. There is no well-defined gradient at that point.

The same applies to $\min$. Both operations are piecewise linear: flat in one input, linear in the other, with a ridge running along $a = b$.

This ridge is not incidental. It is the geometric expression of idempotency. The operation must return $a$ when both inputs are $a$ — and the only way a symmetric, monotone operation can do that is by having this exact kink.

### The mutual exclusion

A differentiable operation smooths over the transition between inputs. It cannot have a ridge. But a ridge is precisely what idempotency requires of $\max$ and $\min$.

Put directly: you can have a smooth operation, or an idempotent one, but not both.

- Arithmetic $+$ and $\times$ are smooth and differentiable everywhere. They are not idempotent.
- Fuzzy $\max$ and $\min$ are idempotent. They are not differentiable at $a = b$.

| Operation | Idempotent | Differentiable everywhere |
| --------- | ---------- | ------------------------- |
| $+$       | No         | Yes                       |
| $\times$  | No         | Yes                       |
| $\max$    | Yes        | No                        |
| $\min$    | Yes        | No                        |

### The consequence

A system built on $\max$ and $\min$ cannot be trained directly by backpropagation — the gradients it needs do not exist at the points that matter most.

This is the central tension. The algebraic properties that make the fuzzy semiring well-behaved are the same properties that make it incompatible with standard training.

---

## Smoothing

### Activation functions

A neural network is, at its core, a chain of tensor operations. Each layer takes an input, applies a weight matrix, and produces an output. If that were all, the entire chain would collapse into a single matrix — no matter how many layers you stack, the result would be equivalent to one linear transformation. A network with a hundred layers would be no more expressive than a network with one.

Activation functions break this. After each layer, a non-linear operation is applied to the output before it is passed to the next layer. This is what gives deep networks their expressive power — the ability to represent complex, curved relationships rather than flat linear ones.

### ReLU

The most widely used activation function is **ReLU** (Rectified Linear Unit):

$$\text{ReLU}(x) = \max(x, 0)$$

It passes positive values through unchanged and sets negative values to zero. Simple, fast, and effective.

But notice what it is: a $\max$ operation. The same operation we identified as idempotent and non-differentiable. ReLU has a kink at $x = 0$ — it is flat for $x < 0$ and linear for $x > 0$, with no well-defined gradient at the transition point.

This is not a flaw unique to ReLU. It is the same structural property that appeared in the previous section. The most common activation function in modern AI is already living in this algebra, already with the same kink.

### Softplus

The smooth approximation to ReLU is **Softplus**:

$$\text{Softplus}(x) = \frac{1}{\beta} \ln\left(1 + e^{\beta x}\right)$$

This is differentiable everywhere. As $\beta \to \infty$, it converges to ReLU exactly — the kink sharpens until it becomes the hard corner. At finite $\beta$, the corner is rounded and gradients flow through smoothly.

Softplus is LogSumExp applied to the two-element list $(0, x)$. It returns the smooth version of $\max(x, 0)$.

### LogSumExp

Softplus is a special case of a more general operation. Given any collection of values, **LogSumExp** returns a smooth approximation to their maximum:

$$\text{LSE}(a_1, \ldots, a_n) = \frac{1}{\beta} \ln\left(\sum_i e^{\beta a_i}\right)$$

As $\beta$ increases, the largest value exponentially dominates the sum — the other terms become negligible — and the expression converges to the exact $\max$. At finite $\beta$, it is differentiable everywhere and strictly larger than $\max$.

Softplus is LogSumExp over $(0, x)$. The smooth approximation to $\max(a, b)$ is LogSumExp over $(a, b)$. They are the same operation at different scales.

### Nesterov's guarantee

Nesterov (2005) showed that the error introduced by LogSumExp is not arbitrary — it is bounded and controllable:

$$\max(a_1, \ldots, a_n) \leq \text{LSE}(a_1, \ldots, a_n) \leq \max(a_1, \ldots, a_n) + \frac{\ln n}{\beta}$$

The smooth approximation always overshoots the true $\max$, but never by more than $\frac{\ln n}{\beta}$. Increasing $\beta$ tightens the bound. For any desired precision $\varepsilon$, there exists a $\beta$ large enough to guarantee the error stays within $\varepsilon$.

This is what makes the approximation principled rather than heuristic. The gap is known, one-sided, and shrinks predictably.

### The dial

$\beta$ is the single parameter that controls the trade-off established in the previous section:

- At $\beta \to \infty$: LogSumExp converges to exact $\max$. Operations are idempotent. Gradients vanish at the kink — training is not possible.
- At finite $\beta$: operations are smooth and differentiable everywhere. Gradients flow. Idempotency is violated by a bounded, predictable margin.

This is not a compromise forced by implementation. It is the direct consequence of the mutual exclusion between idempotency and differentiability. $\beta$ navigates that trade-off continuously, and Nesterov's bound tells you exactly how much you are giving up at any point on the dial.

**References:**
- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. *Mathematical Programming*, Series A, 103, 127–152.

---

## Duality

### De Morgan's laws

De Morgan's laws describe a symmetry between $\min$ and $\max$. Negating a minimum gives the maximum of the negated values, and vice versa:

$$-\min(a, b) = \max(-a, -b)$$
$$-\max(a, b) = \min(-a, -b)$$

Rearranging the first:

$$\min(a, b) = -\max(-a, -b)$$

In plain terms: to find the minimum of two values, negate both, take the maximum, then negate the result. $\min$ and $\max$ are mirror images of each other under negation.

**References:**
- Hehner, E. C. R. (2007, revised 2021). Unified algebra. *International Journal of Mathematical Sciences*, 1(1), 20–37. https://www.cs.toronto.edu/~hehner/UA.pdf

### Deriving smooth min from smooth max

LogSumExp gives a smooth approximation to $\max$. De Morgan's law gives smooth $\min$ for free — no separate construction required.

Given:

$$\text{smooth\_max}(a, b) = \frac{1}{\beta} \ln\left(e^{\beta a} + e^{\beta b}\right)$$

apply De Morgan:

$$\text{smooth\_min}(a, b) = -\text{smooth\_max}(-a, -b) = -\frac{1}{\beta} \ln\left(e^{-\beta a} + e^{-\beta b}\right)$$

One primitive. Two operations. The duality handles the rest.

### The asymmetry

LogSumExp always overshoots $\max$ — it returns something slightly above the true maximum, as Nesterov's bound guarantees. Applying De Morgan to derive smooth $\min$ reverses this: smooth $\min$ always undershoots the true minimum, returning something slightly below it.

$$\text{smooth\_min}(a, b) \leq \min(a, b) \leq \max(a, b) \leq \text{smooth\_max}(a, b)$$

The two approximations sit on opposite sides of the exact values. This is not a flaw — it follows directly from the decision to use LogSumExp as the single primitive. The asymmetry is structural, predictable, and bounded by the same $\beta$ parameter that controls both operations.

### Why it matters

The duality means the fuzzy semiring's two operations are not independent choices. Once you commit to LogSumExp for $\max$, De Morgan's law determines the smooth $\min$ completely. The algebra is internally consistent by construction.

Both operations share the same $\beta$ and the same bounded error. The approximation quality is uniform — there is no asymmetry in how closely each smooth operation tracks its exact counterpart.
