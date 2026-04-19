# Mathematical Foundations

- [Mathematical Foundations](#mathematical-foundations)
  - [Semirings](#semirings)
    - [Definition](#definition)
      - [The $\\oplus$ operation](#the-oplus-operation)
      - [The $\\otimes$ operation](#the-otimes-operation)
      - [How the two operations interact](#how-the-two-operations-interact)
      - [Notation summary](#notation-summary)
    - [Arithmetic as a semiring](#arithmetic-as-a-semiring)
    - [Tropical geometry as a semiring](#tropical-geometry-as-a-semiring)
    - [Fuzzy logic as a semiring](#fuzzy-logic-as-a-semiring)
  - [The Tensor Equation](#the-tensor-equation)
    - [The equation](#the-equation)
    - [In standard arithmetic](#in-standard-arithmetic)
    - [In tropical geometry](#in-tropical-geometry)
    - [In fuzzy logic](#in-fuzzy-logic)
    - [Summary](#summary)
  - [Residuals](#residuals)
    - [The idea](#the-idea)
    - [In standard arithmetic](#in-standard-arithmetic-1)
    - [In fuzzy logic](#in-fuzzy-logic-1)
    - [In tropical geometry](#in-tropical-geometry-1)
    - [When the residual fails](#when-the-residual-fails)
    - [The pattern](#the-pattern)
    - [At the matrix level](#at-the-matrix-level)
  - [Reasoning Chains](#reasoning-chains)
    - [One step, then two](#one-step-then-two)
    - [Where the chain breaks](#where-the-chain-breaks)
    - [Where the chain holds](#where-the-chain-holds)
    - [What this means](#what-this-means)


## Semirings

### Definition

A **semiring** is a set $S$ equipped with two binary operations, written $\oplus$ and $\otimes$, satisfying the rules below.

#### The $\oplus$ operation

- **Closure:** for any $a, b \in S$, the result $a \oplus b$ is also in $S$.
- **Associativity:** $(a \oplus b) \oplus c = a \oplus (b \oplus c)$.
- **Commutativity:** $a \oplus b = b \oplus a$.
- **Identity $\varepsilon$:** there exists an element $\varepsilon \in S$ such that $a \oplus \varepsilon = a$ for all $a$.

#### The $\otimes$ operation

- **Closure:** for any $a, b \in S$, the result $a \otimes b$ is also in $S$.
- **Associativity:** $(a \otimes b) \otimes c = a \otimes (b \otimes c)$.
- **Identity $\mathbf{e}$:** there exists an element $\mathbf{e} \in S$ such that $\mathbf{e} \otimes a = a \otimes \mathbf{e} = a$ for all $a$.

#### How the two operations interact

- **Distributivity:** $\otimes$ spreads over $\oplus$ from both sides:
  $$a \otimes (b \oplus c) = (a \otimes b) \oplus (a \otimes c)$$
  $$(a \oplus b) \otimes c = (a \otimes c) \oplus (b \otimes c)$$
- **Annihilation:** the $\oplus$-identity $\varepsilon$ kills everything under $\otimes$:
  $$\varepsilon \otimes a = a \otimes \varepsilon = \varepsilon$$

#### Notation summary

| Symbol | Role |
|--------|------|
| $S$ | the set of values the semiring operates over |
| $\oplus$ | first binary operation |
| $\otimes$ | second binary operation |
| $\varepsilon$ | identity for $\oplus$; annihilator for $\otimes$ |
| $\mathbf{e}$ | identity for $\otimes$ |

### Arithmetic as a semiring

The familiar numbers under addition $+$ and multiplication $\times$ form a semiring. The correspondence is direct.

| Abstract | Arithmetic |
|----------|------------|
| $S$ | $\mathbb{R}$ (or $\mathbb{N}$, $\mathbb{Z}$, …) |
| $\oplus$ | $+$ |
| $\otimes$ | $\times$ |
| $\varepsilon$ | $0$ |
| $\mathbf{e}$ | $1$ |

Each axiom holds:

- **Addition** $+$ is associative, commutative, and has identity $0$: 
  $$(a + b) + c = a + (b + c)$$ 
  $$a + b = b + a$$
  $$a + 0 = a$$
- **Multiplication** $\times$ is associative and has identity $1$: 
  $$(a \times b) \times c = a \times (b \times c)$$ 
  $$a \times 1 = a$$
- **Distributivity:** $a \times (b + c) = a \times b + a \times c$.
- **Annihilation:** $0 \times a = 0$.

Arithmetic is the semiring most people already know — the abstract definition strips away everything except these four properties.

### Tropical geometry as a semiring

Tropical geometry replaces the two familiar arithmetic operations with a different pair. The underlying set is $\mathbb{R} \cup \{\infty\}$, the union of the real numbers and infinity. The operations are minimum $\wedge$ and addition $+$:

| Abstract | Tropical |
|----------|----------|
| $S$ | $\mathbb{R} \cup \{\infty\}$ |
| $\oplus$ | $\wedge$ |
| $\otimes$ | $+$ |
| $\varepsilon$ | $\infty$ |
| $\mathbf{e}$ | $0$ |

Each axiom holds:

- **Minimum** $\wedge$ is associative, commutative, and has identity $\infty$: 
  $$(a \wedge b) \wedge c = a \wedge (b \wedge c)$$ 
  $$a \wedge b = b \wedge a$$ 
  $$a \wedge \infty = a$$
- **Addition** $+$ is associative and has identity $0$: 
  $$(a + b) + c = a + (b + c)$$ 
  $$a + 0  = a$$
- **Distributivity:** $a + (b \wedge c) = (a + b) \wedge (a + c)$.
- **Annihilation:** $\infty + a = \infty$.

The striking thing is that minimum $\wedge$ satisfies the same abstract role as addition $+$ in arithmetic, and addition $+$ satisfies the same role as multiplication $\times$. Nothing in the semiring axioms demands the operations look like the familiar ones — only that the axioms hold.

**References:**
- Zhang, L., Naitzat, G., & Lim, L.-H. (2018). Tropical geometry of deep neural networks. *ICML 2018*, PMLR 80. arXiv:1805.07091.
- Alfarra, M. H. A. (2020). Applications of tropical geometry in deep neural networks. MSc Thesis, KAUST.
- Maragos, P., Charisopoulos, V., & Theodosis, E. (2021). Tropical geometry and machine learning. *Proceedings of the IEEE*, 109(5), 2073–2088. DOI: 10.1109/JPROC.2021.3065238.
- Alfarra, M., Bibi, A., Hammoud, H., Gaafar, M., & Ghanem, B. (2023). On the decision boundaries of neural networks: a tropical geometry perspective. *IEEE TPAMI*, 45(4), 5027–5037. DOI: 10.1109/TPAMI.2022.3201490.

### Fuzzy logic as a semiring

Fuzzy logic works over truth values in the interval $[0, 1]$, where $0$ means fully false and $1$ means fully true. The operations are maximum $\vee$ and minimum $\wedge$:

| Abstract | Fuzzy logic |
|----------|-------------|
| $S$ | $[0, 1]$ |
| $\oplus$ | $\vee$ |
| $\otimes$ | $\wedge$ |
| $\varepsilon$ | $0$ |
| $\mathbf{e}$ | $1$ |

Each axiom holds:

- **Maximum** $\vee$ is associative, commutative, and has identity $0$:
  $$(a \vee b) \vee c = a \vee (b \vee c)$$
  $$a \vee b = b \vee a$$
  $$a \vee 0 = a$$
- **Minimum** $\wedge$ is associative and has identity $1$:
  $$(a \wedge b) \wedge c = a \wedge (b \wedge c)$$
  $$a \wedge 1 = a$$
- **Distributivity:** $a \wedge (b \vee c) = (a \wedge b) \vee (a \wedge c)$.
- **Annihilation:** $0 \wedge a = 0$.

Here $\vee$ combines evidence by taking the stronger signal, and $\wedge$ measures joint truth by taking the weaker one. The semiring structure is what makes it possible to propagate fuzzy values through a computation in a principled way.

**References:**
- Hehner, E. C. R. (2004). From boolean algebra to unified algebra. *The Mathematical Intelligencer*, 26(2), 3–19. DOI: 10.1007/BF02985647.
- Hehner, E. C. R. (2007, revised 2021). Unified algebra. *International Journal of Mathematical Sciences*, 1(1), 20–37. https://www.cs.toronto.edu/~hehner/UA.pdf

---

## The Tensor Equation

### The equation

A semiring gives two operations, $\oplus$ and $\otimes$. The tensor equation puts them to work on arrays.

Given two matrices $A$ and $B$, their contraction along a shared index $j$ produces a new matrix $C$:

$$C_{ik} = \bigoplus_j \; A_{ij} \otimes B_{jk}$$

For each output cell $(i, k)$: combine every path through the shared index $j$ — first pair entries with $\otimes$, then accumulate across all $j$ with $\oplus$.

This is the only equation. Everything else is a choice of semiring.

### In standard arithmetic

Under arithmetic ($\oplus = +$, $\otimes = \times$), the equation is standard matrix multiplication:

$$C_{ik} = \sum_j A_{ij} \times B_{jk}$$

For each output cell, multiply matching entries row-by-column and sum the results. This is the core operation in linear algebra and the building block of neural networks — every layer of a neural network is this equation applied to its weight matrix.

**References:**
- Domingos, P. (2025). Tensor logic: The language of AI. arXiv:2510.12269.
- Shah, S., & Zadrozny, W. (2026). Implementing tensor logic: Unifying Datalog and neural reasoning via tensor contraction. arXiv:2601.17188.

### In tropical geometry

Under the tropical semiring ($\oplus = \wedge$, $\otimes = +$):

$$C_{ik} = \bigwedge_j \; (A_{ij} + B_{jk})$$

Now $A_{ij}$ is the cost of travelling from $i$ to $j$, and $B_{jk}$ the cost from $j$ to $k$. Adding gives the total cost of the two-hop path; $\bigwedge_j$ finds the cheapest route.

This is the Bellman-Ford shortest-path step — one of the central algorithms in graph theory — expressed as a single tensor equation under a different semiring.

**References:**
- Zhang, L., Naitzat, G., & Lim, L.-H. (2018). Tropical geometry of deep neural networks. *ICML 2018*, PMLR 80. arXiv:1805.07091.
- Maragos, P., Charisopoulos, V., & Theodosis, E. (2021). Tropical geometry and machine learning. *Proceedings of the IEEE*, 109(5), 2073–2088. DOI: 10.1109/JPROC.2021.3065238.
- Alfarra, M., Bibi, A., Hammoud, H., Gaafar, M., & Ghanem, B. (2023). On the decision boundaries of neural networks: a tropical geometry perspective. *IEEE TPAMI*, 45(4), 5027–5037. DOI: 10.1109/TPAMI.2022.3201490.

### In fuzzy logic

Under the fuzzy semiring ($\oplus = \vee$, $\otimes = \wedge$), the equation becomes Zadeh's max-min composition:

$$C_{ik} = \bigvee_j \; (A_{ij} \wedge B_{jk})$$

Now $A_{ij}$ is a degree of membership — how strongly $i$ relates to $j$ — and $B_{jk}$ is how strongly $j$ relates to $k$. For each intermediate $j$, $\wedge$ finds the weakest link in the chain $i \to j \to k$. Then $\bigvee_j$ selects the best such chain.

The result is a relational inference: $C_{ik}$ is the strength of the best two-hop path from $i$ to $k$ through any intermediate node $j$.

This is the same equation as before. Only the semiring has changed.

**References:**
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. *Information and Control*, 30(1), 38–48.
- Hehner, E. C. R. (2004). From boolean algebra to unified algebra. *The Mathematical Intelligencer*, 26(2), 3–19. DOI: 10.1007/BF02985647.
- Hehner, E. C. R. (2007, revised 2021). Unified algebra. *International Journal of Mathematical Sciences*, 1(1), 20–37. https://www.cs.toronto.edu/~hehner/UA.pdf

### Summary

The equation $C_{ik} = \bigoplus_j A_{ij} \otimes B_{jk}$ is identical in all three cases. Only the semiring changes:

| Semiring | $\oplus$ | $\otimes$ | What it computes |
|----------|----------|-----------|-----------------|
| Arithmetic | $+$ | $\times$ | weighted sum of paths |
| Fuzzy logic | $\vee$ | $\wedge$ | strongest two-hop relation |
| Tropical | $\wedge$ | $+$ | cheapest two-hop path |

Any computation expressible as a tensor contraction inherits this freedom — change the semiring, change the meaning, keep the structure.

---

## Residuals

### The idea

The second operation $\otimes$ of a semiring takes two inputs and produces an output:

$$a \otimes b = c$$

The residual asks the reverse question: given $a$ and $c$, what is the largest $b$ such that $a \otimes b \leq c$?

This is written $a \backslash c$ (read: "$a$ under $c$"), and called the **right residual** of $\otimes$. Symmetrically, the largest $b$ such that $b \otimes a \leq c$ is written $c / a$ and called the **left residual**.

When $\otimes$ is commutative, the two coincide.

### In standard arithmetic

Under arithmetic, $\otimes = \times$. The residual asks: what is the largest $b$ such that $a \times b \leq c$?

$$a \backslash c = \frac{c}{a}$$

The residual of $\times$ is division. This is not a coincidence — it is exactly what division means.

### In fuzzy logic

Under the fuzzy semiring, $\otimes = \wedge$. The residual asks: what is the largest $b$ such that $a \wedge b \leq c$?

- If $a \leq c$: then $(a \wedge b) \leq a \leq c$ no matter what $b$ is. So $b$ can be as large as $\mathbf{e} = 1$.
- If $a > c$: then $a \wedge b \leq c$ requires $b \leq c$. The largest such $b$ is $c$ itself.

$$a \backslash c = \begin{cases} 1 & \text{if } a \leq c \\ c & \text{if } a > c \end{cases}$$

This is the **Gödel implication** $(a \to c)$ from fuzzy logic: if $a$ does not exceed $c$, there is no constraint on $b$; if $a$ exceeds $c$, $b$ is pulled down to $c$.

**References:**
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. *Information and Control*, 30(1), 38–48.

### In tropical geometry

Under the tropical semiring, $\otimes = +$. The residual asks: what is the largest $b$ such that $a + b \leq c$?

$$a \backslash c = c - a$$

The residual of $+$ is subtraction. Again, not a coincidence.

### When the residual fails

The residual does not always exist. For arithmetic, $c / a$ is undefined when $a = 0$. For the tropical semiring, $c - a$ is undefined when $a = +\infty$ (the annihilator). In both cases the failure occurs at the annihilator element. The fuzzy semiring does not have this problem: the Gödel implication $(a \to c)$ is defined for all $a, c \in [0,1]$, including at $a = 0$.

### The pattern

| Semiring | $\otimes$ | Residual $a \backslash c$ |
|----------|-----------|--------------------------|
| Arithmetic | $\times$ | $c / a$ |
| Fuzzy logic | $\wedge$ | $a \to c$ |
| Tropical | $+$ | $c - a$ |

Each semiring carries its residual with it. Division, implication, and subtraction are not three separate ideas — they are the same operation instantiated under three different semirings.

### At the matrix level

The residual extends to the tensor equation. If $A \otimes B \leq C$ is the forward question — given $A$ and $B$, compute $C$ — the residual asks the backward question: given $A$ and $C$, find the largest $B$ consistent with them.

Under the fuzzy semiring this is:

$$B_{jk} = \bigwedge_i \left( A_{ij} \to C_{ik} \right) = \bigwedge_i \begin{cases} 1 & \text{if } A_{ij} \leq C_{ik} \\ C_{ik} & \text{if } A_{ij} > C_{ik} \end{cases}$$

Sanchez (1976) calls this the relative pseudocomplement. For each entry $B_{jk}$: scan every row $i$ of $A$. Row $i$ tells you how strongly $i$ connects to $j$; $C_{ik}$ tells you how strongly $i$ is allowed to connect to $k$ in the output. Each row places a ceiling on $B_{jk}$ — if the connection through $j$ is already weak, the ceiling is $1$; if it is strong, the ceiling is lower. The entry $B_{jk}$ is the lowest of all those ceilings.

Standard linear algebra has an analogous operation — the matrix pseudoinverse recovers an input from an output and a weight matrix. In practice it is never used this way in machine learning, because zeros introduced by activation functions like ReLU permanently destroy information. Once a value has been zeroed out, no inversion can recover what caused it. The relative pseudocomplement does not have this problem: a zero in $A$ places no constraint on $B$ and is simply ignored. The backwards pass is always well-defined.

**References:**
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. *Information and Control*, 30(1), 38–48.

---

## Reasoning Chains

### One step, then two

The previous section showed that the residual of a single tensor operation lets you go backwards: given the output $C$ and one input $A$, recover the other input $B$.

Now consider two operations in sequence. A first step produces an intermediate result, and a second step produces the final output:

$$A \xrightarrow{\text{step 1}} B \xrightarrow{\text{step 2}} C$$

Each step is a tensor contraction under some semiring. To trace back from $C$ to $A$:

1. Apply the residual of step 2 to $C$, recovering $B$.
2. Apply the residual of step 1 to $B$, recovering $A$.

The same logic extends to any number of steps. A chain of ten operations can be traced back through ten residuals, one at a time, arriving at the original inputs.

This is the key idea. **A chain of operations is only fully traceable if every residual in the chain is defined.**

### Where the chain breaks

In standard arithmetic, the residual of multiplication $\times$ is division $\div$. Division $\div$ fails at zero. This means that if any intermediate result passes through a zero — and in a deep network, values pass through zero constantly — the backwards trace cannot continue. The chain breaks.

This is not a practical inconvenience. It is a structural property of the arithmetic semiring: tracing backwards is not always possible.

### Where the chain holds

Under the fuzzy semiring, the residual of minimum $\wedge$ is the Gödel implication $(a \to c)$, and it is defined for every pair of values in $[0,1]$, including at zero. No value can break the chain.

This means: given any output, you can always apply residuals backwards through every step and arrive at a consistent account of what inputs produced it. The trace is always available. Nothing is lost.

### What this means

Under the fuzzy semiring, every output carries a complete record of the reasoning that produced it — not as a separate log, but as a consequence of the algebra itself. The same operation that computes the forward result also supports the backward question.

This is not limited to two steps or ten steps. It holds for any chain of any length, because the guarantee comes from the semiring, not from the size of the computation.

A system built on a fully residuated semiring is, by construction, one whose outputs can always be questioned. Given any conclusion, you can ask what it was built from. Given any intermediate result, you can ask the same. The answer is always available, and the algebra provides it.

This is what distinguishes the fuzzy semiring from the arithmetic one for the purposes of reasoning. It is not that fuzzy logic is more expressive, or more accurate. It is that the residual is always defined — and that single property, carried forward through a chain of operations, is what makes a transparent reasoning system possible.

**References:**
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. *Information and Control*, 30(1), 38–48.
- Hehner, E. C. R. (2007, revised 2021). Unified algebra. *International Journal of Mathematical Sciences*, 1(1), 20–37. https://www.cs.toronto.edu/~hehner/UA.pdf
