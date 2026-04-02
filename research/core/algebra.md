# algebra

The foundation of the entire system. Every other module imports from this one.

## Inspiration: Unified Algebra

This module is directly inspired by Eric Hehner's Unified Algebra (UA) — a framework
that unifies logic and arithmetic by treating them as the same operations applied to
different domains. The key insight is that logic and arithmetic differ not in kind but
in scope: logical operators are just arithmetic operators applied to a domain whose only
values are its two extrema.

Hehner deliberately avoids the terms "true" and "false", using Top (⊤) and Bottom (⊥)
instead — the greatest and least elements of whatever domain the system is working in.
In boolean logic, the domain has only two values and they happen to be its extrema. On
the real number line, the extrema are +∞ and −∞. The operators (max, min, negation)
work uniformly across all domains without special-casing.

Project Lambert would not have been realized without this as a starting point. UA gave
the project its core insight: that logical inference and numeric computation are the same
operation viewed at different points on a spectrum, which makes the model's internal
reasoning interpretable by construction.

The implementation here does not reproduce Hehner's class hierarchy or notation. What it
takes from Unified Algebra is the conceptual structure: Top and Bottom as domain extrema, max as the
fundamental operation, and the identification of logical implication with the ordering
relation.

**References:**
- Hehner, E.C.R. (2004). From Boolean Algebra to Unified Algebra. *The Mathematical
  Intelligencer*, Springer, 26(2), pp. 3–19. https://www.cs.toronto.edu/~hehner/BAUA.pdf
- Hehner, E.C.R. (2007, revised 2021). *Unified Algebra.* International Journal of
  Mathematical Sciences, 1(1), pp. 20–37. https://www.cs.utoronto.ca/~hehner/UA.pdf

## Top and Bottom

```python
Top    =  1e9   # practical stand-in for +∞
Bottom = -1e9   # practical stand-in for −∞
```

Top and Bottom are the greatest and least elements of the domain. For boolean logic the
domain has only two values and they are its extrema. For real numbers the extrema are
+∞ and −∞. Here they are set to 1e9 and -1e9 as practical stand-ins: NumPy handles
finite values more predictably than `float('inf')` in edge cases involving comparison
and composition, and these values are large enough that no legitimate computation reaches
them.

## Max and Min

Hehner's central thesis is that the apparent differences between logic and arithmetic
are differences of domain, not of kind. The same two operations — Max and Min — appear
in every domain wearing different names. Lambert inherits this directly: the same Max
and Min are used for logical inference, numeric comparison, set operations, and
quantification throughout the codebase.

**Max (∨)**, read "x max y", is the least upper bound of its inputs — the smallest
value that is at least as large as both:

| Domain | Operation | Notation |
|---|---|---|
| Boolean logic | disjunction — logical OR (∨) | x ∨ y |
| Arithmetic | numeric maximum | max(x, y) |
| Set theory | union (∪) | A ∪ B |
| Existential quantification | there exists | ∃x ∈ D: f(x)  (UA: ∨⟨x: D· f(x)⟩) |

**Min (∧)**, read "x min y", is the greatest lower bound of its inputs — the largest
value that is no greater than both. It is the dual of Max under negation:
-(x ∧ y) = -x ∨ -y.

| Domain | Operation | Notation |
|---|---|---|
| Boolean logic | conjunction — logical AND (∧) | x ∧ y |
| Arithmetic | numeric minimum | min(x, y) |
| Set theory | intersection (∩) | A ∩ B |
| Universal quantification | for all | ∀x ∈ D: f(x)  (UA: ∧⟨x: D· f(x)⟩) |

Max over a domain is existence; Min over a domain is universality.

Min is not defined as a standalone function in `algebra.py` — it is implemented
directly as `SmoothMin` in the activations layer and as `np.minimum` inline where
needed. It is documented here because understanding it is essential to understanding
Join, Residuate, and the max-min semiring throughout the codebase.

## Implies

`Implies(a, b)` asks: does a imply b? In Unified Algebra, implication is the ordering relation ≤.
These are all the same statement:

| Domain | Reading | Notation |
|---|---|---|
| Logic | a implies b | a → b  (UA: a ≤ b) |
| Sets | A is contained in B | A ⊆ B |
| Arithmetic | a is less than or equal to b | a ≤ b |

`Implies` implements a fuzzy version of this: it returns Top if a ≤ b — the implication holds without
restriction. Otherwise it returns b, capping at the weaker value. This is the α
operation from Sanchez (1976), the algebraic foundation of the Residuate operation in
`tensor.py`.

**Reference:**
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
  *Information and Control*, 30, 38–48. (α operation)
