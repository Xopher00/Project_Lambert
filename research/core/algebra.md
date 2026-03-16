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
takes from UA is the conceptual structure: Top and Bottom as domain extrema, max as the
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

## Max: the core operation

Max (∨), read "x max y", is the least upper bound of its inputs — the smallest value
that is at least as large as both. On the boolean domain it is disjunction: the result
is Bottom only when both inputs are Bottom. On the real number line it is the numeric
maximum. As a quantifier over a domain, ∨⟨x: D· f(x)⟩ is existential quantification —
"there exists an x in D such that f holds."

These are not three separate operations that happen to share laws. Hehner's central
thesis is that they are one operation, and that the apparent differences between logic
and arithmetic are differences of domain, not of kind. Lambert inherits this directly:
the same Max function is used for logical inference, numeric comparison, and existential
reasoning throughout the codebase.

## Implies and Refutes

These two functions implement fuzzy logical operators derived from fuzzy set theory.

**Implies(a, b)** asks: does a imply b? It returns Top if a ≤ b — the implication holds
without restriction. Otherwise it returns b, capping at the weaker value. This is the α
operation from Sanchez (1976), the algebraic foundation of the Residuate operation in
`tensor.py`.

**Refutes(a, b)** is the dual of Implies. It is derived from the dual Brouwerian lattice
— a lattice L where for all a, b ∈ L, the set {x ∈ L : sup(a, x) > b} has a greatest
lower bound, denoted a ε b (Kaufmann). By duality with α: if a ≥ b, b is fully dominated
and the result is Bottom; otherwise b stands. The name "Refutes" is our own — the source
does not use it.

One important caveat: the unit interval [0,1] is neither Brouwerian nor dual Brouwerian
in the strict lattice-theoretic sense, so this operation is best understood as an
extension of the dual Brouwerian structure to [0,1] rather than a strict consequence
of it.

**References:**
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations.
  *Information and Control*, 30, 38–48. (Implies / α operation)
- Kaufmann, A. *Introduction to the Theory of Fuzzy Subsets.* Ch. 1, p. 39.
  (Refutes / dual Brouwerian ε operation)
