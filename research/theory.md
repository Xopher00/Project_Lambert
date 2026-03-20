# Theory

Mathematical foundations of Project Lambert, organised by layer.

## Core stack

- [algebra](core/algebra.md) — Unified Algebra, Top/Bottom, Max, Implies, Refutes
- [activations](core/activations.md) — LogSumExp, temperature spectrum, smooth UA operators
- [tensor](core/tensor.md) — max-min semiring, Join, Residuate, Closure, witness tracking
- [fixpoint](core/fixpoint.md) — fixpoint iteration, energy function, temperature annealing

## Lattice layer

- [embed](lattice/embed.md) — formal concepts, ConceptEmbed, coverage-based concept selection
- [attention](lattice/attention.md) — attention as dense associative memory, correction principle, conjunctive queries, multi-head retrieval
- [explorer](lattice/explorer.md) — multi-relational formal concepts, second-order FCA, lattice closure

---

## Query semantics

Lambert currently lacks a formal specification of what a query *is* and what a valid answer *means* algebraically. The retrieval mechanism works and provably converges, but the semantics are implicit. Making them explicit is the single highest-leverage theoretical step remaining, because provenance, multi-head combination, lattice navigation, and scaling via Tucker decomposition all depend on it.

### The definition

A query is a partial attribute vector `q ∈ [0,1]^m` — some values known, others zero (unknown). The correct answer is the **smallest formal concept whose intent contains q**:

```
B = Residuate(R, q)      # intent: attributes implied by q
A = Residuate(R.T, B)    # extent: entities consistent with that intent
```

The answer is the concept `(A, B)`. The extent `A` is the set of entities satisfying the query. The intent `B` is the set of attributes the answer implies — provenance falls out directly from the lattice structure, no separate mechanism required.

This is exactly what `_concept_fixpoint` computes. Bělohlávek (2000) Theorem 1 proves it converges to this concept in at most two steps.

### What this unlocks

- **Provenance**: `B` is the reason set. Attributes in the intent are what the query entails.
- **Multi-head combination**: The correct answer across heads is the meet of their answer concepts — the largest concept contained in all simultaneously (Bělohlávek Theorem 2). The hard `np.minimum` in `_outer_step` computes this correctly.
- **Lattice navigation**: More specific = meet with another concept (add constraints). More general = join (relax constraints). Both are operations on `emb` columns via the partial order.
- **Tucker decomposition**: Once query semantics is grounded, `Project(R, emb)` is the correct object for multi-hop queries in compressed concept space (Domingos 2025).

---

## References

### Unified Algebra

- Hehner, E.C.R. (2004). From Boolean Algebra to Unified Algebra. *The Mathematical Intelligencer*, 26(2), 3–19.
- Hehner, E.C.R. (2007, revised 2021). *Unified Algebra.* International Journal of Mathematical Sciences, 1(1), 20–37.

### Fuzzy set theory and relational composition

- Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
  — foundation of the max-min semiring; max-min relational composition. Referenced in [tensor](core/tensor.md).
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. *Information and Control*, 30, 38–48.
  — greatest solution to `A ∘ B = C` under max-min composition (Theorem 5); basis for `Residuate`, Join, and the attention correction step. Referenced in [tensor](core/tensor.md), [embed](lattice/embed.md), [attention](lattice/attention.md).
- Kaufmann, A. *Introduction to the Theory of Fuzzy Subsets.* Ch. 1, p. 39.

### Smooth approximations

- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. *Mathematical Programming*, 103(1), 127–152.
  — bounded-gap approximation property of LogSumExp. Referenced in [activations](core/activations.md).

### Fixpoint theory

- Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its applications. *Pacific Journal of Mathematics*, 5(2), 285–309.
  — guarantees convergence of monotone operators on complete lattices. Referenced in [embed](lattice/embed.md).

### Fuzzy logical associative memory

- Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory. *Neural Network World*, 10(5).
  — Theorem 1: Lambert's `_concept_fixpoint` (`O*`/`A∧` alternation with Gödel implication) converges to a formal concept in exactly two steps via idempotence of Galois adjunctions (Ore 1944). Theorem 2: stable points form the complete concept lattice. Theorem 6: constructive learning rule for R from labeled concepts. Lambert's algebra (min, Gödel implication) is Example 2 — one of three canonical cases the theorems directly cover. Referenced in [embed](lattice/embed.md), [attention](lattice/attention.md).

### Formal concept analysis

- Brito, P. et al. *Fuzzy Formal Concept Analysis.*
  — Definition 15: fuzzy formal context `⟨O, A, I_f⟩`; Definition 17: formal concept as simultaneous fixpoint of `O*` and `A∧`; Theorem 8: completeness of the concept lattice. Referenced in [embed](lattice/embed.md), [attention](lattice/attention.md), [explorer](lattice/explorer.md).
- Trnecka, M. & Vyjidacek, R. (2020). Revisiting the GreCon Algorithm for Boolean Matrix Factorization. *CLA 2020.*
  — coverage-based concept selection as matrix decomposition. Referenced in [embed](lattice/embed.md).
- Belohlavek, R. & Vychodil, V. (2007). Fuzzy concept lattices constrained by hedges. *JACIII*, 11.
  — fuzzy FCA with graded membership. Referenced in [embed](lattice/embed.md).

### Associative memory and attention

- Krotov, D. & Hopfield, J. (2021). Large Associative Memory Problem in Neurobiology and Machine Learning. *ICLR 2021.*
  — §3.2 Model B: derivation of transformer attention as the fast-memory limit of dense associative memory; partial pattern initialisation and attractor convergence. Referenced in [attention](lattice/attention.md), [explorer](lattice/explorer.md).
- Ramsauer, H. et al. (2020). Hopfield Networks is All You Need. *arXiv:2008.07320.*
  — modern Hopfield networks and their connection to attention. Referenced in [embed](lattice/embed.md).

### Predictive coding and free energy

- Parr, T., Pezzulo, G. & Friston, K.J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
  — eq. 4.19: energy as sum of prediction errors; theoretical basis for the energy function in `FixpointIterator`. Referenced in [fixpoint](core/fixpoint.md).

### Tensor logic and temperature

- Domingos, P. (2025). *Tensor Logic: The Language of AI.* arXiv:2510.12269v3. https://arxiv.org/abs/2510.12269
  — Datalog rules as einsums; Join as tensor logic relational composition; temperature spectrum from deductive (T=0) to analogical (T>0); optimal T varies by data sparsity. Referenced in [tensor](core/tensor.md), [fixpoint](core/fixpoint.md).
