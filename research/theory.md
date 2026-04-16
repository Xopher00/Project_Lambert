# Theory

Mathematical foundations of Project Lambert, organised by layer.

## Core stack

- [algebra](core/algebra.md) — Unified Algebra, Top/Bottom, Max, Implies
- [activations](core/activations.md) — LogSumExp, temperature spectrum, smooth UA operators
- [tensor](core/tensor.md) — max-min semiring, Join, Residuate, Closure, witness tracking
- [fixpoint](core/fixpoint.md) — fixpoint iteration, energy function, temperature annealing

## Lattice layer (legacy — see legacy/)

These documents describe the original implementation, now preserved in `legacy/`. The patterns they describe are expressed through the engine DSL in `engine/`.

- [embed](legacy/embed.md) — formal concepts, ConceptEmbed, coverage-based concept selection
- [attention](legacy/attention.md) — attention as dense associative memory, correction principle, conjunctive queries, multi-head retrieval
- [explorer](legacy/explorer.md) — multi-relational formal concepts, second-order FCA, lattice closure

---

## Engine DSL (active development)

`engine/` implements a domain-specific language for expressing ML architectures as
compositions of typed morphisms over arbitrary semirings. The DSL makes the categorical
structure described in the preceding sections executable.

- [decl](engine/decl.md) — `SemiringDecl`, `MorphismDecl`, `PathDecl`, `FanDecl`, `CaseDecl`, `ArchDecl` (Lawvere 1973, Gavranović 2024, Green 2007, Domingos 2025, Schultz 2017/2025)
- [functor](engine/functor.md) — `Functor`, `Case`, `Interpreter`; catamorphism and anamorphism (Gavranović 2024, Tarski 1955)
- [runtime](engine/runtime.md) — `MorphismSpec`, `chain`, `fan`, `check_sorts`; V-functor composition (Lawvere 1973, Shen & Tang 2021)
- [compiler](engine/compiler.md) — five-phase compilation pipeline; semiring resolution, template instantiation, iterate groups (Gavranović 2024, Green 2007, Dannert 2021, Domingos 2025)
- [arch](engine/arch.md) — `ArchDef`, `ArchInterpreter`; algebra/coalgebra duality, convergence (Gavranović 2024, Ramsauer 2021, Schultz 2025)

**Key files:** `parser.py` (grammar → AST), `compiler.py` (AST → `ArchDef`), `decl.py` (declarations), `functor.py` (recursive types), `runtime.py` (morphism composition), `arch.py` (compiled runtime), `sorts.py` (Hydra type bridge), `primitives.py` (Hydra primitives).

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

### The Kan framing

The query semantics defined above — find the smallest formal concept whose intent
contains q — is the **right Kan extension** of the query q along the relation R,
evaluated at the identity. Concretely:

```
Π_R(q) = Residuate(R, q)      # right Kan: universal, "all entities consistent with q"
Σ_R(q) = Join(q, R)           # left Kan:  existential, "entities reachable from q"
```

These are dual operations. The right Kan gives the tightest upper bound — the minimal
set of entities that must be in the answer given the constraints. The left Kan gives
the broadest reachable set — all entities that can be reached from q through at least
one path in R.

Lambert currently implements only the right Kan path. Every query goes through
`Residuate` (as `_concept_fixpoint`), retrieving the smallest concept above q. This
is the **universal** semantics: an entity appears in the answer only if it is
consistent with all constraints simultaneously.

The **left Kan path** — `Join(q, EmbR)` chained across relation types — would give the
existential semantics: entities reachable from q through at least one relational chain.
This is generation rather than retrieval. The algebra is present; the query path is not.

### The adjoint triple and multi-hop queries

CQL's adjoint triple (Σ ⊣ Δ ⊣ Π) applied to Lambert's concept space gives the correct
structure for multi-hop inference. Let EmbR_i be the Tucker core for relation type i.
A two-hop query "entities related to q via R₁, then R₂" is:

```
q₁ = Join(q,  EmbR₁)     # one hop forward (left Kan along R₁)
q₂ = Join(q₁, EmbR₂)     # second hop forward (left Kan along R₂)
A  = Join(q₂, emb.T)     # project back to entity space
```

Each `Join` step is a left Kan extension in concept space. The chain is well-typed:
`EmbR: (k, k)` maps concept vectors to concept vectors, so the composition is valid.
Projecting back through `emb.T` gives the entity-level answer.

The right Kan version (universal multi-hop: "entities consistent with all relations
simultaneously") uses Residuate at each step instead:

```
B  = Residuate(EmbR₁, q)
B₂ = Residuate(EmbR₂, B)
A  = Residuate(emb.T, B₂)
```

The meet-based multi-head combination in `_outer_step` is the right Kan path applied
across heads in parallel. The missing generation capability is the left Kan path applied
sequentially across hops.

Schultz, Wisnesky, Vasilakopoulou & Spivak (2017) show that all standard relational
algebra operations factor through the adjoint triple (Σ ⊣ Δ ⊣ Π) — every query is a
composition of left Kan extension, restriction, and right Kan extension.

**Reference:** Schultz, P., Wisnesky, R., Vasilakopoulou, C., & Spivak, D. I. (2017). Algebraic databases.
*Theory and Applications of Categories*, 32(16), 547–619.  cite{schultz2017}

---

## References

### Unified Algebra

- Hehner, E.C.R. (2004). From Boolean Algebra to Unified Algebra. *The Mathematical Intelligencer*, 26(2), 3–19.
- Hehner, E.C.R. (2007, revised 2021). *Unified Algebra.* International Journal of Mathematical Sciences, 1(1), 20–37.

### Fuzzy set theory and relational composition

- Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
  — foundation of the max-min semiring; max-min relational composition. Referenced in [tensor](core/tensor.md).
- Sanchez, E. (1976). Resolution of composite fuzzy relation equations. *Information and Control*, 30, 38–48.
  — greatest solution to `A ∘ B = C` under max-min composition (Theorem 5); basis for `Residuate`, Join, and the attention correction step. Referenced in [tensor](core/tensor.md), [embed](legacy/embed.md), [attention](legacy/attention.md).
- Shen, L., & Tang, X. (2021). Isbell adjunctions and Kan adjunctions via quantale-enriched two-variable adjunctions. *Applied Categorical Structures*, 29, 823–858.
  — situates Lambert's Join ⊣ Residuate pair inside the framework of quantale-enriched two-variable adjunctions (Definition 3.3). Lambert's quantale is V = ([0,1], min, 1) with Gödel implication as residuum — one of the paper's canonical cases. Key results: (1) every V-bifunctor φ: A^op ⊗ B → Z induces an Isbell adjunction φ↑ ⊣ φ↓ whose fixed points Mφ form a complete V-category (Theorem 6.2) — this is Lambert's concept lattice; (2) the Kan adjunctions (Proposition 5.3) arising from suitable associated two-variable adjunctions are exactly Lambert's Join and Residuate in vector form (equations 5.iv–5.v); (3) multi-head combination by lattice meet is justified because Mφ is a complete V-category, so arbitrary meets exist. Referenced in [tensor](core/tensor.md), [embed](legacy/embed.md).
### Smooth approximations

- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. *Mathematical Programming*, 103(1), 127–152.
  — bounded-gap approximation property of LogSumExp. Referenced in [activations](core/activations.md).

### Fixpoint theory

- Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its applications. *Pacific Journal of Mathematics*, 5(2), 285–309.
  — guarantees convergence of monotone operators on complete lattices. Referenced in [embed](legacy/embed.md).

### Fuzzy logical associative memory

- Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory. *Information Sciences*, 128, 91–103.
  — Theorem 1: Lambert's `_concept_fixpoint` (`O*`/`A∧` alternation with Gödel implication) converges to a formal concept in exactly two steps via idempotence of Galois adjunctions (Ore 1944). Theorem 2: stable points form the complete concept lattice. Theorem 6: constructive learning rule for R from labeled concepts. Lambert's algebra (min, Gödel implication) is Example 2 — one of three canonical cases the theorems directly cover. Referenced in [embed](legacy/embed.md), [attention](legacy/attention.md).

### Formal concept analysis

- Brito, A. M. et al. (2018). *Fuzzy Formal Concept Analysis.*
  — Definition 15: fuzzy formal context `⟨O, A, I_f⟩`; Definition 17: formal concept as simultaneous fixpoint of `O*` and `A∧`; Theorem 8: completeness of the concept lattice. Referenced in [embed](legacy/embed.md), [attention](legacy/attention.md), [explorer](legacy/explorer.md).
- Trnecka, M. & Vyjidacek, R. (2020). Revisiting the GreCon Algorithm for Boolean Matrix Factorization. *CLA 2020.*
  — coverage-based concept selection as matrix decomposition. Referenced in [embed](legacy/embed.md).
- Belohlavek, R. & Vychodil, V. (2007). Fuzzy concept lattices constrained by hedges. *Journal of Advanced Computational Intelligence and Intelligent Informatics*, 11(6), 536–545.
  — fuzzy FCA with graded membership. Referenced in [embed](legacy/embed.md).

### Associative memory and attention

- Krotov, D. & Hopfield, J. (2021). Large Associative Memory Problem in Neurobiology and Machine Learning. *ICLR 2021.*
  — §3.2 Model B: derivation of transformer attention as the fast-memory limit of dense associative memory; partial pattern initialisation and attractor convergence. Referenced in [attention](legacy/attention.md), [explorer](legacy/explorer.md).
- Ramsauer, H. et al. (2020, revised 2021). Hopfield Networks is All You Need. *arXiv:2008.02217.*
  — modern Hopfield networks and their connection to attention. Referenced in [embed](legacy/embed.md).

### Predictive coding and free energy

- Parr, T., Pezzulo, G. & Friston, K.J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
  — eq. 4.19: energy as sum of prediction errors; theoretical basis for the energy function in `FixpointIterator`. Referenced in [fixpoint](core/fixpoint.md).

### Tensor logic and temperature

- Domingos, P. (2025). *Tensor Logic: The Language of AI.* arXiv:2510.12269v3. https://arxiv.org/abs/2510.12269
  — Datalog rules as einsums; Join as tensor logic relational composition; temperature spectrum from deductive (T=0) to analogical (T>0); optimal T varies by data sparsity. Referenced in [tensor](core/tensor.md), [fixpoint](core/fixpoint.md).

