# Theoretical Synthesis

A unified account of the mathematical foundations of Project Lambert, synthesised from
`research/some_notes.txt` and a literature review of the six most relevant papers in
`/home/scanbot/research/`. Intended as a stable reference to orient new sessions.

---

## 1. The Research Problem

Project Lambert is a **differentiable relational query algebra** — a neurosymbolic system
where reasoning is expressed as typed paths through a small set of primitive operators over
two dual spaces. The core claim is that these operators, and their algebraic guarantees
(idempotence, convergence, provenance), arise necessarily from a pair of adjunctions — not
by design, but because adjoint pairs over two spaces universally generate this structure.

The practical goal: a system that can (a) traverse relational knowledge bases, (b) compose
multi-hop queries, (c) carry provenance transparently through computation, and (d) learn
the query program itself rather than requiring it to be pre-specified.

---

## 2. The Two Spaces and Four Primitive Operators

The system operates over two dual spaces:

- **E** — entity space (extents, objects, data rows)
- **C** — concept space (intents, attributes, schema columns)

Four primitive generators transport between them:

| Operator | Type | Semantic name | Direction |
|---|---|---|---|
| `sd` | E → C | propagate / image | join-like, forward |
| `pd` | C → E | support / preimage | join-like, backward |
| `pe` | E → C | abstract / intent | residuate-like, forward |
| `se` | C → E | realize / extent | residuate-like, backward |

These form **two adjoint pairs**:

```
sd ⊣ pd      (join/propagate adjunction)
pe ⊣ se      (residuate/abstract adjunction)
```

The adjoint laws constrain the system universally:

```
sd(x) ≤ c  ⟺  x ≤ pd(c)
pe(e) ≤ c  ⟺  e ≤ se(c)
```

Operators must alternate spaces (E→C→E→... or C→E→C→...). The type system enforces this;
invalid compositions (sd sd, pe pe) do not arise.

---

## 3. Canonical Composite Operators

Long paths collapse to a finite set of canonical blocks via idempotence. These are the
effective vocabulary of the system:

| Block | Path | Guaranteed idempotent? | Semantic role |
|---|---|---|---|
| **Hop** | `sd` | — (primitive) | single relational traversal |
| **Attend** | `se sd` | empirical | retrieve / reconstruct (Σ-like) |
| **Recall** | `pe pd` | empirical | concept tightening / closure (Π-like) |
| **Correct** | `pd sd` | yes (same-pair) | support filtering / constraint |

**Same-pair roundtrips** (algebraically guaranteed idempotent via closure/interior theory):

```
(pd sd)(pd sd) = pd sd      # closure on E
(sd pd)(sd pd) = sd pd      # projection on C
(se pe)(se pe) = se pe      # closure on C
(pe se)(pe se) = pe se      # projection on E
```

**Mixed roundtrips** (empirically closure-like, not algebraically guaranteed):

```
(se sd)(se sd) ≈ se sd      # Attend
(pe pd)(pe pd) ≈ pe pd      # Recall
```

The distinction matters: same-pair idempotence has a theorem behind it; mixed idempotence
is an observation. Do not overstate the algebraic guarantee for Attend and Recall.

---

## 4. Semiring Path Composition

Paths compose via a temperature-indexed semiring:

```
a ⊕_T b = T log(e^(a/T) + e^(b/T))    # aggregation (logsumexp)
a ⊗ b   = a + b                         # serial composition
```

The master composition law (factoring through witness y):

```
(left ∘ right)(x, z) = ⊕_T_y (left(x, y) ⊗ right(y, z))
                     = T log Σ_y exp((left(x, y) + right(y, z)) / T)
```

This is exactly the **semiring provenance evaluation** of Green et al. (2007):
- `⊗` (serial composition) corresponds to join / conjunction — both paths must be taken
- `⊕_T` (aggregation) corresponds to projection / union — alternative paths sum

At the temperature limit:

```
T → 0 :  ⊕_T → max     (tropical / Viterbi / Gödel semantics — one best witness)
T > 0 :  ⊕_T = logsumexp  (partition-function semantics — all witnesses blend)
```

This is one temperature-indexed family of semiring semantics, not two unrelated systems.

---

## 5. Stable States are Formal Concepts (Bělohlávek 2000)

**Source:** `fuzzy-logic-bidirectional-associative-mem.pdf` — Bělohlávek (2000),
*Information Sciences* 128, 91–103.

The FLBAM paper is the ground truth for the system's fixed-point dynamics. Its architecture
is exactly the two-space adjoint system over residuated lattices:

- Layer G (objects) = entity space E; Layer M (attributes) = concept space C
- Up-direction `↑`: `B(mⱼ) = ⋀{A(gᵢ) → Iᵢⱼ}` — this is `pe` (entity → concept via residuation)
- Down-direction `↓`: `A(gᵢ) = ⋀{B(mⱼ) → Iᵢⱼ}` — this is `pd` (concept → entity)
- The pair `(↑, ↓)` is a Galois connection

**Theorem 1 (Bělohlávek):** Convergence in exactly **2 steps** — one up (E→C), one down
(C→E). No oscillation possible.

**Theorem 2:** The set of all stable points forms a **complete lattice** (Tarski guarantee).

**Theorem 6 (learning rule):**
```
Iᵢⱼ = ⋁{ Aᵖ(gᵢ) ⊗ Bᵖ(mⱼ) }
```
The outer product / join structure underlying the weight matrices.

**Key interpretation:** Every stable state `(A, B)` is a **formal fuzzy concept** —
`A` is the fuzzy extent (which entities belong), `B` is the fuzzy intent (which attributes
hold). The `pe pd` Recall block is the fixpoint of this system. `_concept_fixpoint` in
Lambert computes exactly this.

---

## 6. Provenance as Semiring Annotation (Green et al. 2007)

**Source:** `provenance-semirings.pdf` — Green, Karvounarakis & Tannen (2007), PODS 2007.

Every relational operator has a dual provenance operator:

| Relational operator | Provenance operator | Meaning |
|---|---|---|
| join (⋈) | multiply (`·`) | both source tuples required |
| projection (π) | add (`+`) | multiple derivations collapse |
| union (∪) | add (`+`) | multiple sources contribute |
| selection (σ) | multiply with 0/1 | filter preserves/zeros provenance |

Provenance of a query result is a **polynomial in N[X]** over input tuple identifiers.
Each monomial = one derivation path; the polynomial = the complete witness structure.

**Proposition 3.5:** Semiring homomorphisms commute with all RA⁺ operators. This means:
compute provenance in the most general semiring (N[[X]]) and specialise via homomorphism
to extract: tuple identity (boolean), counting (natural numbers), confidence (tropical),
probability (probabilistic semiring).

**Direct implication for Lambert:** Every path through (se, sd, pe, pd) carries a provenance
polynomial. The witness score `w_t` in the composition loop is evaluating this polynomial
in a specific semiring. The temperature parameter selects the semiring instance (tropical
at T=0; partition-function at T>0).

---

## 7. Provenance Under Recursion (Dannert et al. 2021)

**Source:** `semiring-provenance-fizedpoint-logic.pdf` — Dannert, Grädel, Naaf & Tannen
(2021), CSL 2021.

Extends Green et al. to **fixpoint logic** — recursive and iterative queries. Key results:

- Requires **fully chain-complete** semiringes (not just ω-continuous) for general fixpoint
  iteration with negation
- **Universal semiring:** S^∞[X] (generalized absorptive polynomials) is the universal
  object; all other semiringes specialise via homomorphism
- **Fundamental compositionality property:** `h(π[φ]) = (h ∘ π)[φ]` — provenance is
  preserved under any continuous homomorphism across fixpoint iterations
- **Theorem 23 (Game-Theoretic):** Provenance of a fixpoint formula = sum over all
  evaluation strategies: `π[φ] = ⊔_{S ∈ Strat(φ)} τ[S]`
- Fixpoint iterations stabilise at closure ordinal ω — consistent with Bělohlávek's
  2-step convergence

**Direct implication for Lambert:** The composition loop `S_{t+1} = F(S_t)` with halt
condition is a fixpoint computation in LFP. Provenance is preserved through each iteration
by the compositionality theorem. The witness state `w_t` is the current provenance
polynomial evaluation — it tracks which evaluation strategies have contributed to the
current query state. This justifies including witness strength in the scoring objective
without breaking the algebraic provenance guarantees.

---

## 8. The Σ⊣Δ⊣Π Triple and Its Relation to the Four-Operator Diamond

**Sources:** `algebraic-data-integration.pdf` — Schultz & Wisnesky (2025, based on 2017
journal paper); `algebraicc-databases.pdf` — Schultz, Spivak, Vasilakopoulou & Wisnesky
(2017), *Theory and Applications of Categories* 32(16), 547–619.

### The triple

For any schema mapping F: S → T, three adjoint data migration functors arise:

```
Σ_F ⊣ Δ_F ⊣ Π_F
```

| Functor | Direction | What it does |
|---|---|---|
| Σ_F | S-Inst → T-Inst | push forward (union / aggregation) |
| Δ_F | T-Inst → S-Inst | reindex / restrict (central transport) |
| Π_F | S-Inst → T-Inst | union-then-merge / join (right push) |

The query evaluation factorizations are:

```
eval(Q)   = Δ ∘ Π      (forward query evaluation)
coeval(Q) = Δ ∘ Σ      (backward / co-evaluation)
```

And these form their own adjunction: **coeval(Q) ⊣ eval(Q)**.

### Resolution of the "hidden Δ" question

The notes raised: is the four-operator diamond secretly a Σ⊣Δ⊣Π triple with a hidden
central Δ?

**Answer (from Schultz & Wisnesky):** The Σ⊣Δ⊣Π triple is the *fundamental* structure.
The diamond (two adjoint pairs sd⊣pd and pe⊣se) is a **derived factorization** — it
arises from decomposing eval/coeval through intermediate schemas. Δ is present in both
eval and coeval as the central mediating operator.

Correspondences:
- `sd` ↔ Σ_F (join-like forward propagation)
- `pe` ↔ Π_F (residuate-like constraint closure)
- Both `pd` and `se` ↔ aspects of Δ_F (the central transport, appearing twice)

**Practical consequence:** The four operators are not four independent primitives. They
correspond to compositions of three underlying categorical operators where the central
one (Δ, relational transport) appears twice — once in each adjoint pair. The diamond
structure is the view from the eval/coeval factorization. Both pictures are correct at
their respective levels of abstraction.

### Databases as categories

From the Algebraic Databases paper (Schultz et al. 2017):
- Schemas = categories (entities are nodes, foreign keys are edges, path equalities are constraints)
- Instances = product-preserving functors from schema to Set
- Queries = bimodules between schema categories
- Path composition in schema = relational query composition
- Path equivalences = query rewrite rules (Knuth-Bendix completion = query optimization)
- Any bimodule M decomposes into entity component (E) and type/concept component (C)
  — exactly the two spaces in Lambert

Query containment, normalization, and rewriting are all consequences of categorical path
equivalence. The operator reduction rules in Lambert (e.g., `(pe pd)² = pe pd`) are
instances of this general principle.

---

## 9. The Iterative Composition Loop

The notes develop a dynamic framing of reasoning as **iterative relational stabilization**
rather than static path evaluation. Explicit state:

```python
S_t = {
    "q": q_t,        # current activation / query state
    "p": p_t,        # current path / program
    "e": e_t,        # prediction / consistency error
    "w": w_t,        # witnesses / provenance polynomial
    "s": score_t,    # utility of current path
    "halt": h_t,     # stop flag
}
```

Core recursion:

```
p_{t+1} = p_t ⊕ π(S_t)                    # path extension via learned policy
q_{t+1} = Exec(p_{t+1}, q_0)              # execution
w_{t+1} = Trace(p_{t+1}, q_0)             # semiring provenance evaluation
e_{t+1} = Err(q_{t+1}, target, constraints)
halt     = ||e_t|| < ε  or  S_{t+1} = S_t
```

Scoring objective:

```
score_t = task_fit(q̂_t)
        - λ_err  * inconsistency(e_t)
        - λ_len  * complexity(p_t)
        + λ_wit  * witness_strength(w_t)
        + λ_fix  * stability(q̂_t)
```

The error signal is multidimensional:

```
e_t = {
    "task":       target - q̂_t,
    "closure":    q̂_t - Closure(q̂_t),
    "support":    unsupported_mass(q̂_t, w_t),
    "redundancy": redundant_steps(p_t),
}
```

This loop is a fixpoint computation in LFP (Dannert et al.) — provenance flows through it
by the compositionality theorem. The closure error term directly exploits Bělohlávek
convergence: if `q̂_t` is not a formal concept, `Closure(q̂_t) ≠ q̂_t` and the error is
non-zero.

---

## 10. Positioning: What Makes This System Distinct

**Source:** `integrating-symbolic-reasoning-into-neural-networks.pdf` — Hamilton, Vance &
Wright (2026), *Frontiers in AI Research* 3(1).

Current neurosymbolic systems (NSLP, TensorLog, Logic Tensor Networks) are
**structure-fixed, weight-learned**: the reasoning program is pre-specified (Horn clauses,
Datalog rules), and the network learns the weights over that fixed structure. This causes
exponential grounding explosion as predicate count grows and requires a pre-defined domain
ontology.

Lambert is **operator-fixed, composition-learned**: the four primitive operators are fixed
(with algebraic guarantees), but the composition — the reasoning program itself — is
discovered. No pre-defined ontology; predicates emerge from the concept lattice. The
grounding problem does not arise because computation is over embeddings, not explicit
predicate groundings.

The structural bias (adjoint pairs, formal concepts, convergence guarantees) provides
strong inductive bias without requiring human-authored rules.

---

## 11. Open Questions (from `some_notes.txt`)

1. **Audit of mixed-path idempotence:** Build a rewrite table of all typed roundtrips with
   columns: type (E→E or C→C), extensive or contractive, closure-like or projection-like,
   empirically idempotent, witness-preserving, temperature sensitivity. This separates
   algebraically guaranteed from empirically observed claims.

2. **Identification of sd and pe:** Can `sd` and `pe` be identified up to transpose, dual,
   or temperature limit? Can `pd` and `se` similarly be identified? If yes, the diamond
   collapses to a triple; if no, the double-adjunction is genuinely fundamental.

3. **Provenance semiring selection:** Which semiring instance best serves the witness score
   during training? Tropical (T=0, best-path) vs. partition-function (T>0, all-paths blend)
   have different gradient properties and different expressivity.

4. **Path normalization in practice:** The algebraic reduction rules predict that many paths
   are equivalent. Does the learned composition policy discover these normal forms, or does
   it find redundant paths? Measuring path redundancy via the provenance polynomial would
   answer this.

5. **Δ identification:** The notes conclude the diamond is the genuine structure (not a
   hidden triple), but the Schultz/Spivak papers show Δ is the central mediating functor
   in both eval and coeval. A concrete experiment: can a single learned operator play the
   role of Δ and recover `sd` and `pe` as `Δ ∘ Σ` and `Δ ∘ Π` respectively?

---

## 12. Bibliography (papers in `/home/scanbot/research/` most relevant to this document)

| Key | Citation |
|---|---|
| belohlavek2000 | Bělohlávek, R. (2000). Fuzzy logical bidirectional associative memory. *Information Sciences*, 128, 91–103. |
| green2007 | Green, T.J., Karvounarakis, G. & Tannen, V. (2007). Provenance semirings. *PODS 2007*. |
| dannert2021 | Dannert, K.M., Grädel, E., Naaf, M. & Tannen, V. (2021). Semiring provenance for fixed-point logic. *CSL 2021*, LIPIcs 17:1–17:22. |
| schultz2017 | Schultz, P., Spivak, D.I., Vasilakopoulou, C. & Wisnesky, R. (2017). Algebraic databases. *Theory and Applications of Categories*, 32(16), 547–619. |
| schultz2025 | Schultz, P. & Wisnesky, R. (2025). Algebraic data integration. arXiv:1502.05947 (expanded from JFP 2017). |
| tarski1955 | Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its applications. *Pacific Journal of Mathematics*, 5(2), 285–309. |
| dusell2023 | DuSell, B. (2023). Nondeterministic stacks in neural networks. PhD dissertation. (arXiv:2010.04674, 2109.01982, 2210.01343) |
| lang1974 | Lang, B. (1974). Deterministic techniques for efficient non-deterministic parsers. *ICALP 1974*, LNCS 14, 255–269. |

See `research/bibliography.md` for the full project bibliography.

---

## 13. Multi-Relational Inference and the Composer Architecture

### The two Kan paths

`theory.md` identifies the central gap: Lambert currently implements only the **right Kan path**
(universal semantics, `Residuate`-based, `pe pd` closure). The **left Kan path** (existential
semantics, `Join`-based, `sd^k`) is present as `Hop` but not integrated under the same
fixpoint discipline.

The two paths are dual:

```
Left Kan  (existential): q₁ = Join(q, EmbR)       — sd, generative, "reachable from q"
Right Kan (universal):   B  = Residuate(EmbR, q)  — pe, restrictive, "consistent with q"
```

Multi-relational hops (`sd^k`) chain the left Kan step across different Tucker-core matrices:

```
q₁ = Join(q,  EmbR₁)    # hop through relation 1
q₂ = Join(q₁, EmbR₂)    # hop through relation 2
A  = Join(q₂, emb.T)    # project back to entity space
```

The type checker in `PathCoder` enforces E↔C alternation for **single-relation path specs**.
Multi-relational hops bypass this deliberately — each `Hop` call is a standalone C→C operation
on a Tucker-core matrix, outside the type system by design. This is architecturally correct.

### Temperature, fixpoint, and fold_empirical

`FixpointIterator` (fixpoint.py) is the Y combinator for this system: `Y F = F (Y F)`, halt
when `state_{t+1} = state_t`. Critically, **temperature is derived from energy**, not passed
externally. As the iterator runs, energy falls and temperature falls with it toward zero.

At `T = 0`, the semiring becomes tropical (max replaces logsumexp) and addition becomes
idempotent: `A ⊕_0 A = A`. This means the mixed-pair roundtrips (Attend `se sd`, Recall
`pe pd`) become exactly idempotent — not empirically, but as a theorem of the semiring.

Therefore `fold_empirical ⟺ temp == 0` is a rule that always holds, not an opt-in flag.
The `Composer` module should read `fold_empirical` from the iterator's current temperature,
not accept it as a caller argument.

### The learning gap

`Learner.learn()` implements the correct algebraic learning rule (Bělohlávek Theorem 6 /
Sussner & Valle 2006):

```
R_new = max(R_old, Residuate(Y, X))
```

This is the Hebbian construction that stores (entity, attribute) pattern pairs as stable
attractors. It is not gradient descent — it is the adjoint-based weight update that directly
encodes formal concepts into the relation matrix. The problem is that `learn()` updates `R`
but does not refresh `emb` and `EmbR`, so subsequent queries do not see the learned patterns.
A `fit()` method that chains `learn()` → `ConceptEmbed()` is the plumbing fix needed.

### Lang's Algorithm and Nondeterministic Stack RNNs (DuSell 2023)

**Source:** `nondeterministic-stacks-neural-networks.pdf` — DuSell, B. (2023), PhD dissertation
(advisor: Chiang). Building on DuSell & Chiang arXiv:2010.04674, 2109.01982, 2210.01343.

---

#### What Lang's algorithm solves

Lang's algorithm (Lang, 1974; reformulated by Butoi et al. 2022) solves the **weighted PDA
recognition problem**: given a nondeterministic weighted pushdown automaton (WPDA) and input
string w = w₁···wₙ, compute the total weight of *all* accepting runs in polynomial time —
without enumerating them.

The WPDA is a 7-tuple (Q, Σ, Γ, δ, q₀, F) where:
- Q = finite set of automaton states
- Γ = finite stack alphabet (with bottom marker ⊥)
- δ: Q × Γ × (Σ ∪ {ε}) × Q × Γ* → ℝ≥0 — nonneg transition weight function
- Weight of run π = τ₁,...,τₘ: `ψ(π) = ∏ᵢ Λᵢ[τᵢ]`

The **restricted normal form** (Definition 17) limits transitions to stack-size changes of ≤1:
```
Push:    q, x →ᵃ r, xy      (push y on top of x)
Replace: q, x →ᵃ r, y       (replace x with y)
Pop:     q, x →ᵃ r, ε       (pop x)
```
Any PDA can be converted to this restricted form without loss of expressive power (Proposition 1).
The nondeterministic variant is strictly more powerful than the deterministic; real-time
nondeterministic PDAs are equivalent to full nondeterministic PDAs (Greibach 1965).

---

#### The DP recurrence

Lang's algorithm computes two tensors via mutual recursion:

**Inner weights γ** — weight of all partial runs from configuration (i,q,x) to (t,r,y)
that never expose x to the top between i and t (Eq. 4.6):

```
γ[i→t][q,x→r,y] =
    𝟙[i=t-1] · Λₜ[q,x→r,xy]                              (push)
  + Σ_{s,z} γ[i→t-1][q,x→s,z] · Λₜ[s,z→r,y]             (replace)
  + Σ_{k=i+1}^{t-1} Σ_{u,s,z} γ[i→k][q,x→u,y]            (pop)
                              · γ[k→t-1][u,y→s,z]
                              · Λₜ[s,z→r,ε]
```

**Forward weights α** — weight of all runs from the initial config to (t,r,y) (Eqs. 4.7–4.8):
```
α[0][r,y]  = 𝟙[r = q₀ ∧ y = ⊥]
α[t][r,y]  = Σ_{i=0}^{t-1} Σ_{q,x} α[i][q,x] · γ[i→t][q,x→r,y]
```

**Normalized stack reading** — marginal distribution over top stack symbol at time t (Eq. 4.9):
```
rₜ[y] = (Σᵣ α[t][r,y]) / (Σ_{y'} Σᵣ α[t][r,y'])
```

Complexity: **O(|Q|²|Γ|³n³)** time, **O(|Q|²|Γ|²n²)** space. The exponential blowup of
nondeterministic configurations is avoided because stacks of height k only need to store
the top symbol and a pointer to the height-(k-1) stack — the DAG of configurations
implicitly encodes exponentially many paths.

All computations run in **log-space** for numerical stability (log-semiring: logsumexp
replaces multiplication, addition replaces log-multiplication). Final normalization is
deferred to Eq. 4.9 — intermediate products are never renormalized, analogous to running
a partition function without committing to any single configuration.

---

#### The NS-RNN controller

The LSTM controller at timestep t:
1. Reads: previous hidden state h_{t-1}, input x_t, stack reading r_{t-1}
2. Computes: (h_t, c_t) = LSTM((h_{t-1}, c_{t-1}), [x_t; r_{t-1}])
3. Outputs action tensor: **A_t = softmax(W_a h_t + b_a)** — the transition weight tensor Λₜ

The key variant (RNS-RNN, Chapter 6) uses **unnormalized exponential weights**
`A_t = exp(W_a h_t + b_a)` rather than softmax, allowing individual transitions to amplify
shared runs. This is essential for long-distance dependencies (e.g., waw^R with padding).

---

#### Empirical findings (relevant to Lambert)

Evaluated on formal language tasks (w#w^R, ww^R, Dyck, Hardest CFL) and Penn Treebank:

- **NS-RNN wins on nondeterministic CFLs**: lowest cross-entropy on ww^R and Hardest CFL
  vs. LSTM and deterministic stack baselines
- **Generalization failure beyond training length**: struggles on strings >80 symbols (wa^p w^R)
- **Discrete stack symbol bottleneck**: small Γ limits expressiveness; `O(|Γ|³)` prevents
  scaling |Γ| further
- **No benefit on natural language** (Penn Treebank): superposition stack (continuous vectors)
  outperforms on perplexity; NS-RNN achieves best syntactic generalization score (0.471 SG)
- **Speed**: ~20× slower than LSTM; ~1000 s/epoch vs. 51 s/epoch

The architecture is **syntax-driven and sequential**; Lambert is content-addressable and
lattice-driven. The architecture is not transferable, but the algorithm is informative.

---

#### Structural connections to Lambert

| NS-RNN / Lang | Lambert equivalent |
|---|---|
| Transition tensor Λᵢ | Relation matrix R (or Tucker core EmbR) |
| Run weight `ψ(π) = ∏ᵢ Λᵢ[τᵢ]` | Serial composition `⊗` (addition in log-space) |
| Marginal over configs (Eq. 4.9, logsumexp) | `⊕_T` aggregation over witness chains |
| α forward weights | Path weights accumulated through `sd^k` multi-hop chain |
| γ inner weights | Intermediate evidence within a single Hop block |
| Configuration (i, q, x) | Lattice element / activation at step i |
| Deferred renormalization | Temperature schedule: full fuzzy values until T→0 |
| Nondeterministic stack (all configs weighted) | Soft entity activations across E simultaneously |
| Stack WFA / chart DAG | Concept lattice encoding all reachable states |

The γ recurrence is structurally a **semiring path problem** — the same algebraic object as
Lambert's `⊕_T`-composition loop. The three cases (push, replace, pop) correspond to three
structural moves in a multi-relational traversal: extend the path, rewrite the current step,
or collapse back to a previous anchoring point.

**Deferred renormalization** is the most directly useful insight: Lang's algorithm never
normalizes intermediate sums — it runs the full forward pass in unnormalized log-space and
normalizes only at the final marginalization step. This is algorithmically identical to
Lambert's energy-driven temperature schedule: the `FixpointIterator` maintains full soft
activations throughout and converges toward hard assignments only as T→0. The connection
validates the design choice.

**Inside/outside interpretation**: α is a forward (inside) weight; the denominator in Eq. 4.9
is a partition function. The stack WFA is structurally a chart parser (cf. Tomita 1987 GLR
parsing); Lang's algorithm is CKY with a stack. This means multi-hop inference in Lambert
can be viewed as **chart-based relational parsing** over the concept lattice, with
`⊕_T`-aggregation playing the role of the chart combination rule.

---

#### Open questions arising from this connection

**Q1 (max-min semiring):** Is there a version of Lang's γ recurrence operating over the
max-min semiring — `⊗ = min`, `⊕ = max` — whose fixpoint coincides with `Closure` in
`tensor.py`? If yes, multi-hop inference becomes a provenance-tracked chart parse over the
concept lattice, and convergence follows from the lattice's completeness (Bělohlávek Theorem 2).

**Q2 (unnormalized weights → Lambert):** The RNS-RNN improvement (unnormalized exp weights
instead of softmax) mirrors Lambert's open question about whether `fold_empirical` should be
driven by temperature rather than being a caller flag. In both cases the question is: *when*
should the system commit to normalization? Lang's algorithm answers: at query time only.

**Q3 (replace as rewrite):** The "replace" case in γ (q, x →ᵃ r, y: pop x, push y atomically)
is structurally a **term rewrite** — exactly what Lambert's term rewriting system (added in
commit e413eaa) implements. Is there a formal correspondence between the replace transition
and the rewrite rules? If yes, the rewrite system gains a DP semantics: the cost of applying
a rewrite rule is its weight in the semiring, and the optimal rewrite sequence is the Viterbi
path through the γ tensor.

**Q4 (multi-relational γ):** Each Tucker-core EmbRₖ in Lambert is one relation. A
multi-relational version of γ would index the inner tensor by relation: `γ[i→t][k]` where k
indexes which EmbRₖ was traversed at each step. The three cases (push, replace, pop) would
map to: extend through relation k, rewrite via a relation equivalence, return to an earlier
anchoring concept. This would give the concrete multi-relational inference algorithm that is
currently missing from Lambert.

---

## 14. Path-Selection Policy and Learning Rule

### The policy sketch (`test_path_policy.ipynb`)

The path-selection policy drives `PathCoder` using the three Lang rules (push/replace/pop)
with `FixpointIterator` as the outer loop. Key design points:

- **State**: a 1-element float array `[mock_energy(current_spec)]`. The spec string is
  carried in a mutable closure cell — the iterator state is purely numeric.
- **Push**: enumerate the two valid leg extensions from the current output space via
  `LEG_TYPE`; aggregate via `logsumexp(-energy/T)` (⊕_T semiring). Branching factor is
  always exactly 2.
- **Replace**: delegate entirely to `coder.compile(spec, fold_empirical=...)`. No manual
  pair-checking. This is the entire replace rule.
- **Pop**: backtrack one leg when the best push candidate has higher energy than current.
  `FixpointIterator` sees `||new-old||² = 0` when stuck → converge.
- **Temperature**: driven by `FixpointIterator._update_temp()` (Boltzmann, fixpoint.py:97–115).
  No manual annealing schedule.

The current mock energy table is a placeholder. In production the energy comes from
`FixpointIterator.energy` after actually running the compiled path on data.

### Why the paths are not the weights

The reasoning paths are programs compiled from R, not weights themselves. R is the weight
matrix. A path that converges at low energy is evidence that the (extent, intent) structure
already encoded in R supports that inference. The paths index into R; R is what is learned.

### The learning policy

When a path converges (outer `FixpointIterator` settles, `fp.energy <= threshold`), the
system has a pair: input state `x` and converged output `z`. This is the `(X, Y)` pair for
`Learner.learn()`:

```
delta_R = Residuate(Y=z, X=x,  temp=0)   # Belohlavek eq. 2
R_new   = max(R_old, delta_R)
```

The selection criterion is: **only call `learn()` on convergent low-energy paths**, not on
every path explored. High-energy paths are evidence of bad structure; reinforcing them
would corrupt R.

This is the third Friston level (parametric error, ε_θ). The iterator tracks dynamic error
(ε_x) and sensory error (ε_y); parametric update happens one level above, after convergence.
`FixpointIterator` does not and should not drive the parametric update — it fires after the
iterator returns.

### Open question: path selection criterion

The current sketch picks paths purely from a pre-defined energy dict. The real signal
needs to be derived from `FixpointIterator.energy` after running the compiled path on actual
data. The question is how to map from `fp.energy` back to a score over candidate specs —
the energy surface over path space is currently unobserved.

---

## 15. The Engine DSL: From Theory to Executable Specification

The theoretical synthesis in the preceding sections identifies the mathematical structure
underlying Project Lambert: adjoint pairs generating four primitive operators, semiring
composition over typed paths, fixpoint convergence to formal concepts, and provenance as
a semiring polynomial threading through every inference step. The engine DSL in `engine/`
makes that structure **executable** — every concept described in the synthesis is expressed
as a DSL declaration rather than hand-coded Python.

### Correspondences between synthesis and DSL

The four operators `sd`, `pd`, `pe`, `se` are declared as `MorphismDecl` entries, each
carrying explicit `src_sort` and `tgt_sort` annotations. The type system that enforces
E↔C alternation (section 2) is not a runtime check — it is a constraint expressed at
declaration time and checked by the compiler when building a path. Invalid compositions
(`sd sd`, `pe pe`) do not arise because the compiler refuses to link morphisms whose sort
boundaries do not match.

Semiring path composition (section 4) is expressed as `PathDecl` — a named, ordered chain
of morphisms. The compiler translates each `PathDecl` into a sequence of tensor operations
using the active `SemiringDecl` (which supplies the `⊕` and `⊗` implementations). Any
semiring can be declared: `SemiringDecl` takes a `contract` (the algebraic laws) and a
`compiler` (the tensor-level implementation). The logsumexp family from section 4 is one
instance; the max-min (tropical, Gödel) semiring is another; both are declared the same
way.

The Σ⊣Δ⊣Π adjoint triple (section 8) maps to `ArchDecl` with two dual sides:

- The **algebra** side (Σ direction, forward/generative) declares how output is assembled
  from parts — morphism compositions derived automatically from `PathDecl` entries via the
  `morphisms = path_name` shorthand on `CaseDecl`.
- The **coalgebra** side (Π direction, universal/restrictive) declares the dual streaming
  view: how input is decomposed and routed through the same operators in the other direction.

`CaseDecl` with `morphisms = path_name` is the main mechanism for eliminating Python cells:
the compiler auto-derives the algebra cell from the named path composition, so the user
states *what* the transformation is categorically, not *how* to implement it imperatively.

Fixpoint iteration (sections 5, 7, 9) is handled entirely by the `Functor`/`Interpreter`
runtime. The DSL author declares the algebra and coalgebra shapes; the runtime executes the
fixpoint loop, tracks convergence, and manages state — exactly the separation that
Bělohlávek's two-step theorem (section 5) demands. No fixpoint plumbing appears in the
declarations.

Provenance is structural, not annotated. The path through the functor *is* the provenance:
every `PathDecl` composition is a semiring computation whose witness polynomial is
implicitly maintained by the logsumexp aggregation. There are no `@provenance` tags or
metadata labels — the algebraic structure from sections 6 and 7 is enforced by the type
and semiring machinery, not by decoration.

### What the DSL adds beyond the theoretical synthesis

The synthesis is a mathematical account of one class of semirings (the temperature-indexed
logsumexp family). The DSL generalises this:

- **Arbitrary semirings**: any `(⊕, ⊗, 0, 1)` satisfying the declared contract can be
  registered via `SemiringDecl`. The architecture declarations are semiring-agnostic; the
  semiring is a runtime parameter.
- **Fan combinators**: `FanDecl` declares a set of parallel computation paths that run
  simultaneously and whose outputs are collected into a named bundle. This expresses the
  multi-headed attention pattern (section 13's multi-relational hops) directly in the DSL
  without branching the architecture graph by hand.
- **Augment combinators**: the `[fan_name]` syntax inside a `CaseDecl` runs a fan and
  merges its dict output into the current `y` payload, leaving `x` unchanged. This solved
  the bundle assembly problem (attention key/query/value projection) that previously required
  a Python escape hatch.
- **Architecture composition**: `ArchDecl` makes the algebra and coalgebra dual views
  first-class. The same morphism declarations are interpreted in both directions by the
  compiler, enforcing the adjoint relationship structurally rather than by convention.

### Practical consequence

A GPT-2 pre-norm transformer has been expressed entirely through the DSL: all algebra
cells are derived automatically from morphism compositions declared in `engine/arch.py`.
The `[fan_name]` augment combinator eliminated the last algebra-side Python cell (attention
bundle assembly). The only remaining Python cell is the coalgebra streaming cell
(`stream_cell`), which handles KV cache concatenation and the per-layer loop — a stateful
computation that requires imperative sequencing not yet expressible in the DSL's declarative
layer. That remaining gap marks the current boundary of the DSL's expressive reach.
