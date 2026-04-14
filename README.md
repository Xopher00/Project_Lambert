# Project Lambert
## Background
The use of generative AI models across different industries and businesses has become commonplace in the last few years. However, these models lack something crucial: provenance. Provenance is the ability to trace data and facts reliably to a source and verify them. LLMs do not provide provenance or transparency as a feature at all. Instead, they are treated like black boxes. AI providers do not elaborate on how the models are trained or built, and act like it is mysterious or even surprising when a model reaches a conclusion deemed "problematic".

This lack of transparency can be deadly. LLMs are prone to hallucinations — something that cannot be mitigated no matter how many guardrails you try to add. It is simply a consequence of the mathematics behind how these models operate. Models are also more likely to make mistakes when handling data that is sparsely represented during training. Under-represented groups of people, including women, LGBT individuals, and ethnic minorities, are therefore more likely to be negatively impacted by these hallucinations.

Every solution to this problem so far has been inadequate — something slapped on after the fact. Prompt and context engineering will never be enough. In fact, this problem may be intractable no matter what kind of model you are using. This has been a challenge in data science since its inception.

However, it may be possible to make these models transparent and auditable. By understanding the mathematics behind how models work, we can build a new framework from the ground up and incorporate provenance at the deepest level. This is what Project Lambert is trying to achieve.

## The Math
This project was initially inspired by the work of mathematician Eric Hehner. Hehner recognized that arithmetic and logic obey the same mathematical laws, and developed a notation system that reflects this and combines them into a single, [Unified Algebra](https://www.cs.utoronto.ca/~hehner/UA.pdf). The idea that math and logic are two sides of the same coin means we can theoretically build an AI not based on statistics and probability, but on logic itself. By incorporating logic at the model's deepest layer, we can make it easier to understand on an intuitive and interpretable level how a model reaches a decision or processes data.

We also take inspiration from [Tensor Logic](https://arxiv.org/pdf/2510.12269), a framework proposed by Pedro Domingos in which every AI operation reduces to the same core equation — Einstein summation over a tensor product:

$$R[x,z] = \sum_y A[x,y] \cdot B[y,z]$$

Lambert replaces the arithmetic operations with fuzzy logic — max and min — giving:

$$R[x,z] = \bigvee_y \bigl(A[x,y] \wedge B[y,z]\bigr)$$

This makes the model's reasoning semantically transparent and directly interpretable.

The deeper goal — central to Domingos' vision — is a genuinely neurosymbolic AI: one where the neural and symbolic aspects are not separate systems bolted together, but the same computation expressed at different levels of abstraction. In Lambert, learning, inference, and logical reasoning all reduce to the same relational operations. There is no symbolic layer on top of a neural layer; the logic is the network.

## Mathematical Foundations

#### **Fuzzy relational composition:**
At its core, Lambert reasons by asking: "given that X relates to Y, and Y relates to Z, how strongly does X relate to Z?" This is relational composition, implemented using fuzzy logic — min and max — rather than linear algebra matrix multiplication. The core inference step is:

$$R[x,z] = \bigvee_y \bigl(A[x,y] \wedge B[y,z]\bigr)$$

For each output pair (x, z), this finds the strongest chain of evidence through all possible intermediaries y, where chain strength is the weakest link [(Zadeh, 1965)](https://www.sciencedirect.com/science/article/pii/S001999586590241X).

Its adjoint operation is residuation — the inverse of composition, used for querying and concept closure:

$$b_j = \bigwedge_i \bigl(a_i \to R_{ij}\bigr)$$

where → is Gödel implication: `a → b = 1 if a ≤ b, else b`. Residuation finds the tightest set of attributes consistent with a given set of entities, and vice versa — the two directions of relational inference [(Sanchez, 1976)](https://www.sciencedirect.com/science/article/pii/S0019995876904460).

#### **Smoothing:**
Pure min and max are not differentiable. SmoothMin and SmoothMax are implemented via LogSumExp, with a temperature parameter controlling how close the approximation is to the exact operations. At low temperature the system reasons crisply; at higher temperature reasoning becomes softer and more analogical. The system operates anywhere on this spectrum without changing the underlying framework [(Nesterov, 2005)](https://link.springer.com/article/10.1007/s10107-004-0552-5).

#### **Formal concept analysis and concept embeddings:**
Lambert's embedding is grounded in fuzzy formal concept analysis. A **formal concept** is a pair (extent, intent): the extent is the set of entities that share a collection of attributes; the intent is the set of attributes that defines them. Lambert selects representative concepts as embedding dimensions, building the embedding directly from the algebraic structure of the data [(Bělohlávek et. al., 2019)](http://belohlavek.inf.upol.cz/publications/BeOuTr-Fbmufciuee.pdf).

#### **Convergence:**
Rather than training via gradient descent and backpropagation, Lambert uses predictive coding [(Friston, 2010)](https://www.nature.com/articles/nrn2787): the system iterates to minimise free energy, updating beliefs until they stop changing. The stable points of this process are the formal concepts of the relation, whose set forms a complete lattice [(Bělohlávek, 2000)](https://www.sciencedirect.com/science/article/pii/S002002550000044X). As free energy falls, temperature falls with it, hardening fuzzy operations toward crisp logical outcomes.

#### **Attention:**
Lambert's attention mechanism performs retrieval over concept embeddings using max-min composition rather than dot-product arithmetic — converging on the concept that most strongly subsumes the query rather than the one most correlated with it [(Krotov & Hopfield, 2021)](https://arxiv.org/abs/2008.06996). Each head iterates a query to convergence, constrained by residuation so the result stays within what the embedding can support. Multi-head retrieval runs one head per relation and combines results via elementwise minimum — the lattice meet of the concepts each head returns.

#### **Provenance:**
Because every conclusion is a formal concept, provenance is a structural property of the lattice the computation produces — not added on top of it. The concept itself is the proof: its extent identifies which entities are implicated, its intent identifies why [(Green et. al., 2007)](https://dl.acm.org/doi/10.1145/1265530.1265535).

## Project Structure

The codebase is organised as a layered stack. Each layer inherits from the one below.

```
core/
  algebra.py      — base constants (Top, Bottom), Max, Implies, and arithmetic helpers
  activations.py  — temperature-controlled smooth operators (LogSumExp, SmoothMax, SmoothMin, SoftMax)
  fixpoint.py     — FixpointIterator: iterates any operator to convergence with energy-derived temperature annealing
  tensor.py       — relational operations (Join, Residuate, Closure) built on top of Activations

engine/                      — active development: architecture DSL and compiler
  __init__.py     — public surface (Arch, Decl, compile, run)
  arch.py         — Arch declaration: named cases, fan-outs, and augment combinators
  compiler.py     — compiles Arch + Decl into an executable functor tree
  decl.py         — sort and morphism declarations (SortDecl, MorphDecl)
  functor.py      — algebra / coalgebra functor types; interpreter loop
  parser.py       — S-expression parser for DSL source strings

streamlined/
  __init__.py
  composer.py     — high-level composition helpers
  relational_einsum.py — einsum-style relational ops

legacy/
  lattice/
    embed.py      — concept embedding: selects representative formal concepts from a relation matrix
    attention.py  — single-head and multi-head retrieval over concept embeddings via fixpoint iteration
    explorer.py   — CategoryExplorer: systematic discovery and closure of the full concept lattice
  model.py        — Lambert: top-level pipeline (chunking, embedding, exploration, feature mapping)
  query.py        — Query: high-level interface for entity and feature retrieval with provenance
  lattice.py      — earlier Concept/Lattice utilities, superseded by lattice/
  language.py     — LLM-based category labeller (not currently wired into the pipeline)
  provenance/     — retired witness-based proof tree implementation
```

The `tests/` directory contains Jupyter notebooks covering join operations, attention, embeddings, and domain-specific experiments (knowledge graphs, PyPI dependencies, countries). Research notes are in `research/`.

## Status

Active development is focused on the **engine DSL** (`engine/`): a domain-specific language for expressing arbitrary AI architectures over semirings, compiled to a functor/coalgebra tree and interpreted by a fixpoint loop. The DSL can express relational composition, concept embedding, multi-head attention, and KV-cache streaming as declarative algebra and coalgebra cases, eliminating hand-written Python cells for most architecture patterns.

The original lattice pipeline — relational algebra, concept embedding, multi-head attention, lattice exploration, and the `Lambert`/`Query` interface — is preserved in `legacy/` as a reference implementation. It is functional and documents the design that the engine DSL is intended to generalise.

**Known gaps (engine DSL):**

- Final layer norm: the algebra path uses identity gamma/beta; the coalgebra `stream_cell` uses learned parameters. They agree only when parameters are initialised to identity. Fix requires a `ln_final` binary morphism passing final LN params as the case payload.
- `stream_cell` is the last remaining hand-written Python cell (coalgebra). It handles token embedding, per-layer KV cache iteration, final LN, and unembed. The per-layer loop is the blocking case for full DSL elimination.
- Structured sorts (`SortDecl` with named fields) are parsed and compiled but no compilation phase consumes the field info yet.

**Active research directions:**

- Eliminating `stream_cell` via a DSL construct for stateful iteration over layers with persistent KV cache
- Scaling the engine DSL to express multi-hop inference chains across relation types (`EmbR` Tucker core)
- Connecting Lambert's max-min algebra to standard transformer arithmetic — characterising what transformers approximate in max-min terms and what is lost
- Scaling to large medical and scientific knowledge graphs
