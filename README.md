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
Lambert's embedding is grounded in fuzzy formal concept analysis. A **formal concept** is a pair (extent, intent): the extent is the set of entities that share a collection of attributes; the intent is the set of attributes that defines them. Lambert selects representative concepts as embedding dimensions, building the embedding directly from the algebraic structure of the data [(Bělohlávek, Outrata & Trnecka, 2019)](http://belohlavek.inf.upol.cz/publications/BeOuTr-Fbmufciuee.pdf).

#### **Convergence:**
Rather than training via gradient descent and backpropagation, Lambert uses predictive coding [(Friston et al., 2010)](https://www.nature.com/articles/nrn2787): the system iterates to minimise free energy, updating beliefs until they stop changing. The stable points of this process are the formal concepts of the relation, whose set forms a complete lattice [(Bělohlávek, 2000)](https://www.sciencedirect.com/science/article/pii/S002002550000044X). As free energy falls, temperature falls with it, hardening fuzzy operations toward crisp logical outcomes.

#### **Attention:**
Lambert's attention mechanism performs retrieval over concept embeddings using max-min composition rather than dot-product arithmetic — converging on the concept that most strongly subsumes the query rather than the one most correlated with it [(Krotov & Hopfield, 2021)](https://arxiv.org/abs/2008.06996). Each head iterates a query to convergence, constrained by residuation so the result stays within what the embedding can support. Multi-head retrieval runs one head per relation and combines results via elementwise minimum — the lattice meet of the concepts each head returns.

#### **Provenance:**
Because every conclusion is a formal concept, provenance is a structural property of the lattice the computation produces — not added on top of it. The concept itself is the proof: its extent identifies which entities are implicated, its intent identifies why [(Green, Karvounarakis & Tannen, 2007)](https://dl.acm.org/doi/10.1145/1265530.1265535).

## Status

The core algebraic framework, embedding, attention, and lattice exploration components are implemented and functional. The system discovers multi-relational concepts invisible to standard similarity measures, and produces fully traceable provenance via the concept lattice.

**Active research directions:**

- Characterising the fixpoint set of multi-head attention formally — whether it constitutes a complete lattice and under what conditions
- Relation matrix updates from new relational evidence — incorporating genuinely novel facts and propagating their consequences through the lattice
- Connecting Lambert's architecture back to standard transformer-based frameworks — characterising what transformers approximate in max-min algebraic terms, and what is lost in that approximation
- Scaling to large medical and scientific knowledge graphs
