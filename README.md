# Project Lambert
## Background
The use of generative AI models across different industries and businesses has become commonplace in the last few years. However, these models lack something crucial: provenance. Provenance is the ability to trace data and facts reliably to a source and verify them. LLMs do not provide provenance or transparency as a feature at all. Instead, they are treated like black boxes. AI providers do not elaborate on how the models are trained or built, and act like it is mysterious or even surprising when a model reaches a conclusion deemed "problematic".

This lack of transparency can be deadly. LLMs are prone to hallucinations — something that cannot be mitigated no matter how many guardrails you try to add. It is simply a consequence of the mathematics behind how these models operate. Models are also more likely to make mistakes when handling data that is sparsely represented during training. Under-represented groups of people, including women, LGBT individuals, and ethnic minorities, are therefore more likely to be negatively impacted by these hallucinations.

Every solution to this problem so far has been inadequate — something slapped on after the fact. Prompt and context engineering will never be enough. In fact, this problem may be intractable no matter what kind of model you are using. This has been a challenge in data science since its inception.

However, it may be possible to make these models transparent and auditable. By understanding the mathematics behind how models work, we can build a new framework from the ground up and incorporate provenance at the deepest level. This is what Project Lambert is trying to achieve.

## The Math
This project was initially inspired by the work of mathematician Eric Hehner. Hehner developed a unique mathematical notation system that unifies arithmetic and logic — a [Unified Algebra](https://www.cs.utoronto.ca/~hehner/UA.pdf). The idea that math and logic are two sides of the same coin means we can theoretically build an AI not based on statistics and probability, but on logic itself. By incorporating logic at the model's deepest layer, we can make it easier to understand on an intuitive and interpretable level how a model reaches a decision or processes data.

We also take inspiration from [Tensor Logic](https://arxiv.org/pdf/2510.12269), a framework for creating an AI programming language proposed by Pedro Domingos. To summarize Domingos' paper: every operation an AI model performs can be reduced to the same core equation — Einstein summation (tensor contraction):

$$R[x,z] = \vee\langle y \cdot A[x,y] \wedge B[y,z]\rangle$$

For each output pair (x, z), this finds the best intermediate "witness" y by combining relations A and B through logical conjunction (∧) and existential quantification (∨).

This equation can be expressed using different algebraic structures called semirings. We use the logical semiring — Disjunction (∨ = max) and Conjunction (∧ = min) — rather than the standard arithmetic semiring. This makes the model's internal logic semantically transparent and directly interpretable.

This also eliminates a technical overhead present in Domingos's original formulation. Standard Tensor Logic operates on $\{0, 1\}$ and must apply a Heaviside step function after each join to convert continuous sums back to Boolean values: $A[x,z] = H\left(\sum_y S[x,y] \cdot P[y,z]\right)$. In UA, using $\top/\bot = \pm\infty$ with min/max, Boolean operations are closed by construction — the max or min of values from $\{-\infty, +\infty\}$ is always in $\{-\infty, +\infty\}$. No thresholding is needed. This means making the model "soft" (continuous, learnable) is not a separate mode switch; it is the same framework with a temperature parameter controlling how sharp the min/max operations are.

### Mathematical Foundations

#### **Fuzzy relational composition:** 
The core inference step is a smooth approximation of [Zadeh's max-min relational composition (1971)](https://www.sciencedirect.com/science/article/pii/0020025571900165):

$$R[x,z] = \text{SmoothMax}_y\bigl(\text{SmoothMin}(A[x,y],\; B[y,z])\bigr)$$

The underlying algebraic structure $([0,1], \max, \min)$ is a valid distributive lattice and semiring, with over 50 years of theoretical grounding in fuzzy set theory and relational algebra [(Sanchez, 1976)](https://www.sciencedirect.com/science/article/pii/S0019995876904460).

#### **Smoothing:** 
SmoothMin and SmoothMax are implemented via LogSumExp, a well-established smoothing technique [(Nesterov, 2005)](https://link.springer.com/article/10.1007/s10107-004-0552-5). The approximation error is controllable:

$$\left|\text{SmoothMax}(\mathbf{x}) - \max(\mathbf{x})\right| \leq \frac{\log n}{\alpha}$$

where $\alpha$ is the temperature parameter. A known consequence is that smoothing breaks the distributivity and idempotency of the exact semiring; algebraic guarantees of the crisp max-min semiring do not transfer, and error accumulates with composition depth.

#### **Witness tracking and provenance:** 
During each Join, all intermediate indices $y =  \min(A[x,y], B[y,z])$ — the entities that "witness" the inference — above a contribution threshold are recorded along with their contribution score. [Green, Karvounarakis & Tannen (2007, PODS)](https://dl.acm.org/doi/10.1145/1265530.1265535) proved that query annotations propagate through relational algebra via semiring operations. Under the fuzzy semiring, Lambert's composition is exactly relational composition with provenance: the witness is the provenance certificate. These recorded witnesses are then used to reconstruct a human-readable proof tree tracing exactly which intermediate entities justified each conclusion — the concrete mechanism behind the provenance goal described above. 

#### **Fixed-point convergence:** 
The iteration $A_{n+1} = \text{Join}(A_n, B)$ has guaranteed fixed-point existence via the [Knaster-Tarski theorem (1955)](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-5/issue-2/A-lattice-theoretical-fixpoint-theorem-and-its-applications/pjm/1103044538.full): any monotone function on a complete lattice has fixed points, and $([0,1]^N, \leq)$ is a complete lattice. At positive temperature, SmoothMax/SmoothMin are locally contractive, giving geometric convergence near a fixed point:

$$\lVert x_n - x^* \rVert \leq q^n \lVert x_0 - x^* \rVert, \quad q < 1$$

Whether the map is globally contractive — and thus whether the iteration always converges to the same fixed point regardless of initialization — is an open question.

#### **Free energy:** 
The convergence diagnostic:    $$F = \sum (A_n - A)^2$$   measures the squared change between successive iterates, analogous to monitoring a Lyapunov function ([Hopfield, 1982](https://www.pnas.org/doi/10.1073/pnas.79.8.2554);  [Ramsauer et al., 2021](https://arxiv.org/abs/2008.02217)). It relates to variational free energy [(Friston et al., 2010)](https://www.nature.com/articles/nrn2787) under restrictive assumptions but omits precision weighting, an explicit generative model, and the entropy term. It is best understood as a fixed-point residual rather than a formal evidence lower bound.

#### **Temperature schedule:** 
The adaptive cooling formula:     $$T = \frac{-E}{N \cdot \overline{\log A}}$$   is derived by analogy from thermodynamics ($F = U - TS$). It produces qualitatively correct behavior — temperature drops as $E \to 0$, hardening soft operations toward crisp boolean logic — but uses a non-standard entropy definition and has boundary singularities when any $A_i = 0$ or all $A_i = 1$.

The temperature parameter controls a spectrum between two reasoning modes: at $T \to 0$, SmoothMax approaches hard max and SmoothMin approaches hard min, recovering exact crisp Boolean logic; at $T \gg 0$, operations become nearly linear and all evidence contributes proportionally, enabling analogical reasoning. The system can operate anywhere on this spectrum without changing the underlying framework.

#### **SVD/Tucker decomposition:** 
The most significant known gap. Standard SVD and Tucker decomposition rely on additive inverses, multiplicative inverses, and inner products that have no natural counterparts in the max-min semiring. SVD's optimality guarantee (minimum Frobenius-norm reconstruction error) does not transfer, as the Frobenius norm is not the natural metric for max-min algebra. Tropical analogues of matrix decomposition exist but have fundamentally different properties [(Develin, Santos & Sturmfels, 2005)](https://arxiv.org/abs/math/0312114); a proper lattice-based factorization would be more principled and is left as future work.

These limitations have been reviewed and are considered acceptable for the current stage of the project.

## Status
The core components prescribed by Domingos, grounded in Unified Algebra rather than typical statistics, have been assembled and debugged. The UA-based formulation is significantly cleaner and more interpretable than a statistical equivalent would be.

Remaining work:
- Build a full end-to-end pipeline that traces a complete logic path for a moderately complex task or query
- Assess and simplify the mathematics of individual components where possible
- Develop proof-guided training: using witness records from inference to update only the weights that actually contributed to a conclusion, rather than all parameters

