# Compiler

Compiles DSL source text into executable architecture definitions. Takes a parsed
AST (from `parser.py`) plus a Python namespace and produces an `ArchDef`.

## Compilation phases

The compiler in `engine/compiler.py` runs five phases:

1. **Resolve semirings** — map `SemiringDecl` names to contract/compiler/arity callables
2. **Compile morphisms** — `MorphismDecl` → `MorphismSpec` → cached callable
3. **Compile fans** — branch callables + merge strategy from `FanDecl`
4. **Compile paths** — chain/augment/residual composition with sort validation from `PathDecl`
5. **Compile archs** — `Functor` + cell binding for algebra and coalgebra from `ArchDecl`

Each phase preserves the algebraic structure declared in the source: semiring
parametricity (Green et al. 2007), sort composition (Lawvere 1973), and
algebra/coalgebra duality (Gavranović et al. 2024).

## Semiring resolution and provenance

The `compile()` function in `engine/compiler.py` resolves semiring declarations
into callables. Each `MorphismDecl` is assigned to a semiring group; bridge
morphisms (semiring = None) cross semiring boundaries.

Green et al. (2007) Proposition 3.5 states that semiring homomorphisms commute with
all RA⁺ operators. The bridge morphism mechanism is the DSL's representation of this:
a bridge crosses from one semiring to another, and the compiler validates that the
crossing is declared (not implicit).

Dannert et al. (2021) extend provenance to fixpoint logic. The iterate combinator
(`iterate = layers` on `CaseDecl`) performs LFP semantics, and provenance flows
through each iteration by the compositionality theorem.

**References:**
- Green, T. J., Karvounarakis, G., & Tannen, V. (2007). Provenance semirings.
  PODS 2007, pp. 31–40.  cite{green2007}
- Dannert, K. M. et al. (2021). Semiring provenance for fixed-point logic.
  CSL 2021, LIPIcs vol. 183.  cite{dannert2021}

## Template instantiation — Parametric morphisms

`_resolve_template_instance()` in `engine/compiler.py` creates curried morphism
callables from template declarations. A template morphism like `ln[prefix]` is a
natural transformation indexed by a parameter — the DSL's concrete representation
of parametric morphisms (§3 of Gavranović et al. 2024).

The instantiation creates a new `MorphismSpec` with the parameter bound, registers
it in the compiled namespace, and assigns the parent's semiring group.

**Reference:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}

## Iterate groups — Catamorphism detection

`_detect_iterate_groups()` in `engine/compiler.py` identifies which `CaseDecl`
entries form iteration loops. Cases with `iterate = <name>` are grouped; the
compiler constructs the algebra tree via `ArchInterpreter._build_tree()` in
`engine/arch.py`.

This is the catamorphism (initial algebra evaluation) of §5 in Gavranović et al.
(2024). The iteration is also LFP in the sense of Dannert et al. (2021):
convergence when the semiring is absorptive, with provenance preserved through
each step.

**References:**
- Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
  theory of all architectures. ICML 2024.  cite{gavranovic2024b}
- Dannert, K. M. et al. (2021). Semiring provenance for fixed-point logic.
  CSL 2021, LIPIcs vol. 183.  cite{dannert2021}

## Residual wrapper — Coproduct universal map

`_build_residual_wrapper()` in `engine/compiler.py` wraps a path callable with
a residual connection (f(x) + x) and optional normalization. The residual is the
universal map from a coproduct in the sense of §4 of Gavranović et al. (2024).

## Equation compilation — Tensor logic

Each `MorphismDecl` carries an `equation` field (an einsum string). The compiler
resolves equations through the semiring's `compiler` callable via
`_compile_morphisms()` in `engine/compiler.py`.

Domingos (2025) shows that Datalog rules and neural network layers are both einsum
operations. The equation compilation step is this insight made executable: every
morphism is an einsum, compiled against the declared semiring's contract.

**Reference:**
- Domingos, P. (2025). Tensor logic: The language of AI.
  arXiv:2510.12269.  cite{domingos2025}
