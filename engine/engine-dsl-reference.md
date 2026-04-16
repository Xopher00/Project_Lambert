# Engine DSL Reference

## Overview

Write your architecture as a short text program, pass it to `compile()` with your NumPy/PyTorch ops, and get back callable layers and a streaming interpreter. The DSL handles sequential composition, parallel branches, skip connections, and autoregressive state — you supply the math functions.

Formally, the DSL expresses V-enriched functor compositions over arbitrary semirings, but you don't need that to use it.

You write a DSL string, pass it to `compile()` along with a Python namespace, and get back an `ArchDef` — a compiled object whose paths are callable and whose arch declarations support both batch (algebra) and streaming (coalgebra) evaluation.

## Quickstart

A 2-layer GPT-style transformer in ~20 lines of DSL (from `engine/gpt_demo.ipynb`):

```
semiring attn:
    contract = ops.join_fn       # how tensors combine — standard einsum passthrough here
    compiler = ops.compile_einsum

sort model, h, scores, probs    # tensor sorts — used for composition checking

morphism proj[prefix] : model -> h     via "sd,dh->sh"  op ops.proj
# ↑ parameterized linear projection: input sort "model", output sort "h"
#   proj[q] reads y['q_W']/y['q_b']; proj[k] reads y['k_W']/y['k_b'] — same op, different weight keys

morphism ln[prefix]   : model -> model via "sd->sd"     op ops.ln
morphism score        : h -> scores    via "sh,th->st"  op ops.score
morphism normalize    : scores -> probs via "st->st"    op ops.softmax  arity unary
morphism mix          : probs -> h     via "st,th->sh"  op ops.mix

fan kv = proj[k] & proj[v]  merge dict
# ↑ parallel branch: runs proj[k] and proj[v] on the same input, merges into context dict y

path read = proj[q] score normalize mix proj_out[o]
# ↑ sequential pipeline: Q-projection -> attention scores -> softmax -> weighted sum -> output projection

path attn = ln[1] [kv] read  residual
# ↑ full attention sub-layer: layer norm, then inject K/V into context, then read path, then skip connection

arch Transformer:
    cases:
        input: leaf  data=1  cell=identity
        block: node  data=1  morphisms=attn ffn  iterate=layers
        # ↑ one block = attn + ffn; iterate=layers stacks N blocks from a layer list
```

```python
arch = engine.compile(DSL_SOURCE, {'ops': ops})
interp = arch.interpreter('Transformer', params={}, temp=0.0)
result = interp.run_algebra_layers(x0, layers=layer_weights)   # batch forward pass
```

## Architecture

| File | Role |
|------|------|
| `engine/__init__.py` | Public API re-exports: `compile`, `parse`, `load`, `ArchDef` |
| `engine/decl.py` | AST dataclasses (pure data, no logic) |
| `engine/parser.py` | Text → `DSLSource` AST |
| `engine/compiler.py` | `DSLSource` + namespace → `ArchDef` |
| `engine/runtime.py` | `MorphismSpec`, `compile_morphism`, `chain`, `fan`, `check_sorts` |
| `engine/functor.py` | `Case`, `Functor`, `Interpreter`, `UnfoldStep` |
| `engine/arch.py` | `ArchDef`, `ArchInterpreter` |
| `engine/sorts.py` | Hydra type/term bridge |
| `engine/primitives.py` | Hydra primitive registration |

**Data flow**: DSL text → `parse()` → `DSLSource` (typed AST) → `compile()` → `ArchDef` (compiled paths + arch interpreters). Every compiled morphism is a `(x, y, temp) -> result` callable. Paths are chains or augmented chains of those callables. Arches fold or unfold paths over recursive data structures.

## Key Abstractions

### `SemiringDecl` — the algebraic structure (your choice of tensor reduction operations)

Declares the quantale (the algebraic structure that determines how tensors combine — your choice of sum-product, max-min, tropical, etc.) V = (V, ⊗, k) that morphisms compose over. The `contract` function is V's binary (or ternary) product operation; its signature must be `(compiled_eq, x, y, temp=0.0) -> tensor`. The optional `compiler` preprocesses equation strings before they reach `contract`.

```python
@dataclass
class SemiringDecl:
    name:     str
    contract: str        # dotted name resolving to (compiled_eq, x, y, temp) -> tensor
    compiler: str | None # dotted name resolving to equation_str -> compiled_form
    arity:    str        # 'binary' (default) or 'ternary'
```

### `MorphismDecl` / `MorphismSpec` — a single V-functor (a structure-preserving layer operation)

`MorphismDecl` is the parsed AST node. `MorphismSpec` (in `runtime.py`) is the resolved runtime version with actual Python callables. After `compile()`, each morphism is stored in `ArchDef.paths` as a `(x, y, temp) -> result` callable.

Key fields: `src_sort`, `tgt_sort` (for sort-checking), `equation` (passed to `contract`), `arity` (changes how arguments are passed).

### `PathDecl` / path callable — sequential composition

A named sequential composition of morphisms. The path callable threads output to input: `f(g(h(x, y, temp), y, temp), y, temp)`. With `residual`, the output is added back to the input: `result + x`. With `normed <m>`, a normalization morphism is applied after.

Augment steps `[fan_name]` in a path enrich `y` (the context dict) without transforming `x`:

```
path attn = ln[1] [kv] read  residual
#                 ^^^-- fan 'kv' output merged into y; ln[1] and read still transform x
```

### `FanDecl` / fan callable — parallel branch-out

Runs all branches on the same `(x, y, temp)` and merges results. Default merge is `dict` (returns `{branch_name: result, ...}`). Other options: `meet` (elementwise min), `join` (elementwise max), or a dotted name to a custom Python function.

### `ArchDecl` / `ArchInterpreter` — recursive endofunctor

`ArchDecl` declares an endofunctor F via `cases:`. Cases carry `recursive` (number of recursive children) and `data` (number of payload slots). The `iterate=<name>` field marks cases that stack repeatedly in a layer loop.

`ArchInterpreter` (returned by `ArchDef.interpreter()`) has:
- `run_algebra(tree, decompose)` — catamorphism (fold): evaluates bottom-up from a pre-built tree
- `run_algebra_layers(x0, layers, extras=None)` — builds the tree from an `iterate=` group and folds it
- `run_coalgebra(state, token_iter, stop)` — anamorphism (unfold): streams token-by-token

### `ArchDef` — compiled output

The result of `compile()`. Primary public fields:

- `paths: dict[str, Callable]` — all morphisms, paths, and fans by name; each is `(x, y, temp) -> result`
- `morphism_semiring: dict[str, str]` — maps morphism name to semiring name

Methods: `explain(name)`, `trace(name, x, y, temp)`, `loss(name, x, y, temp)`, `interpreter(name, params, temp)`.

## Public API Reference

### `compile(source, namespace) -> ArchDef`

Compile DSL text (or a pre-parsed `DSLSource`) into an `ArchDef`.

- `source` — DSL string, `Path` to a `.ua` file, or `DSLSource` from `parse()`
- `namespace` — `dict` mapping dotted names used in the DSL to Python objects. Example: `{'ops': my_ops_module, 'numpy': np}`
- Returns `ArchDef`

```python
from engine import compile
arch = compile(DSL_SOURCE, {'ops': ops})
```

### `parse(source) -> DSLSource`

Parse DSL text into an AST without compiling. Useful for inspection or validation.

```python
from engine import parse
ast = parse(DSL_SOURCE)
print(ast.morphisms)   # list[MorphismDecl]
print(ast.paths)       # list[PathDecl]
```

### `load(path, namespace) -> ArchDef`

Convenience wrapper: read a `.ua` file and compile it.

```python
from engine import load
arch = load("model.ua", {'ops': ops})
```

### `ArchDef.interpreter(name, params=None, temp=0.0) -> ArchInterpreter`

Create an interpreter for a named `arch` declaration.

- `name` — arch name as declared in the DSL
- `params` — optional dict merged into interpreter params (available in cell functions as `params`)
- `temp` — temperature parameter forwarded to all morphism calls

```python
interp = arch.interpreter('Transformer', params={}, temp=0.0)
```

### `ArchInterpreter.run_algebra_layers(x0, layers, extras=None)`

Run the algebra fold using a flat list of layer payloads (for `iterate=`-based arches).

- `x0` — initial input tensor (leaf payload)
- `layers` — list of dicts, one per layer; each dict is passed as `y` when evaluating that layer's cases
- `extras` — optional dict merged into every layer payload

Returns the final folded value.

### `ArchInterpreter.run_coalgebra(state, token_iter=None, stop=None, convergence_threshold=1e-6)`

Run the coalgebra unfold (streaming mode).

- `state` — initial state dict (carries weights, context, etc.)
- `token_iter` — iterable of input tokens; halts when exhausted
- `stop` — `callable(step, state, outputs) -> bool`; return `True` to halt early
- Returns `(outputs, final_state)` where `outputs` is a list of emitted values

### `ArchDef.explain(name) -> str`

Human-readable path description: lists each morphism in order with its equation string.

### `ArchDef.trace(name, x, y, temp=0.0) -> list[tuple]`

Step-by-step execution. Returns `[(name, equation, shape, value), ...]` rows, one per morphism plus an `"input"` row.

## DSL Syntax

### `semiring`

```
semiring <name>:
    contract = <dotted.name>
    compiler = <dotted.name>    # optional
    arity = binary | ternary    # optional, default binary
```

The `contract` function must have signature `(compiled_eq, x, y, temp=0.0) -> tensor`. If `compiler` is set, the equation string is passed through it before reaching `contract`.

### `sort`

```
sort i, j, k           # opaque sorts (names only)
sort model(Wq: matrix) # structured sort with fields
```

Sorts are used only for composition-checking — the DSL does not enforce tensor shapes, only that sort labels match at path junctions.

### `morphism` / `leg`

The keywords `morphism` and `leg` are interchangeable.

**Compact form** (most common):
```
morphism <name> : <src> -> <tgt>  "<equation>"  [clauses]
```

**Explicit form** (with `via`):
```
morphism <name> : <src> -> <tgt>  via "<equation>"  [clauses]
```

**Multi-line** (continuation lines indented):
```
morphism <name> : <src> -> <tgt>
    "<equation>"
    <dotted.op>
    arity unary
```

**Template morphism** (parameterized with `[param]`):
```
morphism proj[prefix] : model -> h  via "sd,dh->sh"  op ops.proj
```
The `prefix` parameter is bound at instantiation time. Instantiated with `proj[q]`, `proj[k]` etc. in paths or fans.

**Clauses** (separated by 2+ spaces):

| Clause | Meaning |
|--------|---------|
| `using <semiring>` | Assign to named semiring group |
| `op <dotted.name>` | Override the contract (bare dotted name also works) |
| `transform <dotted.name>` | Pre-process `(x, y)` before passing to `op` |
| `compiler <dotted.name>` | Per-morphism equation compiler |
| `arity unary\|binary\|pointwise\|ternary` | Argument passing mode |
| `bridge` | Mark as bridge morphism (allowed in cross-semiring paths) |
| `accumulate cat [on <f1>, <f2>]` | Coalgebra state accumulation |

**Arity modes** determine how arguments reach `op`:
- `binary` (default): `op(eq, x, y, temp=temp)` — standard two-argument form
- `unary`: `op(eq, x, temp=temp)` — `y` is ignored
- `pointwise`: `op(eq, x, y, temp=temp)` — same call but semantically element-wise
- `ternary`: `op(eq, x, y[0], y[1], temp=temp)` — destructures `y` into two args

### `path`

```
path <name> = <m1> [<fan>] <m2> ...  [residual]  [normed <morphism>]
```

- Morphism names are composed left-to-right: `m1` output → `m2` input
- `[fan_name]` augment: runs the fan, merges results into `y` (context), does not transform `x`
- `residual`: adds the input `x` to the path output: `result + x`
- `normed <m>`: applies morphism `<m>` to the result (applied after residual if both present)

### `fan`

```
fan <name> = <m1> & <m2> & ...  [merge dict|meet|join|<dotted.name>]
```

Runs all branches in parallel on the same `(x, y, temp)`. Default merge `dict` returns `{branch_name: result, ...}`. `meet`/`join` apply elementwise min/max across all branches.

### `arch`

```
arch <name>:
    cases:                                       # also accepts 'algebra:' as alias
        <case_name>: leaf  data=<int>  [cell=<dotted.name>|identity]
        <case_name>: node  data=<int>  morphisms=<m1> <m2>  [iterate=<group>]
    state:
        <field>: <type>
    step:
        enter = <morphism_name>
        emit  = <morphism_name>
        compute = <morphism_name>       # optional explicit compute path
    observer:
        convergence = <path_name>
        loss = <path_name>
```

**`cases:`** declares the endofunctor F. `leaf` is shorthand for `recursive=0`; `node` for `recursive=1`. Each case can specify either `cell=<fn>` (explicit Python cell) or `morphisms=<m1> <m2>` (derived from DSL paths). `cell=identity` passes payload[0] through unchanged (used for leaf input nodes).

**`iterate=<group>`**: marks cases that repeat per layer. `run_algebra_layers(x0, layers)` builds a tree where the layer list drives the iterate group.

**`state:`**: declares the coalgebra state shape (field names → type names). Used for Hydra type generation; does not enforce Python types at runtime.

**`step:`**: configures the coalgebra anamorphism.
- `enter = <m>`: runs `m(token, state, temp)` to bind each input token into the state representation
- `emit = <m>`: runs `m(x, state, temp)` after the main computation; its output is collected
- `compute = <m>`: explicit path for the main computation (defaults to the first case's `morphisms`)

**`observer:`**: optional paths for monitoring.
- `convergence = <path>`: halts `run_coalgebra` when max-abs residual drops below threshold
- `loss = <path>`: accessible via `arch_def.loss(name, x, y, temp)`

### Comments and `coerce`

```
# This is a comment — stripped before parsing

coerce i -> j: 0.9       # declare sort coercion with grade [0.0, 1.0]
sort_threshold 0.8        # warn when coercion grade is below this
```

## How to Use the DSL

### Minimal working example

```python
import numpy as np
from engine import compile

def join_fn(eq, x, y=None, temp=0.0):
    # Max-min semiring: max over j of (x[j] * y[j,i])
    return np.minimum(x[:, None], y[None, :]).max(axis=0)

DSL = """
semiring join:
    contract = ops.join_fn

sort j, i

morphism realize   : j -> i  via "j,ji->i"
morphism propagate : i -> j  via "i,ij->j"

path attend = realize propagate
"""

arch = compile(DSL, {'ops': type('ns', (), {'join_fn': staticmethod(join_fn)})()})

# Each path is (x, y, temp) -> result
x = np.array([1.0, 0.5, 0.2])
W = np.eye(3)
result = arch.paths['attend'](x, W, temp=0.0)
```

### Batch forward pass with `run_algebra_layers`

```python
DSL = """
semiring s:
    contract = ops.contract
    compiler = ops.compile_einsum

sort model, h

morphism proj[prefix] : model -> h     via "sd,dh->sh"  op ops.proj
morphism proj_out[prefix] : h -> model via "sh,hd->sd"  op ops.proj

path forward = proj[q] proj_out[o]  residual

arch Encoder:
    cases:
        input: leaf  data=1  cell=identity
        block: node  data=1  morphisms=forward  iterate=layers
"""

arch = compile(DSL, {'ops': ops})
interp = arch.interpreter('Encoder', params={}, temp=0.0)

x0 = np.random.randn(seq_len, d_model)  # initial embeddings
layer_weights = [{'q_W': ..., 'q_b': ..., 'o_W': ..., 'o_b': ...}
                 for _ in range(num_layers)]

result = interp.run_algebra_layers(x0, layers=layer_weights)
```

### Streaming (coalgebra) via `step:`

```python
STREAM_DSL = """
# same morphisms as above, plus:
morphism embed   : model -> model  via "x"  op ops.embed
morphism unembed : model -> model  via "x"  op ops.unembed

arch Decoder:
    cases:
        input: leaf  data=1  cell=identity
        block: node  data=1  output=1  morphisms=forward  iterate=layers
    step:
        enter = embed
        emit  = unembed
"""

arch = compile(STREAM_DSL, {'ops': ops})
interp = arch.interpreter('Decoder', params={}, temp=0.0)

# State carries all weights + embeddings
state = {'tok_embed': tok_embed, **layer_weights[0]}

outputs, final_state = interp.run_coalgebra(
    state,
    token_iter=[3, 7, 1, 15],           # one token per step
    stop=lambda step, s, outs: step >= 4,
)
# outputs[i] is the emit result for step i
```

### Introspection

```python
# Human-readable path breakdown
print(arch.explain('attn'))
# Path: attn
# Normal form: ln[1] score normalize mix proj_out[o]
#   1. ln[1]  [sd->sd]
#   2. score  [sh,th->st]
#   ...

# Step-by-step trace with shapes
rows = arch.trace('attn', x, y, temp=0.0)
for name, eq, shape, val in rows:
    print(f"{name:20s}  {str(shape):20s}  {eq}")
```

## Common Patterns and Gotchas

**Morphism callable signature is always `(x, y, temp)`.**
`x` is the primary signal; `y` is a context dict (weights, kv-cache, etc.); `temp` is a scalar temperature. The `op` function receives `(compiled_eq, x, y, temp=temp)` for binary arity, but `x` and `y` are the same objects you pass to the path callable. To make `y` available as a plain dict inside `op`, just call `arch.paths['name'](x, y_dict, 0.0)`.

**Augment brackets enrich `y`, not `x`.**
`[kv]` in a path runs the fan `kv` and merges its results (a dict) into the current `y`. Subsequent morphisms in the path see the enriched `y`. The fan's output keys are the branch names (e.g. `'proj[k]'`, `'proj[v]'`).

**Template instantiation happens at compile time.**
Writing `proj[q]` in a path or fan auto-instantiates the template with `prefix='q'`. The curried callable `ops.proj(eq, x, y, prefix='q', temp=temp)` is registered under key `'proj[q]'` in `arch.paths`.

**`iterate=` reverses nesting order.**
Cases are declared in data-flow order (e.g. `attn` before `ffn`), but tree nesting reverses this so the algebra fold applies `attn` first, `ffn` second — matching normal evaluation order. You don't need to manage this; `run_algebra_layers` handles it.

**Cross-semiring paths require `bridge`.**
A path that spans two different semirings raises `ValueError` at compile time. Declare the bridging morphism with `bridge` in its clause list (`semiring` becomes `None`), and it can appear in paths that span two semiring groups.

**`functor` keyword removed.**
Use `arch <n>: cases:` instead. The old `functor` keyword raises `SyntaxError` with a migration message.

**`coalgebra:` sub-block removed.**
The unified `cases:` block serves both algebra and coalgebra. Use `step:` for coalgebra-specific configuration. Using `coalgebra:` raises `SyntaxError` with a migration message.

**`cases:` and `algebra:` are aliases.**
Both are accepted and treated identically. `cases:` is preferred.

**Sort validation only warns for undeclared sorts.**
If no `sort` declarations appear in the DSL, sort checking is skipped silently. Sort checking is only strict when sorts are declared and sorts don't match — it raises `TypeError` at compile time.

**`cell=` and `morphisms=` are mutually exclusive per case.**
Specifying both raises `ValueError` at compile time.
