"""
Recursive data types and their interpreters.

Declares the structure of recursive algebraic data (Case, Functor) and
provides the generic Interpreter that drives algebra folds (catamorphisms)
and coalgebra unfolds (anamorphisms) over those structures.

Key abstractions:
  Case       — one variant of a recursive sum type (recursive children + data payload)
  Functor    — a collection of named Cases defining an endofunctor F
  UnfoldStep  — the output of a coalgebra step: case name, payload, next states, and optional output
  Interpreter — drives run_algebra (tree fold) and run_coalgebra (stream unfold)

Depends on: nothing (leaf module in the engine stack)

References
----------
Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
theory of all architectures. ICML 2024.  cite{gavranovic2024b}

Tarski, A. (1955). A lattice-theoretical fixpoint theorem and its applications.
*Pacific Journal of Mathematics*, 5(2), 285–309.  cite{tarski1955}
"""

from dataclasses import dataclass
from typing import Callable

try:
    import numpy as _np
except ImportError:
    _np = None

# Sentinel objects for coalgebra signaling.
# NO_OUTPUT marks cases that transition state without emitting output.
# _EXHAUSTED signals that the token iterator has been fully consumed.
NO_OUTPUT = object()
_EXHAUSTED = object()


# ---------------------------------------------------------------------------
# Functor declarations
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """One variant of a recursive sum type (endofunctor case).

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """

    name:      str
    recursive: int
    data:      int
    output:    int = 0

    def __post_init__(self):
        if self.recursive < 0:
            raise ValueError("recursive must be >= 0")
        if self.data < 0:
            raise ValueError("data must be >= 0")
        if self.output not in (0, 1):
            raise ValueError("output must be 0 or 1")


@dataclass
class Functor:
    """Collection of named Cases defining an endofunctor F.

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """

    cases: list

    def __post_init__(self):
        names = [c.name for c in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate case names in Functor")
        self._by_name = {c.name: c for c in self.cases}

    def __getitem__(self, name: str) -> Case:
        if name not in self._by_name:
            raise KeyError(f"Unknown case '{name}' for this Functor")
        return self._by_name[name]


@dataclass
class UnfoldStep:
    """Output of a coalgebra step: case name, payload, next states, optional output.

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """

    case_name:   str
    payload:     list
    next_states: list
    output:      object = NO_OUTPUT

    @classmethod
    def silent(cls, case_name: str, payload: list, next_states: list) -> 'UnfoldStep':
        """Create a step that transitions state without emitting output."""
        return cls(case_name=case_name, payload=payload, next_states=next_states, output=NO_OUTPUT)


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class Accumulator:
    """Manages per-morphism state accumulation across coalgebra steps."""

    def __init__(self, accumulate_specs: dict | None, backend):
        self.specs = accumulate_specs or {}
        self.backend = backend
        self._store: dict = {}

    def update(self, payload: list) -> None:
        if len(self.specs) > len(payload):
            raise ValueError(
                f"Accumulator has {len(self.specs)} spec(s) but payload has "
                f"{len(payload)} item(s). Specs: {list(self.specs.keys())}"
            )
        for i, (name, (mode, fields)) in enumerate(self.specs.items()):
            item = payload[i]
            if fields is not None:
                if name not in self._store:
                    self._store[name] = {f: item[f] for f in fields}
                else:
                    self._store[name] = {
                        f: self.backend.concatenate(
                            [self._store[name][f], item[f]], axis=0
                        )
                        for f in fields
                    }
            else:
                if name not in self._store:
                    self._store[name] = item
                else:
                    self._store[name] = self.backend.concatenate(
                        [self._store[name], item], axis=0
                    )

    def inject_into_params(self, params: dict) -> dict:
        if self._store:
            return {**params, 'accumulated': self._store}
        return params

    def merge_into_state(self, state: dict) -> dict:
        if self._store and isinstance(state, dict):
            return {**state, **self._store}
        return state

    @property
    def active(self) -> bool:
        return bool(self.specs)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """Generic (co)algebra interpreter.

    Drives declared Functors through algebra folds (run_algebra) and
    coalgebra unfolds (run_coalgebra). The coalgebra runner supports the
    linear single-successor subset: cases with recursive=1.

    References
    ----------
    Gavranović, B. et al. (2024). Position: Categorical deep learning is an algebraic
    theory of all architectures. ICML 2024.  cite{gavranovic2024b}
    """

    def __init__(self, functor: Functor, cell: Callable, params, temp=0.0):
        self.functor = functor
        self.cell    = cell
        self.params  = params
        self.temp    = temp

    def run_algebra(self, data, decompose: Callable):
        """Fold a tree via catamorphism (initial algebra evaluation).

        References
        ----------
        Gavranović, B. et al. (2024). Position: Categorical deep learning is an
        algebraic theory of all architectures. ICML 2024.  cite{gavranovic2024b}
        """
        case_name, payload, children = decompose(data)
        case = self.functor[case_name]

        if len(payload) != case.data:
            raise ValueError(
                f"Case '{case_name}' declared data={case.data}, "
                f"got {len(payload)}"
            )
        if len(children) != case.recursive:
            raise ValueError(
                f"Case '{case_name}' declared recursive={case.recursive}, "
                f"got {len(children)}"
            )

        child_results = [self.run_algebra(c, decompose) for c in children]
        return self.cell(case_name, payload, child_results, self.params, self.temp)

    def run_coalgebra(self, state, token_iter=None, stop: Callable = None,
                      accumulate_specs: dict | None = None,
                      backend=None):
        """
        Generic coalgebra runner for the linear single-successor subset:
        functors where each active case has recursive=1.

        token_iter      : iterable of input tokens, or None
        stop            : callable(step, state, outputs) -> bool, or None
                          returns True to halt
        accumulate_specs : dict mapping leg_name -> mode ('cat'), or None

        If token_iter is exhausted the run halts.
        If stop is None and token_iter is None the caller must ensure
        termination through stop.

        References
        ----------
        Gavranović, B. et al. (2024). Position: Categorical deep learning is an
        algebraic theory of all architectures. ICML 2024.  cite{gavranovic2024b}
        """
        if backend is None:
            if _np is None:
                raise ImportError("numpy is required when no backend is provided")
            from .runtime import NUMPY_BACKEND
            backend = NUMPY_BACKEND
        outputs = []
        tokens  = iter(token_iter) if token_iter is not None else None
        step    = 0
        acc     = Accumulator(accumulate_specs, backend)

        while True:
            token = None
            if tokens is not None:
                token = next(tokens, _EXHAUSTED)
                if token is _EXHAUSTED:
                    break

            params = acc.inject_into_params(self.params)
            result = self.cell(state, token, params, self.temp)
            case   = self.functor[result.case_name]

            if case.recursive != 1:
                raise ValueError(
                    f"run_coalgebra supports only single-successor cases "
                    f"(recursive=1); case '{result.case_name}' has "
                    f"recursive={case.recursive}"
                )
            if len(result.next_states) != case.recursive:
                raise ValueError(
                    f"Case '{result.case_name}' declared recursive={case.recursive}, "
                    f"adapter returned {len(result.next_states)} next_states"
                )
            if len(result.payload) != case.data:
                raise ValueError(
                    f"Case '{result.case_name}' declared data={case.data}, "
                    f"adapter returned {len(result.payload)} payload items"
                )
            has_output = result.output is not NO_OUTPUT
            if has_output != (case.output > 0):
                raise ValueError(
                    f"Case '{result.case_name}' declared output={case.output}, "
                    f"but adapter output presence does not match"
                )

            state = result.next_states[0]
            if has_output:
                outputs.append(result.output)

            # Accumulate morphism outputs from payload
            if acc.active and result.payload:
                acc.update(result.payload)

            # Feed accumulated values forward into state so the next step sees them
            state = acc.merge_into_state(state)

            step += 1
            if stop is not None and stop(step, state, outputs):
                break

        return outputs, state
