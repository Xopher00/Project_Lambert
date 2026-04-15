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
    case_name:   str
    payload:     list
    next_states: list
    output:      object = NO_OUTPUT

    @classmethod
    def silent(cls, case_name: str, payload: list, next_states: list) -> 'UnfoldStep':
        """Create a step that transitions state without emitting output."""
        return cls(case_name=case_name, payload=payload, next_states=next_states, output=NO_OUTPUT)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """Generic (co)algebra interpreter.

    Drives declared Functors through algebra folds (run_algebra) and
    coalgebra unfolds (run_coalgebra). The coalgebra runner supports the
    linear single-successor subset: cases with recursive=1.
    """

    def __init__(self, functor: Functor, cell: Callable, params, temp=0.0):
        self.functor = functor
        self.cell    = cell
        self.params  = params
        self.temp    = temp

    def run_algebra(self, data, decompose: Callable):
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
                      accumulate_legs: dict | None = None):
        """
        Generic coalgebra runner for the linear single-successor subset:
        functors where each active case has recursive=1.

        token_iter      : iterable of input tokens, or None
        stop            : callable(step, state, outputs) -> bool, or None
                          returns True to halt
        accumulate_legs : dict mapping leg_name -> mode ('cat'), or None

        If token_iter is exhausted the run halts.
        If stop is None and token_iter is None the caller must ensure
        termination through stop.
        """
        outputs = []
        tokens  = iter(token_iter) if token_iter is not None else None
        step    = 0
        accumulated = {}

        while True:
            token = None
            if tokens is not None:
                token = next(tokens, _EXHAUSTED)
                if token is _EXHAUSTED:
                    break

            if accumulate_legs:
                self.params['accumulated'] = accumulated

            result = self.cell(state, token, self.params, self.temp)
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

            # Accumulate leg outputs from payload
            if accumulate_legs and result.payload:
                if _np is None:
                    raise ImportError("numpy is required for accumulate_legs")
                for i, (leg_name, (mode, fields)) in enumerate(accumulate_legs.items()):
                    if i < len(result.payload) and mode == 'cat':
                        if fields is not None:
                            # Field-level accumulate: payload item is a dict, concat specific fields
                            new = result.payload[i]
                            if leg_name not in accumulated:
                                accumulated[leg_name] = {}
                            for field in fields:
                                old_field = accumulated[leg_name].get(field)
                                new_field = new[field] if isinstance(new, dict) else new
                                accumulated[leg_name][field] = (
                                    _np.concatenate([old_field, new_field]) if old_field is not None else new_field
                                )
                        else:
                            # Whole-sort accumulate (existing behavior)
                            old = accumulated.get(leg_name)
                            new = result.payload[i]
                            accumulated[leg_name] = (
                                _np.concatenate([old, new]) if old is not None else new
                            )

            step += 1
            if stop is not None and stop(step, state, outputs):
                break

        return outputs, state
