"""
interpreter.py — Generic (co)algebra interpreter for Layer C.

Drives declared Functors through algebra folds (run_algebra) and
coalgebra unfolds (run_coalgebra). The coalgebra runner supports the
linear single-successor subset: cases with recursive=1.
"""

from typing import Callable

from engine.functor import Functor, _NO_OUTPUT

_EXHAUSTED = object()


class Interpreter:
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

    def run_coalgebra(self, state, token_iter=None, stop: Callable = None):
        """
        Generic coalgebra runner for the linear single-successor subset:
        functors where each active case has recursive=1.

        token_iter : iterable of input tokens, or None
        stop       : callable(step, state, outputs) -> bool, or None
                     returns True to halt

        If token_iter is exhausted the run halts.
        If stop is None and token_iter is None the caller must ensure
        termination through stop.
        """
        outputs = []
        tokens  = iter(token_iter) if token_iter is not None else None
        step    = 0

        while True:
            token = None
            if tokens is not None:
                token = next(tokens, _EXHAUSTED)
                if token is _EXHAUSTED:
                    break

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
            has_output = result.output is not _NO_OUTPUT
            if has_output != (case.output > 0):
                raise ValueError(
                    f"Case '{result.case_name}' declared output={case.output}, "
                    f"but adapter output presence does not match"
                )

            state = result.next_states[0]
            if has_output:
                outputs.append(result.output)

            step += 1
            if stop is not None and stop(step, state, outputs):
                break

        return outputs, state
